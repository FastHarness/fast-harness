"""Local, fail-closed arbitration of the two recovery-tail total deadlines.

A guard belongs to one physical launch, not an episode name reused by retries.
No model, simulator, environment, or third-party dependencies are used here.
"""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
import uuid

REVISION = 'tail-guard-1'
OWNERS = ('sim', 'controller')


def start_ticks(pid):
    value = Path(f'/proc/{int(pid)}/stat').read_text()
    fields = value[value.rfind(')') + 2:].split()
    if fields[0] in ('Z', 'X', 'x'):
        raise ProcessLookupError('Controller is no longer running')
    return fields[19]


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _path(value):
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError('Guard paths must be absolute and must not traverse symlinks')
    return path


def _regular(fd):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise ValueError('Guard file must be an unshared, locally owned regular file')


class TailGuard:
    def __init__(self, path):
        self.path = _path(path)
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        self._binding = None
        with self._locked() as state:
            self._binding = self._identity(state)

    @staticmethod
    def _identity(state):
        return tuple(state[k] for k in ('run_uuid', 'path', 'local_output', 'episode_number',
                                       'sim_attempt', 'revision', 'boot_id'))

    @classmethod
    def create(cls, path, *, local_output, episode_number, sim_attempt, revision):
        path, output = _path(path), _path(local_output)
        if revision != REVISION or any(type(n) is not int or n < 1 for n in (episode_number, sim_attempt)):
            raise ValueError('Invalid guard revision or episode/physical attempt')
        if not path.parent.is_dir() or output != path.parent / 'controller':
            raise ValueError('Guard must be placed in its launch directory beside controller output')
        lock_path = path.with_name(path.name + '.lock')
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = dict(schema_version=1, run_uuid=str(uuid.uuid4()), path=str(path),
                         local_output=str(output), episode_number=episode_number, sim_attempt=sim_attempt,
                         revision=revision, boot_id=_boot_id(), controller=None, deadlines={},
                         state='pending', passthrough_entry=None)
            # Exclusive initialization: even an orphan lock is never silently reused.
            out = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(out, 'w') as stream:
                json.dump(state, stream, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        return cls(path)

    def _write(self, state):
        fd, name = tempfile.mkstemp(prefix=self.path.name + '.', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(state, stream, allow_nan=False, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _validate(self, state):
        if (not isinstance(state, dict) or state.get('schema_version') != 1
                or state.get('revision') != REVISION or state.get('boot_id') != _boot_id()
                or state.get('path') != str(self.path)
                or _path(state['local_output']) != self.path.parent / 'controller'
                or any(type(state[k]) is not int or state[k] < 1 for k in ('episode_number', 'sim_attempt'))
                or state.get('state') not in ('pending', 'active', 'expired', 'rejected')):
            raise ValueError('Invalid or stale guard state')
        uuid.UUID(state['run_uuid'])
        if self._binding is not None and self._identity(state) != self._binding:
            raise ValueError('Guard launch binding changed')
        deadlines = state['deadlines']
        if not isinstance(deadlines, dict) or set(deadlines) - set(OWNERS):
            raise ValueError('Invalid deadline owners')
        for deadline in deadlines.values():
            if (not isinstance(deadline, dict)
                    or any(type(deadline[k]) not in (float, int) or not math.isfinite(deadline[k])
                           for k in ('started', 'seconds', 'at'))
                    or deadline['seconds'] <= 0 or deadline['started'] < 0
                    or deadline['at'] != deadline['started'] + deadline['seconds']):
                raise ValueError('Invalid deadline')
        controller = state['controller']
        if controller is not None and (not isinstance(controller, dict)
                or type(controller['pid']) is not int or controller['pid'] <= 0
                or not str(controller['start_ticks']).isdigit()):
            raise ValueError('Invalid controller identity')
        if state['state'] == 'active':
            entry = state['passthrough_entry']
            if (not controller or set(deadlines) != set(OWNERS) or not isinstance(entry, dict)
                    or type(entry['step']) is not int or entry['step'] < 0
                    or not isinstance(entry['reason'], str) or not entry['reason']
                    or not math.isfinite(entry['at'])
                    or any(not item['started'] <= entry['at'] < item['at'] for item in deadlines.values())):
                raise ValueError('Unproven passthrough entry')

    @contextmanager
    def _locked(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            _regular(fd)
            until = time.monotonic() + 2
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= until:
                        raise TimeoutError('Guard lock unavailable')
                    time.sleep(0.005)
            state = None
            try:
                source = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(source) as stream:
                    _regular(stream.fileno())
                    state = json.load(stream)
                self._validate(state)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                rejected = state if isinstance(state, dict) else {}
                rejected.update(state='rejected', rejection='invalid_state')
                self._write(rejected)
                raise ValueError('Invalid guard; timeout exemption refused') from error
            before = json.dumps(state, sort_keys=True)
            try:
                yield state
            finally:
                if json.dumps(state, sort_keys=True) != before:
                    self._write(state)
        finally:
            os.close(fd)

    @staticmethod
    def _reject(state, reason):
        if state['state'] != 'expired':
            state['state'] = 'rejected'
        state['rejection'] = reason
        raise ValueError('Guard refused: ' + reason)

    @staticmethod
    def _live_binding(state):
        controller = state['controller']
        try:
            return bool(controller and start_ticks(controller['pid']) == controller['start_ticks'])
        except (OSError, ValueError, IndexError):
            return False

    @classmethod
    def for_controller(cls, path, *, output, episode_number, sim_attempt):
        guard = cls(path)
        with guard._locked() as state:
            try:
                output = _path(output)
            except ValueError:
                guard._reject(state, 'invalid_output_path')
            if (str(output) != state['local_output'] or type(episode_number) is not int
                    or type(sim_attempt) is not int or episode_number != state['episode_number']
                    or sim_attempt != state['sim_attempt']):
                guard._reject(state, 'output_or_episode_mismatch')
            own = dict(pid=os.getpid(), start_ticks=start_ticks(os.getpid()))
            if state['controller'] is not None and state['controller'] != own:
                guard._reject(state, 'controller_already_bound')
            if state['state'] in ('expired', 'rejected'):
                raise TimeoutError('Guard is already closed')
            if set(state['deadlines']) != set(OWNERS):
                guard._reject(state, 'unregistered_deadlines')
            if state['state'] != 'active' and guard._past_deadline(state):
                raise TimeoutError('Controller bound after deadline')
            state['controller'] = own
        return guard

    @staticmethod
    def _past_deadline(state, now=None):
        now = time.monotonic() if now is None else now
        due = [owner for owner, item in state['deadlines'].items() if now >= item['at']]
        if due:
            state.update(state='expired', expired_owners=due, expired_at=now)
        return bool(due)

    def register_deadline(self, owner, seconds):
        with self._locked() as state:
            if owner not in OWNERS or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                self._reject(state, 'invalid_deadline')
            if state['state'] != 'pending' or owner in state['deadlines']:
                self._reject(state, 'duplicate_or_closed_deadline')
            now = time.monotonic()
            state['deadlines'][owner] = dict(started=now, seconds=seconds, at=now + seconds)

    def expired(self, owner):
        with self._locked() as state:
            if owner not in OWNERS or owner not in state['deadlines']:
                self._reject(state, 'unregistered_deadline')
            if state['state'] in ('expired', 'rejected'):
                return True
            if state['state'] == 'active':
                if not self._live_binding(state):
                    self._reject(state, 'controller_identity_lost')
                return False
            # Any real deadline closes activation, under the same lock as enter.
            return self._past_deadline(state)

    def enter(self, *, step, reason):
        with self._locked() as state:
            if state['state'] in ('expired', 'rejected'):
                raise TimeoutError('Recovery tail entry refused by closed guard')
            if (not self._live_binding(state) or state['controller']['pid'] != os.getpid()
                    or set(state['deadlines']) != set(OWNERS)):
                self._reject(state, 'unbound_controller_entry')
            if type(step) is not int or step < 0 or not isinstance(reason, str) or not reason:
                self._reject(state, 'invalid_entry')
            if state['state'] == 'active':
                entry = state['passthrough_entry']
                if (entry['step'], entry['reason']) != (step, reason):
                    self._reject(state, 'conflicting_entry')
                return
            now = time.monotonic()
            if self._past_deadline(state, now):
                raise TimeoutError('Recovery tail entered after total deadline')
            state.update(state='active', passthrough_entry=dict(step=step, reason=reason, at=now))

    def snapshot(self):
        with self._locked() as state:
            return json.loads(json.dumps(state))

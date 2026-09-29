import copy
import math
from numbers import Real
from pathlib import Path

from .types import HarnessControlError, HarnessError, Observation, Proposal


CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
ARMS = ('left', 'right')
JOINTS = tuple(i for i in range(14) if i not in (6, 13))
PUBLIC_FIELDS = (
    'episode_id', 'step_id', 'task', 'instruction', 'max_episode_steps',
    'remaining_steps', 'require_native_termination', 'current_state', 'images',
    'current_eef', 'robot_profile', 'observation_path', 'arrays_path',
    'request_id', 'decision', 'gate_policy', 'gate_instruction', 'sim_time_s',
    'student_eef_trajectory', 'action_diagnostics', 'fk_check', 'counters',
    'prediction_id', 'inference_index', 'inference_seconds', 'proposal_path',
    'request_path',
)
HISTORY_FIELDS = (
    'decision', 'start_tick', 'end_tick', 'executed_steps', 'source',
    'source_detail', 'selected_student_steps', 'discarded_student_steps',
    'executed_gripper_closed', 'prediction_id',
)
PREVIOUS_FIELDS = (
    'step_id', 'images', 'current_eef', 'current_state', 'executed_steps',
    'executed_gripper_closed', 'source_detail',
)


def _vector(values, size, label):
    try:
        values = tuple(values)
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
            raise ValueError('Expected numeric scalars, not nested arrays')
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise HarnessError(f'{label} must contain finite numbers') from error
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise HarnessError(f'{label} must contain {size} finite numbers')
    return result


def encode_features(payload):
    """Equal-size RGB views plus proprioception; appearance distance is not grasp verification."""
    state = _vector(payload.get('current_state', ()), 14, 'current_state')
    images = payload.get('images', ())
    try:
        paths = {image['camera']: image['path'] for image in images}
    except (TypeError, KeyError) as error:
        raise HarnessError('Expected camera/path image records') from error
    if any(camera not in paths for camera in CAMERAS):
        raise HarnessError('All three RoboDojo RGB cameras are required')
    try:
        from PIL import Image
    except ImportError as error:
        raise HarnessError('Pillow is required for RGB appearance features') from error
    features = []
    for camera in CAMERAS:
        try:
            with Image.open(Path(paths[camera])) as image:
                if image.mode not in ('RGB', 'RGBA'):
                    raise HarnessError(f'{camera}: expected RGB image, got {image.mode}')
                thumbnail = image.convert('RGB').resize((8, 8), Image.Resampling.BOX)
                features.extend(channel / 255.0 for pixel in thumbnail.getdata() for channel in pixel)
        except (OSError, TypeError, ValueError) as error:
            raise HarnessError(f'Cannot encode {camera}: {error}') from error
    features.extend(
        min(1.0, max(0.0, value if i in (6, 13) else (value + math.pi) / (2 * math.pi)))
        for i, value in enumerate(state)
    )
    if not all(math.isfinite(value) for value in features):
        raise HarnessError('Nonfinite observation features')
    return tuple(features)


def _public_payload(packet):
    result = {key: copy.deepcopy(packet[key]) for key in PUBLIC_FIELDS if key in packet}
    for name, fields in (('previous_result', HISTORY_FIELDS), ('previous_observation', PREVIOUS_FIELDS)):
        previous = packet.get(name)
        result[name] = ({key: copy.deepcopy(previous[key]) for key in fields if key in previous}
                        if isinstance(previous, dict) else None)
    return result


class RoboDojoBackend:
    def __init__(self, tools, *, feature_encoder=encode_features,
                 joint_step_threshold_rad=0.35, first_joint_step_threshold_rad=0.5):
        self.tools = tools
        self.feature_encoder = feature_encoder
        self.joint_step_threshold_rad = self._threshold(joint_step_threshold_rad)
        self.first_joint_step_threshold_rad = self._threshold(first_joint_step_threshold_rad)
        self.metadata = {
            'adapter': 'fast_harness.robodojo',
            'uses_external_tools': True,
            'upstream_labels_unchanged': True,
            'actual_source_record': 'engine JSONL; backend.last_source records the latest execution attempt',
            'automatic_review': 'Harness automatic execution; no GPT visual review',
            'features': 'Three equally sized 8x8 RGB views in [0,1] and 14D normalized robot state',
            'feature_scope': 'Lightweight appearance distance, not grasp or task-completion verification',
            'no_reset_or_replay': True,
        }
        self.last_source = None
        self._started = self._live = self._done = self._failed = self._closed = False
        self._observation = None
        self._proposal = None
        self._request_ids = set()
        self._proposal_paths = set()

    @staticmethod
    def _threshold(value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Joint jump thresholds must be positive and finite')
        return value

    def _arguments(self, name):
        call = self.tools.next_call()
        if not isinstance(call, dict) or call.get('tool') != name:
            raise HarnessError(f'Expected next_call {name}, got {call!r}')
        return {key: value for key, value in call.items() if key != 'tool'}

    def _call(self, name, arguments):
        try:
            return getattr(self.tools, name)(**arguments)
        except Exception as error:
            self._failed = True
            raise HarnessError(f'RoboDojo {name} failed; no retry: {error}') from error

    def _active(self):
        if not self._live or self._done or self._failed or self._closed or self.tools.phase == 'done':
            raise HarnessError('No live, usable RoboDojo episode; reset/replay is not supported')

    def _current(self, observation):
        self._active()
        if observation is not self._observation:
            raise HarnessError('Stale or foreign observation')
        if (self.tools.episode, self.tools.tick) != (observation.episode_id, observation.step):
            raise HarnessError('Current observation does not match external episode/step')

    def _current_proposal(self, proposal):
        if proposal is not self._proposal or proposal is None:
            raise HarnessError('Stale, foreign, or already consumed proposal')
        self._current(proposal.observation)
        request = self.tools.request
        if ((request.get('request_id'), request.get('episode_id'), request.get('step_id')) !=
                (proposal.request_id, proposal.observation.episode_id, proposal.observation.step)):
            raise HarnessError('Proposal does not match current external request')
        if str(self.tools.proposal_path) != proposal.payload.get('proposal_path'):
            raise HarnessError('Proposal path does not match current external request')

    def _observe(self, packet):
        try:
            episode, step = packet['episode_id'], packet['step_id']
            if not isinstance(episode, str) or not episode or type(step) is not int or step < 0:
                raise HarnessError('Invalid observation identity')
            if (episode, step) != (self.tools.episode, self.tools.tick):
                raise HarnessError('Observation packet does not match external episode/step')
            if self._observation is not None:
                if episode != self._observation.episode_id or step < self._observation.step:
                    raise HarnessError('Episode changed or observation rewound')
            result = packet.get('result')
            previous = packet.get('previous_result') or {}
            terminal = previous.get('terminal') is True
            success = previous.get('native_success') if terminal else None
            if result is not None:
                if not isinstance(result, dict):
                    raise HarnessError('Invalid native result')
                self._done = True
                flags_present = 'terminated' in result or 'truncated' in result
                native_end = result.get('terminated') is True or result.get('truncated') is True
                if (result.get('complete') is False or
                        not (native_end or result.get('complete') is True) or
                        (flags_present and not native_end)):
                    raise HarnessError(f"Incomplete RoboDojo result: {result.get('reason', 'unknown')}")
                terminal = True
                success = result.get('success', result.get('native_success', success))
            if (packet.get('rollout_finished') or self.tools.phase == 'done') and not terminal:
                self._done = True
                raise HarnessError('Rollout finished without native terminal completion')
            self._done = terminal
            if success is not None and type(success) is not bool:
                raise HarnessError('Native success must be a boolean or None')
            payload = _public_payload(packet)
            features = tuple(float(value) for value in self.feature_encoder(payload))
            if not features or not all(math.isfinite(value) for value in features):
                raise HarnessError('Observation features must be nonempty and finite')
            observation = Observation(episode, step, features, payload, terminal, success)
            self._observation = observation
            return observation
        except Exception as error:
            self._failed = True
            if isinstance(error, HarnessError):
                raise
            raise HarnessError(f'Invalid RoboDojo observation: {error}') from error

    def start(self):
        if self._started or self._closed:
            raise HarnessError('start is single-use; no reset/replay')
        arguments = self._arguments('robodojo_start')
        self._started = True
        packet = self._call('start', arguments)
        self._live = True
        return self._observe(packet)

    def keepalive(self):
        if self._done:
            return
        self._active()
        try:
            # Metadata is read-only; serialize this with actions on the same RPC socket.
            self.tools.sim.request('metadata')
        except Exception as error:
            self._failed = True
            raise HarnessControlError('sim_keepalive_failed') from error

    def infer(self, observation):
        self._current(observation)
        if self._proposal is not None:
            raise HarnessError('Current proposal must be consumed before another inference')
        arguments = self._arguments('pi05_infer')
        if arguments.get('observation_path') != observation.payload.get('observation_path'):
            raise HarnessError('next_call refers to a different observation path')
        packet = self._call('infer', arguments)
        try:
            request_id = packet['request_id']
            path = packet['proposal_path']
            if not isinstance(request_id, str) or not request_id or request_id in self._request_ids:
                raise HarnessError('Inference must return a fresh request_id')
            if not isinstance(path, str) or not path or path in self._proposal_paths:
                raise HarnessError('Inference must return a fresh proposal path')
            if (packet['episode_id'], packet['step_id']) != (observation.episode_id, observation.step):
                raise HarnessError('Inference returned a stale observation identity')
            remaining = packet['remaining_steps']
            if type(remaining) is not int or remaining < 1:
                raise HarnessError('No remaining native steps without a native terminal result')
            maximum = packet.get('max_episode_steps')
            if maximum is not None:
                if type(maximum) is not int or maximum <= observation.step:
                    raise HarnessError('Native step budget exhausted without terminal result')
                remaining = min(remaining, maximum - observation.step)
            observed_remaining = observation.payload.get('remaining_steps', remaining)
            if type(observed_remaining) is not int or observed_remaining < 1:
                raise HarnessError('Invalid remaining observation step budget')
            remaining = min(remaining, observed_remaining)
            warnings, blocked, count = self._diagnostics(packet)
            proposal = Proposal(request_id, observation, _public_payload(packet),
                                min(15, remaining, count), tuple(warnings), blocked)
            self._proposal = proposal
            self._current_proposal(proposal)
            self._request_ids.add(request_id)
            self._proposal_paths.add(path)
            return proposal
        except Exception as error:
            self._failed = True
            if isinstance(error, HarnessError):
                raise
            raise HarnessError(f'Invalid RoboDojo proposal: {error}') from error

    def _diagnostics(self, packet):
        warnings = []
        diagnostics = packet.get('action_diagnostics') or {}
        actions = self.tools.actions
        try:
            count = len(actions)
            rows = [_vector(row, 14, 'action row') for row in actions]
            if not rows:
                raise HarnessError('Empty action proposal')
        except (HarnessError, TypeError) as error:
            try:
                count = len(actions)
            except TypeError:
                count = 0
            return [f'Invalid action shape or nonfinite values: {error}'], True, count
        blocked = False
        if diagnostics.get('shape', [count, 14]) != [count, 14]:
            warnings.append(f"Invalid action_diagnostics shape: {diagnostics.get('shape')}")
            blocked = True
        if diagnostics.get('finite') is False:
            warnings.append('action_diagnostics finite=False')
            blocked = True
        if diagnostics.get('status') != 'computed':
            warnings.append('action_diagnostics unavailable; checked actual tool actions')
        successive = max((abs(b[j] - a[j]) for a, b in zip(rows, rows[1:]) for j in JOINTS), default=0.0)
        reported = diagnostics.get('max_joint_step_rad')
        if reported is not None:
            try:
                reported = float(reported)
            except (TypeError, ValueError):
                reported = math.nan
            if not math.isfinite(reported) or reported < 0:
                warnings.append(f'Invalid action_diagnostics max_joint_step_rad={reported}')
                blocked = True
            elif reported > self.joint_step_threshold_rad:
                warnings.append(f'action_diagnostics max_joint_step_rad={reported:.6g} exceeds '
                                f'{self.joint_step_threshold_rad:.6g} rad (successive proposal rows only)')
        if successive > self.joint_step_threshold_rad:
            warnings.append(f'Actual successive joint jump={successive:.6g} rad exceeds '
                            f'{self.joint_step_threshold_rad:.6g} rad')
        state = _vector(packet['current_state'], 14, 'current_state')
        first = max(abs(rows[0][j] - state[j]) for j in JOINTS)
        if first > self.first_joint_step_threshold_rad:
            warnings.append(f'Measured state to first action joint jump={first:.6g} rad exceeds '
                            f'{self.first_joint_step_threshold_rad:.6g} rad')
        if any(not 0 <= row[j] <= 1 for row in rows for j in (6, 13)):
            warnings.append('Actual gripper opening is outside [0,1]')
            blocked = True
        try:
            for bounds in diagnostics.get('opening_range') or ():
                low, high = _vector(bounds, 2, 'action_diagnostics opening_range')
                if not 0 <= low <= high <= 1:
                    warnings.append(f'action_diagnostics opening_range={low:.6g}..{high:.6g} outside [0,1]')
                    blocked = True
        except (HarnessError, TypeError) as error:
            warnings.append(f'Invalid action_diagnostics opening_range: {error}')
            blocked = True
        return warnings, blocked, count

    def auto_response(self, proposal, steps, stage):
        self._current_proposal(proposal)
        if proposal.blocked:
            raise HarnessError('Cannot automatically execute a blocked proposal')
        if type(steps) is not int or not 1 <= steps <= proposal.max_steps:
            raise HarnessError('Automatic steps exceed the fresh proposal budget')
        target = {}
        for arm in ARMS:
            eef = proposal.payload['current_eef'][arm]
            position = _vector(eef['position'], 3, f'{arm} EEF position')
            quaternion = _vector(eef['quaternion_wxyz'], 4, f'{arm} EEF quaternion')
            norm = math.hypot(*quaternion)
            if not math.isfinite(norm) or norm == 0:
                raise HarnessError('Current EEF quaternion cannot be normalized')
            opening = _vector([eef['gripper_opening_command']], 1, f'{arm} gripper opening')[0]
            target[arm] = dict(position=list(position),
                               quaternion_wxyz=[value / norm for value in quaternion],
                               gripper_closed=opening < 0.5)
        explanation = 'Harness automatic student execution; no GPT visual review. Task progress and intent are unverified.'
        return dict(
            request_id=proposal.request_id, mode='student', steps=steps, reason=explanation,
            edit={arm: dict(delta_position=[0.0, 0.0, 0.0],
                            delta_rotation_vector=[0.0, 0.0, 0.0], gripper='keep') for arm in ARMS},
            target=target,
            assessment=dict(
                task_progress=dict(verified_completed=[],
                                   currently_attempting='Continue the current student proposal without visual verification.',
                                   remaining=['Task completion remains unverified.']),
                current_subgoal='Current harness stage; no visual subgoal verification.',
                execution_status='not_started' if proposal.observation.step == 0 else 'uncertain',
                execution_evidence=explanation,
                expected_next_intent='Continue toward the task instruction; appropriateness is unverified.',
                predicted_next_intent='Student intent is unverified.', intent_status='uncertain',
                intent_evidence=explanation,
            ),
        )

    def execute(self, proposal, response, source):
        self._current_proposal(proposal)
        if not isinstance(response, dict) or response.get('request_id') != proposal.request_id:
            raise HarnessError('Response request_id does not match the fresh proposal')
        if not isinstance(source, str) or not source.strip():
            raise HarnessError('An explicit execution source is required')
        if proposal.blocked:
            raise HarnessError('Cannot execute a blocked proposal')
        if response.get('mode') != 'stop':
            steps = response.get('steps')
            cap = min(proposal.max_steps, 15 if response.get('mode') == 'student' else 5)
            if type(steps) is not int or not 1 <= steps <= cap:
                raise HarnessError('Execution steps exceed the fresh proposal budget')
        arguments = self._arguments('robodojo_execute')
        if arguments.get('proposal_path') != proposal.payload['proposal_path']:
            raise HarnessError('next_call refers to a different proposal path')
        arguments['response'] = response
        self.last_source = source
        self._proposal = None
        # External execute remains responsible for tools.validate and physical execution.
        return self._observe(self._call('execute', arguments))

    def stop(self, reason):
        live = self._live or (self._started and self.tools.episode is not None)
        if not live or self._done or self._closed or self.tools.phase == 'done':
            return
        self._done = True
        self._proposal = None
        self._call('finish', {'reason': reason})

    def close(self):
        if not self._closed:
            self._closed = True
            self._call('close', {})

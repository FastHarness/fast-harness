import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from .engine import Harness
from .memory import Memory
from .scheduler import Config
from .toy import ToyBackend, ToyReviewer
from .types import error_diagnostics


def demo(args) -> None:
    if args.episodes < 1 or not 0 <= args.fault_episode <= args.episodes:
        raise ValueError('Episodes must be positive; fault episode is 0 or an existing episode')
    args.output.mkdir(parents=True, exist_ok=False)
    memory_dir = args.memory_dir or args.output
    memory_dir.mkdir(parents=True, exist_ok=True)
    modes = ('baseline', 'adaptive') if args.mode == 'compare' else (args.mode,)
    rows = []
    for mode in modes:
        memory = Memory(memory_dir / f'{mode}.sqlite', 'toy-v1:symbolic-features-v1')
        try:
            for episode in range(1, args.episodes + 1):
                config = Config(always_review=mode == 'baseline', seed=episode,
                                max_interval=args.max_interval)
                backend = ToyBackend(fault=episode == args.fault_episode)
                result = Harness(backend, ToyReviewer(), memory,
                    args.output / mode / f'episode-{episode:03d}', config).run()
                rows.append(dict(mode=mode, episode=episode, **result))
                print(json.dumps(rows[-1], ensure_ascii=False))
        finally:
            memory.close()
    (args.output / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


def _reviewer(args, *, before_attempt=None):
    from .reviewer import AnthropicReviewer, ResponsesReviewer, TimedRetryReviewer
    if args.reviewer == 'anthropic':
        base = AnthropicReviewer(args.endpoint, args.model, args.api_key_env or 'ANTHROPIC_API_KEY',
                                 timeout=args.timeout, effort=args.effort, max_tokens=args.max_tokens)
    else:
        base = ResponsesReviewer(args.endpoint, args.model, args.api_key_env or 'OPENAI_API_KEY',
                                 timeout=args.timeout, effort=args.effort)
    call_log = getattr(args, 'call_log', None)
    retry_wait = getattr(args, 'retry_transport_wait', 0.0) or 0.0
    schema_retries = getattr(args, 'max_schema_retries', 0) or 0
    if call_log or retry_wait or schema_retries or before_attempt is not None:
        return TimedRetryReviewer(base, log_path=call_log, retry_wait=retry_wait,
                                  max_retries=getattr(args, 'max_transport_retries', 9),
                                  slow_seconds=getattr(args, 'slow_seconds', 90.0),
                                  episode=getattr(args, 'episode_label', None),
                                  schema_retries=schema_retries,
                                  schema_retry_wait=getattr(args, 'schema_retry_wait', 5.0),
                                  before_attempt=before_attempt)
    return base


def probe(args) -> None:
    if not args.allow_model_requests:
        raise ValueError('Probe requires --allow-model-requests; one synthetic review will be sent')
    from dataclasses import asdict
    import struct
    import time
    import zlib
    from .types import Observation, Proposal, ReviewRequest

    reviewer = _reviewer(args)
    args.output.mkdir(parents=True, exist_ok=False)
    images = []
    if args.with_image:
        def chunk(kind, data):
            return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
        image = args.output.resolve() / 'synthetic.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\n'
                          + chunk(b'IHDR', struct.pack('!2I5B', 64, 64, 8, 2, 0, 0, 0))
                          + chunk(b'IDAT', zlib.compress((b'\0' + b'\x80\x80\x80' * 64) * 64))
                          + chunk(b'IEND', b''))
        images = [{'camera': 'synthetic-grey-not-a-robot', 'path': str(image)}]
    state = [0.] * 6 + [1.] + [0.] * 6 + [1.]
    poses = {arm: dict(position=[x, 0., .2], quaternion_wxyz=[1., 0., 0., 0.],
                       gripper_opening_command=1.) for arm, x in (('left', -.1), ('right', .1))}
    observation = Observation('synthetic-probe', 0, (0.,), dict(
        instruction='Hold both arms at their measured poses with open grippers. This is synthetic protocol-test data, not a robot scene. Any supplied image is a generated grey test pattern.',
        current_state=state, current_eef=poses, images=images, remaining_steps=1))
    proposal = Proposal('synthetic-proposal', observation, dict(
        current_state=state, student_eef_trajectory=[poses],
        action_diagnostics=dict(shape=[1, 14], finite=True, status='computed')), max_steps=1)
    request = ReviewRequest(observation, proposal, None, 'hold', ('hold',), (), (), {}, ())
    summary = dict(status='request_pending', model_requests_attempted=1, robot_runs=0,
                   reviewer=args.reviewer, model=args.model, effort=args.effort,
                   synthetic_image=args.with_image)
    result_path = args.output / 'probe.json'
    result_path.write_text(json.dumps(summary, indent=2) + '\n')
    started = time.monotonic()
    try:
        result = reviewer.review(request)
        summary.update(status='passed', stage=result.stage, last_outcome=result.last_outcome,
                       response_mode=result.response['mode'], response_steps=result.response['steps'])
    except Exception as error:
        summary.update(status='failed', error_type=type(error).__name__, **error_diagnostics(error))
        raise
    finally:
        summary.update(model_requests_attempted=reviewer.calls,
                       usage=[asdict(item) for item in reviewer.usage_history],
                       wall_seconds=time.monotonic() - started)
        result_path.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary))


def _reviewer_arguments(parser):
    parser.add_argument('--reviewer', choices=('responses', 'anthropic'), default='responses')
    parser.add_argument('--endpoint', required=True, help='Complete /responses or /v1/messages endpoint')
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key-env', help='Default: OPENAI_API_KEY for responses, ANTHROPIC_API_KEY for anthropic')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Anthropic output token limit')
    parser.add_argument('--effort')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--call-log', help='JSONL path: append per-review latency, usage and outcome')
    parser.add_argument('--retry-transport-wait', type=float, default=0.0,
                        help='Seconds to wait before resending a review after a transport stall (0 disables retry)')
    parser.add_argument('--max-transport-retries', type=int, default=9,
                        help='Max resends per review on stall (default 9 = 10 requests; 0 = unlimited while --retry-transport-wait>0)')
    parser.add_argument('--slow-seconds', type=float, default=90.0,
                        help='Flag successful reviews slower than this many seconds in the call log')
    parser.add_argument('--episode-label', help='Label recorded in the call log for this run/episode')
    parser.add_argument('--max-schema-retries', type=int, default=0,
                        help='Re-ask the model this many times when its output fails strict validation (0=off)')
    parser.add_argument('--schema-retry-wait', type=float, default=5.0,
                        help='Seconds to wait before re-asking after a schema/constraint rejection')
    parser.add_argument('--allow-model-requests', action='store_true')


def live(args) -> None:
    if not args.allow_model_requests:
        raise ValueError('Live mode requires --allow-model-requests; images will be sent to the configured provider')
    if args.output.exists():
        raise FileExistsError('Live output must be a new directory')
    if args.upstream:
        sys.path.insert(0, str(args.upstream.resolve()))
    # The simulator runtime and the policy server are external and caller-provided
    # (e.g. the RoboDojo eval client + a policy server). Point --runtime-module at a
    # package exposing two classes:
    #   PolicyClient(port, checkpoint)  -> .metadata, .close(), the action-chunk client
    #   RoboDojoTools(rollout_dir, task, policy, sim_port=, seed=, max_decisions=)
    # See docs/DESIGN.md ("RoboDojo integration") for the full payload contract.
    import importlib
    runtime = importlib.import_module(args.runtime_module)
    PolicyClient = getattr(runtime, 'PolicyClient')
    RoboDojoTools = getattr(runtime, 'RoboDojoTools')
    from .robodojo import RoboDojoBackend

    config = Config(always_review=args.always_review, seed=args.seed,
                    episode_number=args.episode_number, sim_attempt=args.sim_attempt,
                    max_interval=args.max_interval,
                    max_unreviewed_steps=args.max_unreviewed_steps,
                    max_episode_chunks=args.max_episode_chunks,
                    min_samples=getattr(args, 'min_samples', 4),
                    cooldown_successes=getattr(args, 'cooldown_successes', 3),
                    match_threshold=getattr(args, 'match_threshold', 0.06),
                    audit_probability=getattr(args, 'audit_probability', 0.05),
                    confidence_cap=getattr(args, 'confidence_cap', 1.0),
                    relax_known_stage_change=getattr(args, 'relax_stage_change', False),
                    contact_gating=getattr(args, 'contact_gating', False),
                    contact_low_z=getattr(args, 'contact_low_z', 0.95),
                    contact_descent_m=getattr(args, 'contact_descent_m', 0.05),
                    gate_stage_change_on_contact_only=getattr(args, 'gate_stage_change_on_contact_only', False),
                    commit_gating=getattr(args, 'commit_gating', False),
                    commit_cap=getattr(args, 'commit_cap', 6),
                    commit_min_confidence=getattr(args, 'commit_min_confidence', 0.9),
                    persist_error_review=getattr(args, 'persist_error_review', False))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'rollout').mkdir()
    student = backend = memory = None
    try:
        student = PolicyClient(args.student_port, args.checkpoint.resolve())
        identity = dict(task=args.task, policy=student.metadata['checkpoint_sha256'],
                        policy_config=student.metadata['config'], features='rgb8-state-v1',
                        teacher=args.model, scope=args.scope, reviewer_protocol=args.reviewer,
                        reviewer_endpoint_sha256=hashlib.sha256(args.endpoint.encode()).hexdigest(),
                        teacher_effort=args.effort,
                        teacher_max_tokens=args.max_tokens if args.reviewer == 'anthropic' else None)
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        args.memory.parent.mkdir(parents=True, exist_ok=True)
        memory = Memory(args.memory, namespace, decay=getattr(args, 'memory_decay', 1.0))
        tools = RoboDojoTools(args.output.resolve() / 'rollout', args.task, student,
                             sim_port=args.sim_port, seed=args.seed, max_decisions=0)
        backend = RoboDojoBackend(tools)
        reviewer = _reviewer(args, before_attempt=backend.keepalive)
        (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
        from .tail_guard import REVISION, TailGuard
        guard_path = getattr(args, 'recovery_tail_guard', None)
        guard = TailGuard.for_controller(guard_path, output=args.output.resolve(),
            episode_number=config.episode_number, sim_attempt=config.sim_attempt) if guard_path else None
        package = Path(__file__).resolve().parent
        provenance = dict(harness_revision=REVISION, source=str(package),
                          source_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sorted(package.glob('*.py'))})
        (args.output / 'runtime_provenance.json').write_text(
            json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        result = Harness(backend, reviewer, memory, args.output / 'harness', config, tail_guard=guard).run()
        print(json.dumps(result))
    finally:
        if backend is not None:
            backend.close()
        if student is not None:
            student.close()
        if memory is not None:
            memory.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Experience-guided review of fresh VLA action chunks')
    commands = parser.add_subparsers(dest='command', required=True)
    toy = commands.add_parser('demo', help='Run a deterministic offline symbolic demonstration')
    toy.add_argument('--output', type=Path, required=True)
    toy.add_argument('--memory-dir', type=Path, help='POSIX local directory for SQLite; defaults to output')
    toy.add_argument('--episodes', type=int, default=6)
    toy.add_argument('--fault-episode', type=int, default=0)
    toy.add_argument('--mode', choices=('adaptive', 'baseline', 'compare'), default='compare')
    toy.add_argument('--max-interval', type=int, default=6)
    toy.set_defaults(handler=demo)
    robot = commands.add_parser('live', help='Connect to already running RoboDojo and VLA services')
    robot.add_argument('--output', type=Path, required=True)
    robot.add_argument('--memory', type=Path, required=True)
    robot.add_argument('--task', required=True)
    robot.add_argument('--checkpoint', type=Path, required=True)
    robot.add_argument('--upstream', type=Path)
    robot.add_argument('--runtime-module', default='robodojo_runtime',
                       help='Importable module exposing PolicyClient and RoboDojoTools (see docs/DESIGN.md)')
    _reviewer_arguments(robot)
    robot.add_argument('--student-port', type=int, default=18830)
    robot.add_argument('--sim-port', type=int, default=19113)
    robot.add_argument('--seed', type=int, default=0)
    robot.add_argument('--scope', default='fixed-layout-v1', help='Change this when layout/task semantics change')
    robot.add_argument('--always-review', action='store_true')
    robot.add_argument('--episode-number', type=int, help='Logical campaign episode number')
    robot.add_argument('--sim-attempt', type=int, default=1,
                       help='Physical attempt number; ep1 attempt >=3 permits a student prefix after 10 transport failures')
    robot.add_argument('--recovery-tail-guard', type=Path, help='Physical-launch deadline arbitration state')
    robot.add_argument('--max-interval', type=int, default=6)
    robot.add_argument('--max-unreviewed-steps', type=int, default=60)
    robot.add_argument('--max-episode-chunks', type=int, default=1000, help='Stop incomplete after this many action chunks')
    robot.add_argument('--min-samples', type=int, default=4, help='Confirmed successes before a stage may skip review')
    robot.add_argument('--cooldown-successes', type=int, default=3, help='Clean successes required after an error before skipping resumes')
    robot.add_argument('--match-threshold', type=float, default=0.06, help='Feature distance under which a stage counts as familiar')
    robot.add_argument('--audit-probability', type=float, default=0.05, help='Random re-audit probability while skipping is allowed')
    robot.add_argument('--confidence-cap', type=float, default=1.0, help='>1 lets the skip interval keep growing with accumulated successes (dynamic skip rate)')
    robot.add_argument('--relax-stage-change', action='store_true', help='Allow skipping a transition into an already-proven stage')
    robot.add_argument('--contact-gating', action='store_true', help='force review at manipulation contact/alignment moments (read from the policy proposal), skip transit')
    robot.add_argument('--contact-low-z', type=float, default=0.95, help='EEF height (m) at/below which the arm is near the work surface')
    robot.add_argument('--contact-descent-m', type=float, default=0.05, help='EEF drop within a chunk counted as a pre-contact descent')
    robot.add_argument('--gate-stage-change-on-contact-only', action='store_true', help='a stage change forces review only at contact or into an unproven stage')
    robot.add_argument('--commit-gating', action='store_true', help='proven subtasks run their recorded chunk-span without a mid-subtask forced review (fewer reviews)')
    robot.add_argument('--commit-cap', type=int, default=12, help='max chunks a proven subtask may run before a forced review')
    robot.add_argument('--commit-min-confidence', type=float, default=0.9, help='Wilson lower bound a subtask needs before commit extends its span')
    robot.add_argument('--persist-error-review', action='store_true', help='keep reviewing near a past failure location until robustly re-solved (no coarse stage-streak decay)')
    robot.add_argument('--memory-decay', type=float, default=1.0, help='success/failure count decay; 1.0 = pure lifetime accumulation (improves with experience), <1 = EWMA that adapts but forgets old mastery')
    robot.set_defaults(handler=live)
    check = commands.add_parser('probe', help='Send one synthetic review; never connect to a robot')
    check.add_argument('--output', type=Path, required=True)
    check.add_argument('--with-image', action='store_true', help='Attach a generated grey PNG, not a workspace image')
    _reviewer_arguments(check)
    check.set_defaults(handler=probe)
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as error:
        location = 'probe.json' if args.command == 'probe' else 'events.jsonl'
        print(f'fast-harness stopped ({type(error).__name__}); inspect local {location} if created.', file=sys.stderr)
        diagnostics = error_diagnostics(error)
        if diagnostics:
            print(json.dumps(diagnostics), file=sys.stderr)
        if isinstance(error, sqlite3.OperationalError):
            print('SQLite needs a local filesystem with locking and transactions. Use --memory-dir (demo) or --memory (live) outside object storage.', file=sys.stderr)
        raise SystemExit(1) from None

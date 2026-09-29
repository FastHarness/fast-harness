import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from .memory import Memory
from .recovery import Recovery
from .scheduler import Config, MotionMonitor, Scheduler
from .types import (Backend, Checkpoint, HarnessControlError, ReviewRequest, Reviewer,
                    ReviewTransportExhaustedError, Transition, error_diagnostics)


class Harness:
    def __init__(self, backend: Backend, reviewer: Reviewer, memory: Memory,
                 output: str | Path, config: Config | None = None, *, tail_guard=None):
        self.backend, self.reviewer, self.memory = backend, reviewer, memory
        self.output = Path(output)
        self.tail_guard = tail_guard
        self.config = config or Config()
        self.checkpoints: list[Checkpoint] = []
        self.recovery = Recovery(self.config.attempt_chunks, self.config.recovery_chunks)
        self.metrics = dict(chunks=0, teacher_calls=0, automatic_chunks=0, student_steps=0,
                            teacher_steps=0, control_steps=0, teacher_seconds=0.0,
                            memory_seconds=0.0, native_success=None, complete=False,
                            status='running', checkpoints=0, recovery_events=0,
                            episode_number=self.config.episode_number, sim_attempt=self.config.sim_attempt,
                            degraded=False, timeout_fallbacks=0, degraded_chunks=0, degraded_steps=0,
                            recovery_passthrough=False, recovery_passthrough_chunks=0,
                            recovery_passthrough_steps=0, recovery_passthrough_reason=None,
                            recovery_passthrough_started_step=None,
                            harness_revision='tail-guard-1')
        self.usages = []
        self.started = 0.0
        self.stream = None

    def emit(self, event: str, **fields) -> None:
        record = dict(event=event, elapsed_seconds=time.monotonic() - self.started, **fields)
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.stream = (self.output / 'events.jsonl').open('x', encoding='utf-8')
        try:
            self.emit('start', config=asdict(self.config), namespace=self.memory.namespace,
                      method='fast_harness_v1', baseline_compatible=False,
                      harness_revision=self.metrics['harness_revision'])
            self._run()
        except Exception as error:
            self.metrics['status'] = 'audit_failed' if self.metrics['complete'] else 'incomplete'
            diagnostics = dict(error_type=type(error).__name__, **error_diagnostics(error))
            self.metrics.update(diagnostics)
            self.emit('failure', **diagnostics)
            try:
                self.backend.stop(diagnostics.get('failure_reason', 'fast_harness_error'))
            except Exception as stop_error:
                self.emit('stop_failure', error_type=type(stop_error).__name__)
            raise
        finally:
            usage = getattr(self.reviewer, 'usage_history', self.usages)
            self.metrics['teacher_calls'] = getattr(self.reviewer, 'calls', self.metrics['teacher_calls'])
            self.metrics['usage_known_calls'] = sum(u.input_tokens is not None and
                                                    u.output_tokens is not None for u in usage)
            for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'cache_creation_input_tokens'):
                values = [getattr(item, key) for item in usage]
                self.metrics[key] = sum(values) if len(values) == self.metrics['teacher_calls'] and all(
                    value is not None for value in values) else None
            self.metrics['wall_seconds'] = time.monotonic() - self.started
            self.emit('summary', **self.metrics)
            (self.output / 'summary.json').write_text(
                json.dumps(self.metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            self.stream.close()
        return dict(self.metrics)

    def _run(self) -> None:
        from .reviewer import validate_review

        cfg = self.config
        scheduler, monitor = Scheduler(cfg), MotionMonitor(cfg.stall_chunks)
        observation = self.backend.start()
        last_review_step, previous_stage = None, None
        chunks_since_review, force_review, stage_run = 0, False, 0
        previous = None
        unreviewed: list[Transition] = []
        seen_requests: set[str] = set()
        while True:
            passthrough_reason = None
            started = time.monotonic()
            match = self.memory.match(observation.features, cfg.match_threshold, cfg.ambiguity_margin)
            self.metrics['memory_seconds'] += time.monotonic() - started
            if observation.terminal:
                self.metrics.update(complete=True, native_success=observation.native_success)
                proposal, reason, gap = None, 'terminal_audit', 1
                step_limit = cfg.max_unreviewed_steps
            else:
                if self.metrics['chunks'] >= cfg.max_episode_chunks:
                    raise HarnessControlError('episode_decision_budget')
                proposal = self._infer(observation, seen_requests)
                reason, gap, step_limit = scheduler.reason(self.memory, match, observation, proposal,
                    last_review_step=last_review_step, chunks_since_review=chunks_since_review,
                    previous_stage=previous_stage, recovery_active=self.recovery.active,
                    force_review=force_review, stalled=monitor.stalled)
                if cfg.episode_number == 1:
                    reason = reason or 'first_episode_review'
            self.emit('schedule', step=observation.step, stage=match.stage, reason=reason,
                      interval=gap, step_limit=step_limit,
                      unreviewed_steps=observation.step - last_review_step if last_review_step is not None else 0,
                      match_distance=match.distance if math.isfinite(match.distance) else None)
            stage = match.stage
            if reason:
                source = 'teacher'
                account_previous = True
                for _ in range(8):
                    options = (self.recovery.candidates(self.checkpoints) if self.recovery.active
                               else tuple(self.checkpoints))
                    if self.recovery.target and self.recovery.target not in options:
                        options = (*options, self.recovery.target)
                    request = ReviewRequest(observation, proposal, previous, stage,
                        self.memory.stages(), self.memory.errors(stage), options,
                        self.recovery.context(), tuple(unreviewed))
                    self.metrics['teacher_calls'] += 1
                    started = time.monotonic()
                    try:
                        review = self.reviewer.review(request)
                    except ReviewTransportExhaustedError as error:
                        if (cfg.episode_number != 1 or cfg.sim_attempt < 3 or error.attempts != 10
                                or observation.terminal):
                            raise
                        if (self.recovery.phase == 'exhausted' or
                                (self.recovery.active and
                                 (self.recovery.total_steps >= cfg.recovery_chunks or
                                  self.recovery.attempt_steps >= cfg.attempt_chunks))):
                            passthrough_reason = 'recovery_exhausted'
                            break
                        steps = min(proposal.max_steps, cfg.max_unreviewed_steps)
                        response = self.backend.auto_response(proposal, steps, stage)
                        if response.get('mode') != 'student':
                            raise HarnessControlError('automatic_nonstudent_action') from error
                        response = dict(response, reason='Reviewer transport exhausted after 10 failures; '
                                        'unreviewed student prefix in a degraded first-episode attempt.')
                        source, force_review = 'reviewer_timeout_fallback', True
                        self.metrics['degraded'] = True
                        self.metrics['timeout_fallbacks'] += 1
                        self.emit('reviewer_timeout_fallback', step=observation.step,
                                  request_id=proposal.request_id, episode_number=cfg.episode_number,
                                  sim_attempt=cfg.sim_attempt, transport_failures=error.attempts,
                                  review_verified=False, recovery=self.recovery.context())
                        break
                    finally:
                        self.metrics['teacher_seconds'] += time.monotonic() - started
                    self.usages.append(review.usage)
                    validate_review(review, request)
                    self.emit('review', step=observation.step, review=asdict(review),
                              reason=reason, recovery=self.recovery.context())
                    stage = review.stage
                    was_active = self.recovery.active
                    if account_previous:
                        if proposal and review.response['assessment']['intent_status'] == 'misaligned':
                            case_id = self.memory.record(episode=observation.episode_id,
                                step=observation.step, kind='intent', stage=review.stage,
                                features=observation.features, outcome='error',
                                evidence=review.evidence, student=False, error_type='misaligned_intent')
                            if case_id:
                                self.recovery.error_cases.add(case_id)
                        self._remember(review, previous, observation, was_active)
                        account_previous = False
                        unreviewed.clear()
                    last_review_step, chunks_since_review = observation.step, 0
                    if observation.terminal:
                        self.metrics.update(status='completed', complete=True,
                                            native_success=observation.native_success)
                        return
                    self.memory.learn_stage(stage, observation.features)
                    if was_active:
                        changed = self.recovery.update(review, self.checkpoints)
                        if self.recovery.phase == 'normal':
                            self.memory.recovered(self.recovery.error_cases, review.evidence)
                            self.recovery.error_cases.clear()
                    elif (review.last_outcome == 'error' or
                          review.response['assessment']['intent_status'] == 'misaligned' or
                          review.response['assessment']['execution_status'] == 'failed'):
                        self.recovery.start(observation.step,
                            'Resolve the observed problem and resume: ' +
                            review.response['assessment']['expected_next_intent'] +
                            '. Original evidence: ' + review.evidence)
                        changed = True
                        self.metrics['recovery_events'] += 1
                    else:
                        changed = False
                    if self.recovery.phase == 'exhausted':
                        passthrough_reason = 'recovery_exhausted'
                        break
                    if changed:
                        self.emit('recovery_transition', step=observation.step,
                                  recovery=self.recovery.context())
                        reason = 'recovery_replan'
                        continue
                    response = dict(review.response, steps=min(review.response['steps'], cfg.max_unreviewed_steps))
                    force_review = (review.last_outcome == 'unknown' or
                                    response['assessment']['intent_status'] == 'uncertain')
                    break
                else:
                    passthrough_reason = 'recovery_replanning_exhausted'
            else:
                remaining = step_limit - (observation.step - last_review_step)
                steps = min(proposal.max_steps, remaining)
                if steps <= 0:
                    raise HarnessControlError('automatic_review_budget')
                response = self.backend.auto_response(proposal, steps, stage)
                if response.get('mode') != 'student':
                    raise HarnessControlError('automatic_nonstudent_action')
                source = 'harness_auto'
            if passthrough_reason is not None:
                self._run_passthrough(observation, proposal, stage, seen_requests, passthrough_reason)
                return
            if response['request_id'] != proposal.request_id:
                raise HarnessControlError('response_request_mismatch')
            if not 1 <= response['steps'] <= proposal.max_steps:
                raise HarnessControlError('response_horizon')
            self.emit('execution_requested', request_id=proposal.request_id,
                      step=observation.step, source=source, response=response)
            after = self.backend.execute(proposal, response, source)
            if after.episode_id != observation.episode_id or not 0 < after.step - observation.step <= response['steps']:
                raise HarnessControlError('execution_acknowledgement')
            if after.terminal:
                self.metrics.update(complete=True, native_success=after.native_success)
            previous = Transition(observation, after, proposal.request_id, stage,
                                  response['mode'], source, after.step - observation.step)
            unreviewed.append(previous)
            monitor.observe(previous)
            self.recovery.executed()
            self.metrics['chunks'] += 1
            self.metrics['automatic_chunks'] += int(source == 'harness_auto')
            self.metrics['degraded_chunks'] += int(source == 'reviewer_timeout_fallback')
            if source == 'reviewer_timeout_fallback':
                self.metrics['degraded_steps'] += previous.steps
            self.metrics['control_steps'] += previous.steps
            key = 'student_steps' if response['mode'] == 'student' else 'teacher_steps'
            self.metrics[key] += previous.steps
            self.emit('execution_completed', request_id=proposal.request_id, source=source,
                      mode=response['mode'], stage=stage, start_step=observation.step,
                      end_step=after.step, terminal=after.terminal)
            if source == 'reviewer_timeout_fallback':
                previous_stage, stage_run = None, 0
            elif cfg.commit_gating:                       # chunk table: record the completed run length
                if previous_stage is not None and stage != previous_stage:
                    self.memory.record_span(previous_stage, stage_run)
                    stage_run = 1
                else:
                    stage_run += 1
            previous_stage = None if source == 'reviewer_timeout_fallback' else stage
            observation = after
            chunks_since_review += 1

    def _infer(self, observation, seen_requests):
        proposal = self.backend.infer(observation)
        if (proposal.observation.episode_id != observation.episode_id or
                proposal.observation.step != observation.step or
                proposal.request_id in seen_requests or not proposal.request_id):
            raise HarnessControlError('stale_proposal')
        seen_requests.add(proposal.request_id)
        if proposal.blocked or type(proposal.max_steps) is not int or not 1 <= proposal.max_steps <= 15:
            raise HarnessControlError('invalid_proposal')
        return proposal

    def _run_passthrough(self, observation, proposal, stage, seen_requests, reason):
        source = 'recovery_exhausted_passthrough'
        if self.tail_guard is not None:
            self.tail_guard.enter(step=observation.step, reason=reason)
        self.metrics.update(recovery_passthrough=True, recovery_passthrough_reason=reason,
                            recovery_passthrough_started_step=observation.step)
        self.emit(source, step=observation.step, request_id=proposal.request_id,
                  reason=reason, episode_number=self.config.episode_number,
                  sim_attempt=self.config.sim_attempt, recovery=self.recovery.context(),
                  review_verified=False, harness_revision=self.metrics['harness_revision'])
        while not observation.terminal:
            response = self.backend.auto_response(proposal, proposal.max_steps, stage)
            if response.get('mode') != 'student':
                raise HarnessControlError('automatic_nonstudent_action')
            if response['request_id'] != proposal.request_id:
                raise HarnessControlError('response_request_mismatch')
            if type(response['steps']) is not int or not 1 <= response['steps'] <= proposal.max_steps:
                raise HarnessControlError('response_horizon')
            response = dict(response, reason='Recovery exhausted; execute the fresh student proposal '
                            'without review until native termination. Progress remains unverified.')
            self.emit('execution_requested', request_id=proposal.request_id,
                      step=observation.step, source=source, response=response)
            after = self.backend.execute(proposal, response, source)
            steps = after.step - observation.step
            if after.episode_id != observation.episode_id or not 0 < steps <= response['steps']:
                raise HarnessControlError('execution_acknowledgement')
            self.metrics['chunks'] += 1
            self.metrics['control_steps'] += steps
            self.metrics['student_steps'] += steps
            self.metrics['recovery_passthrough_chunks'] += 1
            self.metrics['recovery_passthrough_steps'] += steps
            if after.terminal:
                self.metrics.update(status='completed', complete=True, native_success=after.native_success)
            self.emit('execution_completed', request_id=proposal.request_id, source=source,
                      mode=response['mode'], stage=stage, start_step=observation.step,
                      end_step=after.step, terminal=after.terminal)
            observation = after
            if not observation.terminal:
                proposal = self._infer(observation, seen_requests)

    def _remember(self, review, previous, observation, recovery_active):
        if previous is None:
            return
        started = time.monotonic()
        case_id = self.memory.record(episode=previous.before.episode_id,
            step=previous.before.step, kind='execution', stage=previous.stage,
            features=previous.before.features, outcome=review.last_outcome,
            evidence=review.evidence, student=previous.mode == 'student' and not recovery_active,
            error_type=review.error_type)
        if case_id:
            self.recovery.error_cases.add(case_id)
        if review.checkpoint and review.last_outcome == 'ok' and not recovery_active:
            point = Checkpoint(f'{observation.episode_id}:{observation.step}', observation,
                               review.stage, review.checkpoint['goal'],
                               tuple(review.checkpoint['prerequisites']))
            if not self.checkpoints or self.checkpoints[-1].id != point.id:
                self.checkpoints.append(point)
                self.checkpoints[:] = self.checkpoints[-self.config.max_checkpoints:]
                self.metrics['checkpoints'] += 1
                self.emit('checkpoint', id=point.id, stage=point.stage, goal=point.goal,
                          prerequisites=point.prerequisites, step=observation.step,
                          observation=asdict(observation))
        self.metrics['memory_seconds'] += time.monotonic() - started

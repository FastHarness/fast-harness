import ast
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import Mock, patch

from fast_harness import cli
from fast_harness.engine import Harness
from fast_harness.memory import Memory
from fast_harness.scheduler import Config
from fast_harness.toy import ToyBackend, ToyReviewer
from fast_harness.types import HarnessControlError


class TransitBackend(ToyBackend):
    def observation(self):
        obs = super().observation()
        obs.payload['toy']['stage'] = 'carry'
        terminal = self.step >= 300
        return replace(obs, features=(0.,), terminal=terminal,
                       native_success=True if terminal else None)


class BudgetTests(unittest.TestCase):
    def run_episode(self, **options):
        memory = Memory(':memory:', 'budget-regression', decay=1.0)
        self.addCleanup(memory.close)
        memory.learn_stage('carry', (0.,))
        for step in range(100):
            memory.record(episode='prior', step=step, kind='execution', stage='carry',
                          features=(0.,), outcome='ok', evidence='Offline fixture', student=True)
        memory.record_span('carry', 99)
        config = dict(commit_gating=True, commit_cap=12, max_unreviewed_steps=120,
                      confidence_cap=6, max_interval=20, min_samples=2,
                      audit_probability=0, max_episode_chunks=100)
        config.update(options)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'harness'
            result = Harness(TransitBackend(chunks=1000), ToyReviewer(), memory,
                             output, Config(**config)).run()
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['complete'])
        self.assertTrue(result['native_success'])
        return events

    def test_commit_runs_past_135_and_reviews_at_effective_limit(self):
        events = self.run_episode()
        actions = [e for e in events if e['event'] == 'execution_requested']
        self.assertTrue(any(e['step'] == 135 and e['source'] == 'harness_auto' for e in actions))
        reviews = [e['step'] for e in events if e['event'] == 'review']
        self.assertIn(210, reviews)
        self.assertNotIn(135, reviews)

    def test_commit_off_preserves_120_step_limit(self):
        events = self.run_episode(commit_gating=False)
        limit_reviews = [e['step'] for e in events
                         if e['event'] == 'schedule' and e['reason'] == 'physical_step_limit']
        self.assertEqual(limit_reviews, [135, 255])

    def test_non_multiple_budget_truncates_prefix_before_review(self):
        events = self.run_episode(max_unreviewed_steps=127, commit_cap=2)
        actions = [e for e in events if e['event'] == 'execution_requested']
        prefix = next(e for e in actions if e['step'] == 135)
        self.assertEqual(prefix['response']['steps'], 7)
        self.assertEqual(prefix['source'], 'harness_auto')
        self.assertTrue(any(e['step'] == 142 and e['source'] == 'teacher' for e in actions))

    def test_every_automatic_action_fits_logged_effective_budget(self):
        events = self.run_episode()
        schedule = None
        for event in events:
            if event['event'] == 'schedule':
                schedule = event
            if event['event'] == 'execution_requested' and event['source'] == 'harness_auto':
                self.assertGreater(event['response']['steps'], 0)
                self.assertLessEqual(schedule['unreviewed_steps'] + event['response']['steps'],
                                     schedule['step_limit'])

    def test_always_review_has_no_automatic_actions(self):
        events = self.run_episode(always_review=True)
        self.assertFalse(any(e.get('source') == 'harness_auto' for e in events))

    def test_small_review_budget_still_caps_teacher(self):
        events = self.run_episode(always_review=True, max_unreviewed_steps=4)
        actions = [e for e in events if e['event'] == 'execution_requested']
        self.assertTrue(all(e['response']['steps'] <= 4 for e in actions))


class ControlDiagnosticTests(unittest.TestCase):
    def failure(self, backend, reviewer, config, error):
        memory = Memory(':memory:', 'diagnostics')
        self.addCleanup(memory.close)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            with self.assertRaises(error):
                Harness(backend, reviewer, memory, output, config).run()
            summary = json.loads((output / 'summary.json').read_text())
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
        self.assertTrue(backend.stopped)
        return summary, next(e for e in events if e['event'] == 'failure')

    def test_transport_exhaustion_has_reason_and_never_executes(self):
        from fast_harness.reviewer import TimedRetryReviewer
        from fast_harness.types import ReviewTransportError, ReviewTransportExhaustedError
        backend = ToyBackend()
        inner = Mock()
        inner.review.side_effect = ReviewTransportError('private upstream text')
        waits = []
        reviewer = TimedRetryReviewer(inner, retry_wait=120, max_retries=9, sleep=waits.append)
        summary, failure = self.failure(backend, reviewer, Config(), ReviewTransportExhaustedError)
        self.assertEqual(inner.review.call_count, 10)
        self.assertEqual(waits, [120] * 9)
        self.assertEqual(backend.executions, 0)
        self.assertEqual(summary['failure_reason'], 'reviewer_no_response')
        self.assertEqual(summary['transport_error'],
                         dict(code='review_transport_exhausted', attempts=10))
        self.assertEqual(failure['transport_error'], summary['transport_error'])
        self.assertFalse(summary['complete'])
        self.assertNotIn('private upstream text', json.dumps([summary, failure]))

    def test_decision_budget_has_fixed_code_in_summary_and_event(self):
        summary, failure = self.failure(ToyBackend(), ToyReviewer(),
            Config(max_episode_chunks=1), HarnessControlError)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['control_error'], {'code': 'episode_decision_budget'})
        self.assertEqual(failure['control_error'], summary['control_error'])

    def test_stale_proposal_is_rejected_before_execution(self):
        class StaleBackend(ToyBackend):
            def infer(self, observation):
                return replace(super().infer(observation),
                               observation=replace(observation, step=observation.step + 1))
        backend = StaleBackend()
        summary, _ = self.failure(backend, ToyReviewer(), Config(), HarnessControlError)
        self.assertEqual(summary['control_error']['code'], 'stale_proposal')
        self.assertEqual(backend.executions, 0)

    def test_unknown_exception_does_not_leak_message(self):
        class BrokenReviewer(ToyReviewer):
            def review(self, request):
                raise RuntimeError('private upstream detail')
        summary, failure = self.failure(ToyBackend(), BrokenReviewer(), Config(), RuntimeError)
        self.assertNotIn('control_error', summary)
        self.assertEqual(summary['error_type'], 'RuntimeError')
        self.assertNotIn('private upstream detail', json.dumps([summary, failure]))

    def test_terminal_native_result_survives_failed_audit(self):
        class BrokenFinalReviewer(ToyReviewer):
            def review(self, request):
                if request.observation.terminal:
                    raise RuntimeError('final audit failed')
                return super().review(request)
        summary, _ = self.failure(ToyBackend(chunks=2), BrokenFinalReviewer(), Config(), RuntimeError)
        self.assertTrue(summary['complete'])
        self.assertTrue(summary['native_success'])
        self.assertEqual(summary['status'], 'audit_failed')


class ReviewerKeepaliveTests(unittest.TestCase):
    def test_metadata_keeps_fresh_proposal_without_actions_or_state_changes(self):
        from fast_harness.robodojo import RoboDojoBackend
        from test_robodojo import FakeTools
        tools = FakeTools()
        tools.sim = Mock()
        tools.sim.request.return_value = {'private_metadata': 'never forward to reviewer'}
        backend = RoboDojoBackend(tools, feature_encoder=lambda payload: (0.,))
        observation = backend.start()
        proposal = backend.infer(observation)
        before = deepcopy({k: v for k, v in vars(tools).items() if k != 'sim'})
        backend.keepalive()
        tools.sim.request.assert_called_once_with('metadata')
        self.assertEqual(before, {k: v for k, v in vars(tools).items() if k != 'sim'})
        self.assertIs(backend._observation, observation)
        self.assertIs(backend._proposal, proposal)
        self.assertIsNone(backend.last_source)
        response = backend.auto_response(proposal, 1, 'carry')
        backend.execute(proposal, response, 'teacher')
        self.assertEqual(tools.physical_steps, 1)

    def test_failed_keepalive_blocks_execution_with_non_model_reason(self):
        from fast_harness.robodojo import RoboDojoBackend
        from fast_harness.types import HarnessError
        from test_robodojo import FakeTools
        tools = FakeTools()
        tools.sim = Mock()
        tools.sim.request.side_effect = TimeoutError('private rpc detail')
        backend = RoboDojoBackend(tools, feature_encoder=lambda payload: (0.,))
        proposal = backend.infer(backend.start())
        with self.assertRaises(HarnessControlError) as caught:
            backend.keepalive()
        self.assertEqual(caught.exception.code, 'sim_keepalive_failed')
        with self.assertRaises(HarnessError):
            backend.auto_response(proposal, 1, 'carry')
        self.assertEqual(tools.physical_steps, 0)

    def test_received_invalid_http_json_is_not_a_transport_stall(self):
        from fast_harness import reviewer as module
        from fast_harness.types import ReviewConstraintError
        from test_retry_extra import request
        response = Mock(status=200)
        response.read.return_value = b'not json private reply'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(module.urllib.request, 'build_opener', return_value=opener), \
                patch.dict(os.environ, {'OFFLINE_KEY': 'offline'}), \
                patch('socket.socket', side_effect=AssertionError('No network')):
            inner = module.ResponsesReviewer('https://example.invalid/v1/responses',
                                             'offline', 'OFFLINE_KEY')
            wrapper = module.TimedRetryReviewer(inner, retry_wait=120, schema_retries=0)
            with self.assertRaises(ReviewConstraintError) as caught:
                wrapper.review(request())
        self.assertEqual(caught.exception.code, 'response_envelope')
        self.assertEqual(wrapper.stalls, 0)
        self.assertEqual(wrapper.schema_rejects, 1)
        self.assertEqual(opener.open.call_count, 1)


class PassthroughToyBackend(ToyBackend):
    """Native-step-limited toy that rejects a second inference before consumption."""
    def __init__(self, native_steps=150, native_success=True):
        super().__init__(chunks=(native_steps + 14) // 15)
        self.native_steps, self.native_result = native_steps, native_success
        self.proposals = []

    def observation(self):
        observation = super().observation()
        terminal = self.step >= self.native_steps
        return replace(observation, terminal=terminal,
                       native_success=self.native_result if terminal else None,
                       payload=dict(observation.payload, remaining_steps=self.native_steps - self.step,
                                    max_episode_steps=self.native_steps))

    def infer(self, observation):
        if self.current is not None:
            raise AssertionError('The outstanding proposal must be consumed, not inferred again')
        self.current = replace(super().infer(observation),
                               max_steps=min(15, observation.payload['remaining_steps']))
        self.proposals.append(self.current)
        return self.current


class RecoveryPassthroughTests(unittest.TestCase):
    source = 'recovery_exhausted_passthrough'

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='recovery-tail-offline-', dir='/tmp')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.memory = Memory(':memory:', 'recovery-tail-offline', decay=1.0)
        self.addCleanup(self.memory.close)
        self.runs = 0
        for target in ('socket.socket', 'socket.create_connection', 'subprocess.Popen', 'time.sleep'):
            guard = patch(target, side_effect=AssertionError('Offline harness only'))
            mocked = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(mocked.assert_not_called)

    def harness(self, scenario='select', backend=None, **settings):
        from fast_harness.toy import decision
        from fast_harness.types import Review
        reviewer = ToyReviewer()
        reviewer.requests = []

        def review(request):
            reviewer.requests.append(request)
            self.assertFalse(request.observation.terminal, 'Passthrough must not request a final audit')
            phase = request.recovery['phase']
            warmup = scenario == 'empty_selection' and request.observation.step < 45
            outcome = 'unknown' if request.previous is None else ('ok' if warmup else 'error')
            stage = request.observation.payload['toy']['stage']
            response = decision(request.proposal, stage, failed=outcome == 'error',
                                steps=request.proposal.max_steps)
            if not warmup:
                response['assessment']['intent_status'] = 'misaligned'
            status = 'failed' if (phase == 'select' or (phase != 'normal' and
                request.recovery['attempt_chunks'] and scenario != 'total')) else 'continue'
            checkpoint = (dict(goal='Return to the observed toy position', prerequisites=[])
                          if warmup and outcome == 'ok' else None)
            return Review(stage, outcome, 'toy_progress_failure' if outcome == 'error' else 'none',
                          'Offline scripted recovery evidence.', response, checkpoint, status)

        reviewer.review = review
        config = dict(always_review=True, commit_gating=True, audit_probability=0,
                      max_episode_chunks=50, episode_number=2)
        config.update(settings)
        self.runs += 1
        harness = Harness(backend or PassthroughToyBackend(), reviewer, self.memory,
                          self.root / f'run-{self.runs}', Config(**config))
        if scenario == 'select':
            # Start at the selection boundary; the real Recovery.update decides exhaustion.
            harness.recovery.phase = 'select'
        return harness

    def events(self, harness):
        return [json.loads(line) for line in (harness.output / 'events.jsonl').read_text().splitlines()]

    def summary(self, harness):
        return json.loads((harness.output / 'summary.json').read_text())

    def assert_completed_tail(self, harness, result, reason='recovery_exhausted'):
        events = self.events(harness)
        entries = [e for e in events if e['event'] == self.source]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        tail = events[events.index(entry) + 1:]
        self.assertTrue(all(e['event'] in ('execution_requested', 'execution_completed', 'summary')
                            for e in tail))
        requested = [e for e in tail if e['event'] == 'execution_requested']
        completed = [e for e in tail if e['event'] == 'execution_completed']
        self.assertGreater(len(completed), 0)
        self.assertEqual([e['request_id'] for e in requested], [e['request_id'] for e in completed])
        self.assertEqual(requested[0]['request_id'], entry['request_id'])
        self.assertFalse(entry['review_verified'])
        self.assertEqual((entry['reason'], result['recovery_passthrough_reason']), (reason, reason))
        self.assertEqual(entry['step'], result['recovery_passthrough_started_step'])
        self.assertTrue(result['recovery_passthrough'])
        self.assertEqual(result['recovery_passthrough_chunks'], len(completed))
        self.assertEqual(result['recovery_passthrough_steps'],
                         sum(e['end_step'] - e['start_step'] for e in completed))
        self.assertTrue(all(e['source'] == self.source and e['mode'] == 'student' for e in completed))
        self.assertTrue(all(e['response']['assessment']['intent_status'] == 'uncertain' for e in requested))
        self.assertEqual((result['status'], result['complete']), ('completed', True))
        self.assertIs(result['native_success'], harness.backend.native_result)
        self.assertEqual((result['automatic_chunks'], result['teacher_steps']), (0, 0))
        self.assertEqual((result['degraded'], result['timeout_fallbacks'], result['degraded_chunks'],
                          result['degraded_steps']), (False, 0, 0, 0))
        self.assertEqual(result['control_steps'], harness.backend.step)
        self.assertEqual(result['student_steps'], result['control_steps'])
        self.assertEqual(result['teacher_calls'], len(harness.reviewer.requests))
        self.assertEqual(result['harness_revision'], 'tail-guard-1')
        self.assertEqual(events[0]['harness_revision'], result['harness_revision'])
        self.assertEqual(entry['harness_revision'], result['harness_revision'])
        actions = [e for e in events if e['event'] == 'execution_requested']
        self.assertEqual([e['request_id'] for e in actions], [p.request_id for p in harness.backend.proposals])
        self.assertEqual([e['step'] for e in actions], [p.observation.step for p in harness.backend.proposals])
        self.assertEqual(len({p.request_id for p in harness.backend.proposals}), len(actions))
        self.assertEqual(result['chunks'], len(actions))
        self.assertEqual(result['chunks'], harness.backend.executions)
        self.assertFalse(harness.backend.stopped)
        self.assertNotIn('control_error', result)
        self.assertEqual(self.summary(harness), result)
        self.assertEqual({k: events[-1][k] for k in result}, result)
        return entry, requested

    def test_empty_checkpoint_selection_with_candidates_enters_passthrough(self):
        harness = self.harness('empty_selection')
        result = harness.run()
        entry, _ = self.assert_completed_tail(harness, result)
        selections = [r for r in harness.reviewer.requests if r.recovery['phase'] == 'select']
        self.assertEqual(len(selections), 1)
        self.assertTrue(selections[0].checkpoints)
        self.assertIsNone(next(e['review']['selected_checkpoint_id'] for e in self.events(harness)
                               if e['event'] == 'review' and e['recovery']['phase'] == 'select'))
        self.assertEqual(entry['step'], 90)
        self.assertEqual(entry['recovery']['phase'], 'exhausted')
        self.assertEqual(result['checkpoints'], 2)

    def test_no_recovery_candidates_exhaust_after_two_local_attempts(self):
        harness = self.harness('no_candidates', attempt_chunks=1)
        result = harness.run()
        entry, _ = self.assert_completed_tail(harness, result)
        self.assertEqual(entry['step'], 30)
        self.assertEqual(entry['recovery']['total_chunks'], 2)
        self.assertEqual(entry['recovery']['phase'], 'exhausted')
        self.assertEqual(harness.checkpoints, [])
        self.assertTrue(all(not r.checkpoints for r in harness.reviewer.requests))
        self.assertEqual(result['recovery_events'], 1)
        self.assertEqual(result['chunks'] - result['recovery_passthrough_chunks'], 2)

    def test_total_recovery_budget_exhaustion_reuses_current_proposal(self):
        harness = self.harness('total', attempt_chunks=8, recovery_chunks=1)
        result = harness.run()
        entry, requested = self.assert_completed_tail(harness, result)
        self.assertEqual(entry['step'], 15)
        self.assertEqual(entry['recovery']['total_chunks'], 1)
        self.assertEqual(entry['recovery']['remaining_total_chunks'], 0)
        self.assertGreater(entry['recovery']['remaining_attempt_chunks'], 0)
        self.assertEqual(harness.recovery.total_steps, 1)  # Tail does not update recovery accounting.
        self.assertIs(harness.reviewer.requests[-1].proposal, harness.backend.proposals[1])
        self.assertEqual(requested[0]['request_id'], harness.backend.proposals[1].request_id)

    def test_eight_replans_reuse_one_proposal_then_enter_passthrough(self):
        harness = self.harness('no_candidates', max_episode_chunks=1)
        harness.recovery.start(0)
        # Model a non-converging replanner, not the Harness loop or passthrough implementation.
        with patch.object(harness.recovery, 'update', return_value=True) as update:
            result = harness.run()
        self.assertEqual(update.call_count, 8)
        self.assertEqual(result['teacher_calls'], 8)
        entry, _ = self.assert_completed_tail(harness, result, 'recovery_replanning_exhausted')
        self.assertEqual(entry['step'], 0)
        self.assertTrue(all(r.proposal is harness.backend.proposals[0] for r in harness.reviewer.requests))
        self.assertEqual(result['chunks'], result['recovery_passthrough_chunks'])

    def test_long_tail_bypasses_chunk_budget_clips_remaining_and_keeps_native_result(self):
        for native_success in (True, False):
            with self.subTest(native_success=native_success):
                harness = self.harness(backend=PassthroughToyBackend(15 * 145 + 4, native_success),
                                       max_episode_chunks=1, max_unreviewed_steps=2)
                result = harness.run()
                entry, requested = self.assert_completed_tail(harness, result)
                self.assertEqual(entry['step'], 0)
                self.assertEqual(result['recovery_passthrough_chunks'], 146)
                self.assertGreater(result['chunks'], 140)
                self.assertEqual([e['response']['steps'] for e in requested], [15] * 145 + [4])
                self.assertEqual(result['control_steps'], 15 * 145 + 4)
                self.assertEqual(result['teacher_calls'], 1)

    def test_tail_freezes_memory_checkpoints_and_spans_then_next_episode_reviews_normally(self):
        harness = self.harness('empty_selection', backend=PassthroughToyBackend(300),
                               episode_number=1, sim_attempt=3)
        self.memory.record_span('approach', 7)
        boundary = []
        original_auto = harness.backend.auto_response
        with patch.object(self.memory, 'learn_stage', wraps=self.memory.learn_stage) as learn, \
                patch.object(self.memory, 'record', wraps=self.memory.record) as record, \
                patch.object(self.memory, 'recovered', wraps=self.memory.recovered) as recovered, \
                patch.object(self.memory, 'record_span', wraps=self.memory.record_span) as span:
            def snapshot():
                return (tuple(self.memory.db.iterdump()), tuple(harness.checkpoints),
                        deepcopy(harness.recovery.context()),
                        tuple(m.call_count for m in (learn, record, recovered, span)))

            def auto(*args):
                if not boundary:
                    self.assertEqual(harness.metrics['recovery_passthrough_chunks'], 0)
                    self.assertEqual(harness.metrics['recovery_passthrough_steps'], 0)
                    boundary.append(snapshot())
                return original_auto(*args)

            with patch.object(harness.backend, 'auto_response', side_effect=auto):
                result = harness.run()
            self.assert_completed_tail(harness, result)
            self.assertEqual(len(boundary), 1)
            self.assertEqual(snapshot(), boundary[0])
            self.assertGreater(record.call_count, 0)  # Reviewed history before entry is retained.
        previous_events = self.memory.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        reviewer = ToyReviewer()
        with patch.object(reviewer, 'review', wraps=reviewer.review) as review, \
                patch.object(self.memory, 'record_span', wraps=self.memory.record_span) as span:
            second = Harness(ToyBackend(chunks=6), reviewer, self.memory, self.root / 'ep2',
                             Config(always_review=True, commit_gating=True, episode_number=2)).run()
            self.assertEqual(review.call_count, 7)
            self.assertTrue(review.call_args.args[0].observation.terminal)
            self.assertGreater(span.call_count, 0)
        self.assertEqual((second['complete'], second['native_success'], second['status']),
                         (True, True, 'completed'))
        self.assertEqual((second['recovery_passthrough'], second['recovery_passthrough_chunks'],
                          second['recovery_passthrough_steps']), (False, 0, 0))
        self.assertIsNone(second['recovery_passthrough_reason'])
        self.assertIsNone(second['recovery_passthrough_started_step'])
        self.assertFalse(second['degraded'])
        self.assertGreater(self.memory.db.execute('SELECT COUNT(*) FROM events').fetchone()[0], previous_events)
        self.assertGreater(sum(self.memory.statistics(s).total for s in self.memory.stages()), 0)

    def test_invalid_proposals_before_entry_or_during_tail_remain_errors(self):
        for index in (0, 1):
            for kind in ('blocked', 'empty_id', 'stale_step', 'stale_episode', 'zero_horizon',
                         'large_horizon', 'bool_horizon', 'reused_id'):
                if index == 0 and kind == 'reused_id':
                    continue
                with self.subTest(index=index, kind=kind):
                    harness = self.harness(backend=PassthroughToyBackend(45))
                    backend = harness.backend
                    original = backend.infer

                    def infer(observation):
                        proposal = original(observation)
                        if len(backend.proposals) != index + 1:
                            return proposal
                        changes = {
                            'blocked': dict(blocked=True), 'empty_id': dict(request_id=''),
                            'stale_step': dict(observation=replace(observation, step=observation.step + 1)),
                            'stale_episode': dict(observation=replace(observation, episode_id='foreign')),
                            'zero_horizon': dict(max_steps=0), 'large_horizon': dict(max_steps=16),
                            'bool_horizon': dict(max_steps=True),
                            'reused_id': dict(request_id=backend.proposals[0].request_id),
                        }
                        return replace(proposal, **changes[kind])

                    with patch.object(backend, 'infer', side_effect=infer), \
                            self.assertRaises(HarnessControlError) as caught:
                        harness.run()
                    code = 'invalid_proposal' if kind == 'blocked' or kind.endswith('horizon') else 'stale_proposal'
                    self.assertEqual(caught.exception.code, code)
                    result = self.summary(harness)
                    self.assertEqual(result['control_error'], dict(code=code))
                    self.assertEqual((result['chunks'], result['recovery_passthrough_chunks'], backend.executions),
                                     (index, index, index))
                    self.assertEqual(result['recovery_passthrough'], index == 1)
                    self.assertFalse(result['complete'])
                    self.assertIsNone(result['native_success'])
                    self.assertTrue(backend.stopped)
                    self.assertEqual(len(backend.proposals), index + 1)
                    self.assertEqual(len(harness.reviewer.requests), index)

    def test_invalid_tail_response_has_entry_but_zero_acknowledged_chunks(self):
        cases = [('mode', 'eef', 'automatic_nonstudent_action'),
                 ('request_id', 'foreign', 'response_request_mismatch')]
        cases += [('steps', value, 'response_horizon') for value in (0, 16, True, 1.0)]
        for key, value, code in cases:
            with self.subTest(key=key, value=value):
                harness = self.harness()
                original = harness.backend.auto_response
                with patch.object(harness.backend, 'auto_response',
                                  side_effect=lambda *args: dict(original(*args), **{key: value})), \
                        self.assertRaises(HarnessControlError) as caught:
                    harness.run()
                self.assertEqual(caught.exception.code, code)
                result = self.summary(harness)
                self.assertTrue(result['recovery_passthrough'])
                self.assertEqual((result['chunks'], result['recovery_passthrough_chunks'],
                                  result['recovery_passthrough_steps'], harness.backend.executions), (0, 0, 0, 0))
                events = self.events(harness)
                self.assertEqual(sum(e['event'] == self.source for e in events), 1)
                self.assertFalse(any(e['event'].startswith('execution_') for e in events))
                self.assertFalse(result['complete'])
                self.assertTrue(harness.backend.stopped)

    def test_invalid_tail_ack_is_not_counted_retried_or_promoted_to_terminal(self):
        for index in (0, 1):
            for kind in ('zero', 'backward', 'overshoot', 'foreign_episode'):
                with self.subTest(index=index, kind=kind):
                    harness = self.harness(backend=PassthroughToyBackend(45))
                    backend = harness.backend
                    original = backend.execute

                    def execute(proposal, response, source):
                        after = original(proposal, response, source)
                        if backend.executions != index + 1:
                            return after
                        step = {'zero': proposal.observation.step, 'backward': proposal.observation.step - 1,
                                'overshoot': proposal.observation.step + response['steps'] + 1,
                                'foreign_episode': after.step}[kind]
                        return replace(after, step=step, terminal=True, native_success=True,
                                       episode_id='foreign' if kind == 'foreign_episode' else after.episode_id)

                    with patch.object(backend, 'execute', side_effect=execute), \
                            self.assertRaises(HarnessControlError) as caught:
                        harness.run()
                    self.assertEqual(caught.exception.code, 'execution_acknowledgement')
                    result = self.summary(harness)
                    self.assertEqual(result['control_error'], dict(code='execution_acknowledgement'))
                    self.assertTrue(result['recovery_passthrough'])
                    self.assertEqual((result['chunks'], result['recovery_passthrough_chunks']), (index, index))
                    self.assertEqual((result['control_steps'], result['recovery_passthrough_steps']),
                                     (index * 15, index * 15))
                    self.assertEqual(backend.executions, index + 1)
                    self.assertEqual(len(backend.proposals), index + 1)
                    self.assertEqual(result['teacher_calls'], 1)
                    self.assertFalse(result['complete'])
                    self.assertIsNone(result['native_success'])
                    self.assertTrue(backend.stopped)
                    events = self.events(harness)
                    self.assertEqual(sum(e['event'] == self.source for e in events), 1)
                    self.assertEqual(sum(e['event'] == 'execution_completed' for e in events), index)


if __name__ == '__main__':
    unittest.main()

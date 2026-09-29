import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fast_harness.engine import Harness
from fast_harness.memory import Memory, Statistics
from fast_harness.recovery import Recovery
from fast_harness.scheduler import Config, Scheduler, interval
from fast_harness.toy import ToyBackend, ToyReviewer
from fast_harness.types import Checkpoint, HarnessError, Observation, Review


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(':memory:', 'task:policy:features')
        self.addCleanup(self.memory.close)

    def record(self, step=0, outcome='ok', student=True):
        return self.memory.record(episode='ep', step=step, kind='execution', stage='approach',
            features=(0., 0.), outcome=outcome, evidence='observed', student=student, error_type='miss')

    def test_unknown_and_correction_do_not_count_as_student_success(self):
        self.record(outcome='unknown')
        self.record(step=1, student=False)
        self.assertEqual(self.memory.statistics('approach').total, 0)

    def test_deduplicates_observed_outcome(self):
        self.record()
        self.record()
        self.assertEqual(self.memory.statistics('approach').total, 1)

    def test_errors_count_and_recovery_evidence(self):
        error = self.record(outcome='error')
        self.record(step=1, outcome='error')
        self.memory.recovered({error}, 'returned to a valid subgoal')
        row = self.memory.errors()[0]
        self.assertEqual((row['count'], row['recovered']), (2, 1))
        self.assertTrue(self.memory.near_error('approach', (0., 0.), 0.01))

    def test_unknown_and_ambiguous_stage_force_review(self):
        self.assertIsNone(self.memory.match((0.,), .1, .01).stage)
        self.memory.learn_stage('a', (0.,))
        self.memory.learn_stage('b', (.001,))
        self.assertEqual(self.memory.match((0.,), .1, .01).reason, 'ambiguous_stage')
        self.assertEqual(self.memory.match((1.,), .1, .01).reason, 'unknown_context')

    def test_persistence_and_namespace_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'memory.sqlite'
            first = Memory(path, 'a')
            first.learn_stage('pick', (0.1,))
            first.close()
            second = Memory(path, 'a')
            other = Memory(path, 'b')
            self.assertEqual(second.stages(), ('pick',))
            self.assertEqual(other.stages(), ())
            second.close()
            other.close()

    def test_feature_dimension_change_is_unknown(self):
        self.memory.learn_stage('a', (0., 1.))
        self.assertIsNone(self.memory.match((0.,), 100., 0.).stage)
        with self.assertRaises(ValueError):
            self.memory.learn_stage('a', (float('nan'),))


class SchedulerTests(unittest.TestCase):
    def test_frequency_improves_with_evidence_not_just_percentage(self):
        config = Config()
        self.assertEqual(interval(Statistics(1, 0, 1, 1), config), 1)
        self.assertGreater(interval(Statistics(40, 0, 40, 40), config), 1)
        self.assertGreater(interval(Statistics(40, 0, 40, 40), config),
                           interval(Statistics(32, 8, 40, 10, True), config))
        self.assertEqual(interval(Statistics(40, 1, 41, 0, True), config), 1)

    def test_invalid_config(self):
        for kwargs in ({'max_interval': 0}, {'audit_probability': 2}, {'attempt_chunks': True}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_error_memory_cools_down_after_verified_clean_streak(self):
        memory = Memory(':memory:', 'a')
        self.addCleanup(memory.close)
        backend = ToyBackend()
        obs = backend.start()
        proposal = backend.infer(obs)
        memory.learn_stage('approach', obs.features)
        for step in range(20):
            memory.record(episode='x', step=step, kind='execution', stage='approach',
                features=obs.features, outcome='error' if step == 0 else 'ok', evidence='feedback', student=True)
        scheduler = Scheduler(Config(audit_probability=0))
        reason, _, _ = scheduler.reason(memory, memory.match(obs.features, .1, .01), obs, proposal,
            last_review_step=0, chunks_since_review=0, previous_stage='approach',
            recovery_active=False, force_review=False, stalled=False)
        self.assertIsNone(reason)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.points = [Checkpoint(str(i), Observation('ep', i, (0.,), {}), 'pick', 'reach', ())
                       for i in (1, 2, 3)]
        self.recovery = Recovery(3, 50)
        self.recovery.start(10)

    def review(self, status='failed', selected=None):
        return Review('pick', 'ok' if status == 'succeeded' else 'error', '', 'observed', None,
                      recovery_status=status, selected_checkpoint_id=selected)

    def attempt(self, status):
        self.recovery.executed()
        self.recovery.update(self.review(status), self.points)

    def test_full_escalation_and_teacher_selection(self):
        self.attempt('failed')
        self.assertEqual(self.recovery.phase, 'local')
        self.attempt('failed')
        self.assertEqual(self.recovery.target.id, '3')
        self.assertEqual(self.recovery.phase, 'return_latest')
        self.attempt('succeeded')
        self.assertEqual(self.recovery.phase, 'after_return')
        self.attempt('failed')
        self.attempt('failed')
        self.assertEqual(self.recovery.phase, 'select')
        self.recovery.update(self.review(selected='1'), self.points)
        self.assertEqual(self.recovery.phase, 'return_selected')
        self.attempt('succeeded')
        self.attempt('succeeded')
        self.assertEqual(self.recovery.phase, 'normal')

    def test_attempt_is_multiple_chunks_and_times_out(self):
        for _ in range(2):
            self.attempt('continue')
            self.assertEqual(self.recovery.attempts_failed, 0)
        self.attempt('continue')
        self.assertEqual(self.recovery.attempts_failed, 1)

    def test_no_checkpoint_is_explicitly_unrecoverable(self):
        self.points.clear()
        self.attempt('failed')
        self.attempt('failed')
        self.assertEqual(self.recovery.phase, 'exhausted')

    def test_rejected_or_missing_historical_checkpoint(self):
        self.recovery.phase = 'select'
        with self.assertRaises(ValueError):
            self.recovery.update(self.review(selected='invented'), self.points)
        self.recovery.update(self.review(selected=None), self.points)
        self.assertEqual(self.recovery.phase, 'exhausted')

    def test_total_budget_is_not_reset_by_new_recovery(self):
        self.recovery.total_steps = 50
        self.recovery.start(20)
        self.assertEqual(self.recovery.phase, 'exhausted')

    def test_error_cannot_be_successful_recovery(self):
        self.recovery.executed()
        with self.assertRaises(ValueError):
            self.recovery.update(replace(self.review('succeeded'), last_outcome='error'), self.points)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.memory = Memory(self.root / 'experience.sqlite', 'toy:v1')
        self.addCleanup(self.memory.close)

    def run_episode(self, name='episode', backend=None, reviewer=None, config=None):
        harness = Harness(backend or ToyBackend(), reviewer or ToyReviewer(), self.memory,
                          self.root / name, config or Config(audit_probability=0))
        summary = harness.run()
        events = [json.loads(line) for line in (self.root / name / 'events.jsonl').read_text().splitlines()]
        return summary, events

    def test_repeated_task_reduces_reviews_without_fabricated_tokens(self):
        first, _ = self.run_episode('first')
        later, events = self.run_episode('later')
        self.assertTrue(first['native_success'] and later['native_success'])
        self.assertLess(later['teacher_calls'], first['teacher_calls'])
        self.assertGreater(later['automatic_chunks'], 0)
        self.assertIsNone(later['input_tokens'])
        auto = [e for e in events if e['event'] == 'execution_requested' and e['source'] == 'harness_auto']
        self.assertTrue(all(e['response']['mode'] == 'student' for e in auto))

    def test_always_review_baseline_and_terminal_audit(self):
        summary, events = self.run_episode(config=Config(always_review=True))
        self.assertEqual(summary['automatic_chunks'], 0)
        self.assertEqual(summary['teacher_calls'], summary['chunks'] + 1)
        terminal = [e for e in events if e['event'] == 'review' and e['reason'] == 'terminal_audit']
        self.assertEqual(len(terminal), 1)
        self.assertIsNone(terminal[0]['review']['response'])

    def test_fault_recovery_reaches_older_checkpoint(self):
        summary, events = self.run_episode(backend=ToyBackend(fault=True))
        self.assertTrue(summary['native_success'])
        phases = [e['recovery']['phase'] for e in events if e['event'] == 'recovery_transition']
        for phase in ('local', 'return_latest', 'after_return', 'select', 'return_selected', 'normal'):
            self.assertIn(phase, phases)
        self.assertTrue(self.memory.errors())

    def test_hard_unreviewed_control_step_limit(self):
        self.run_episode('warmup')
        _, events = self.run_episode(config=Config(max_unreviewed_steps=17, audit_probability=0))
        last_review = 0
        for event in events:
            if event['event'] == 'review':
                last_review = event['step']
            if event['event'] == 'execution_completed' and event['source'] == 'harness_auto':
                self.assertLessEqual(event['end_step'] - last_review, 17)

    def test_ambiguous_execution_is_never_retried(self):
        class Broken(ToyBackend):
            attempts = 0
            def execute(self, *args):
                self.attempts += 1
                raise TimeoutError('Uncertain acknowledgement')
        backend = Broken()
        with self.assertRaises(TimeoutError):
            self.run_episode(backend=backend)
        self.assertEqual(backend.attempts, 1)
        summary = json.loads((self.root / 'episode' / 'summary.json').read_text())
        self.assertFalse(summary['complete'])

    def test_no_overwrite_and_decision_budget(self):
        with self.assertRaises(HarnessError):
            self.run_episode(config=Config(max_episode_chunks=1))
        with self.assertRaises(FileExistsError):
            self.run_episode()

    def test_nonfinite_proposal_never_executes(self):
        class Broken(ToyBackend):
            def infer(self, observation):
                return replace(super().infer(observation), blocked=True)
        backend = Broken()
        with self.assertRaises(HarnessError):
            self.run_episode(backend=backend)
        self.assertEqual(backend.executions, 0)


if __name__ == '__main__':
    unittest.main()

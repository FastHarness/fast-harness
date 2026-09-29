import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fast_harness import cli
from fast_harness.engine import Harness
from fast_harness.memory import Memory
from fast_harness.recovery import Recovery
from fast_harness.robodojo import RoboDojoBackend
from fast_harness.scheduler import Config
from fast_harness.toy import ToyBackend, ToyReviewer, decision
from fast_harness.types import Checkpoint, Observation, Review, Usage
from test_robodojo import FakeTools


class RegressionTests(unittest.TestCase):
    def test_native_success_survives_terminal_reviewer_failure(self):
        class BrokenReviewer(ToyReviewer):
            def review(self, request):
                if request.observation.terminal:
                    raise TimeoutError('Final audit timed out')
                return super().review(request)
        with tempfile.TemporaryDirectory() as root:
            memory = Memory(':memory:', 'test')
            self.addCleanup(memory.close)
            output = Path(root) / 'run'
            harness = Harness(ToyBackend(chunks=2), BrokenReviewer(), memory, output)
            with self.assertRaises(TimeoutError):
                harness.run()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertTrue(summary['native_success'])
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['status'], 'audit_failed')

    def test_native_failure_is_complete_not_infrastructure_failure(self):
        class FailedTask(ToyBackend):
            def observation(self):
                result = super().observation()
                return replace(result, native_success=False) if result.terminal else result
        with tempfile.TemporaryDirectory() as root:
            memory = Memory(':memory:', 'test')
            self.addCleanup(memory.close)
            summary = Harness(FailedTask(chunks=2), ToyReviewer(), memory, Path(root) / 'run').run()
            self.assertFalse(summary['native_success'])
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['status'], 'completed')

    def test_small_physical_budget_also_caps_reviewed_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            memory = Memory(':memory:', 'test')
            self.addCleanup(memory.close)
            output = Path(root) / 'run'
            Harness(ToyBackend(chunks=3), ToyReviewer(), memory, output,
                    Config(max_unreviewed_steps=2, always_review=True)).run()
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            actions = [event for event in events if event['event'] == 'execution_completed']
            self.assertTrue(actions)
            self.assertTrue(all(event['end_step'] - event['start_step'] <= 2 for event in actions))

    def test_new_failure_can_reuse_previously_reached_checkpoint(self):
        point = Checkpoint('recent', Observation('ep', 2, (0.,), {}), 'pick', 'reach', ())
        recovery = Recovery(2, 20)
        recovery.start(5)
        recovery.excluded.add(point.id)
        recovery.total_steps = 3
        recovery.phase = 'normal'
        recovery.start(8)
        self.assertEqual(recovery.candidates([point]), (point,))
        self.assertEqual(recovery.total_steps, 3)

    def test_after_return_goal_is_original_problem_not_checkpoint(self):
        point = Checkpoint('old', Observation('ep', 2, (0.,), {}), 'pick', 'Return to approach', ())
        recovery = Recovery(3, 40)
        recovery.start(10, 'Finish placing the object')
        recovery.target = point
        recovery.phase = 'return_latest'
        self.assertEqual(recovery.context()['goal'], 'Return to approach')
        recovery.phase = 'after_return'
        self.assertEqual(recovery.context()['goal'], 'Finish placing the object')

    def test_nearby_occurrences_share_error_count(self):
        memory = Memory(':memory:', 'task')
        self.addCleanup(memory.close)
        for step, features in enumerate(((0., 0.), (0.01, 0.01))):
            memory.record(episode='ep', step=step, kind='execution', stage='pick', features=features,
                          outcome='error', evidence='missed grasp', student=True, error_type='miss')
        self.assertEqual(len(memory.errors()), 1)
        self.assertEqual(memory.errors()[0]['count'], 2)

    def test_live_cli_with_nested_memory_and_real_adapter_contract(self):
        class Student:
            metadata = dict(checkpoint_sha256='test-policy-sha', config='test-config')
            def __init__(self, *args):
                pass
            def close(self):
                pass
        class Tools(FakeTools):
            class _Sim:
                def request(self, name):
                    return {}
            def __init__(self, output, *args, **kwargs):
                super().__init__()
                self.sim = self._Sim()
                self.output = output
                self.assert_output_exists = output.is_dir()
                self.outcome = dict(complete=True, terminated=True, truncated=False, success=True)
        class Reviewer:
            def __init__(self, *args, **kwargs):
                pass
            def review(self, request):
                response = decision(request.proposal, 'move') if request.proposal else None
                return Review('move', 'ok' if request.previous else 'unknown', 'none',
                              'Observed test transition.', response)
        runtime = types.ModuleType('robodojo_runtime')
        runtime.PolicyClient = Student
        runtime.RoboDojoTools = Tools
        modules = {'robodojo_runtime': runtime}
        namespaces = []
        for protocol, reviewer_class, suffix, key_env in (
                ('responses', 'ResponsesReviewer', 'responses', 'OPENAI_API_KEY'),
                ('anthropic', 'AnthropicReviewer', 'v1/messages', 'ANTHROPIC_API_KEY')):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as root:
                output = Path(root) / 'run'
                args = argparse.Namespace(allow_model_requests=True, output=output, upstream=None,
                    runtime_module='robodojo_runtime',
                    endpoint=f'https://example.invalid/{suffix}', model='test', api_key_env=None,
                    reviewer=protocol, max_tokens=4096, max_episode_chunks=1000,
                    timeout=120, effort=None, always_review=False, seed=0, max_interval=6,
                    episode_number=None, sim_attempt=1,
                    max_unreviewed_steps=60, student_port=1, sim_port=2, checkpoint=Path(root) / 'weights',
                    task='test', scope='fixed', memory=output / 'experience.sqlite')
                with patch.dict(sys.modules, modules), \
                     patch('fast_harness.reviewer.' + reviewer_class, side_effect=Reviewer) as factory, \
                     patch('fast_harness.robodojo.RoboDojoBackend',
                           side_effect=lambda tools: RoboDojoBackend(tools, feature_encoder=lambda _: (0.,))), \
                     contextlib.redirect_stdout(io.StringIO()):
                    cli.live(args)
                factory.assert_called_once()
                self.assertEqual(factory.call_args.args[2], key_env)
                if protocol == 'anthropic':
                    self.assertEqual(factory.call_args.kwargs['max_tokens'], 4096)
                summary = json.loads((output / 'harness' / 'summary.json').read_text())
                self.assertTrue(summary['native_success'])
                self.assertEqual(summary['chunks'], 1)
                identity = json.loads((output / 'identity.json').read_text())
                self.assertEqual(identity['reviewer_protocol'], protocol)
                self.assertNotIn(args.endpoint, json.dumps(identity))
                self.assertTrue(args.memory.is_file())
                first_event = json.loads((output / 'harness' / 'events.jsonl').read_text().splitlines()[0])
                namespaces.append(first_event['namespace'])
        self.assertNotEqual(*namespaces)

    def test_cache_creation_usage_survives_engine_summary(self):
        class CachedReviewer(ToyReviewer):
            def review(self, request):
                return replace(super().review(request), usage=Usage(15, 3, 5, 2))
        with tempfile.TemporaryDirectory() as root:
            memory = Memory(':memory:', 'test')
            self.addCleanup(memory.close)
            summary = Harness(ToyBackend(chunks=2), CachedReviewer(), memory, Path(root) / 'run').run()
            self.assertEqual(summary['cache_creation_input_tokens'], 2 * summary['teacher_calls'])
            self.assertEqual(summary['input_tokens'], 15 * summary['teacher_calls'])


if __name__ == '__main__':
    unittest.main()

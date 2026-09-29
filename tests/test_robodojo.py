import copy
from dataclasses import replace
import importlib
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from fast_harness.robodojo import CAMERAS, RoboDojoBackend, encode_features
from fast_harness.types import HarnessError


class FakeTools:
    def __init__(self, *, remaining=40, count=50):
        self.phase = 'start'
        self.episode = None
        self.tick = 0
        self.remaining = remaining
        self.limit = remaining
        self.decision = 0
        self.actions = [[0.0] * 6 + [1.0] + [0.0] * 6 + [1.0] for _ in range(count)]
        self.state = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
        self.calls = []
        self.validations = 0
        self.physical_steps = 0
        self.history = []
        self.run = {'teacher': 'codex_tools', 'teacher_off': False}
        self.source = 'student'
        self.result = None
        self.outcome = None
        self.actual_steps = None
        self.fail = None
        self.diagnostic_changes = {}
        self.request_changes = {}

    def next_call(self):
        root = '/external/rollout'
        if self.phase == 'start':
            return dict(tool='robodojo_start', task='task', output_dir=f'{root}/observations/000')
        if self.phase == 'infer':
            return dict(tool='pi05_infer', observation_path=self.observation_path,
                        output_dir=f'{root}/proposals/{self.decision:03d}')
        if self.phase == 'execute':
            return dict(tool='robodojo_execute', proposal_path=self.proposal_path,
                        output_dir=f'{root}/observations/{self.decision + 1:03d}',
                        response='Supply a fresh gate response')
        return None

    def _check(self, name, arguments):
        expected = self.next_call()
        assert expected is not None and expected['tool'] == name
        assert set(arguments) == set(expected) - {'tool'}
        assert all(arguments[key] == value for key, value in expected.items()
                   if key not in ('tool', 'response'))
        self.calls.append((name, copy.deepcopy(arguments)))

    def _packet(self):
        return dict(
            episode_id=self.episode, step_id=self.tick, task='task', instruction='Move the item.',
            remaining_steps=self.remaining, max_episode_steps=self.limit,
            current_state=list(self.state),
            images=[dict(camera=camera, path=f'{self.observation_path}/{camera}.png') for camera in CAMERAS],
            current_eef={arm: dict(position=[0.1, 0.2, 0.3], quaternion_wxyz=[1.00001, 0, 0, 0],
                                   gripper_opening_command=self.state[i * 7 + 6])
                         for i, arm in enumerate(('left', 'right'))},
            observation_path=self.observation_path, robot_profile='robodojo',
            previous_result=copy.deepcopy(self.history[-1]) if self.history else None,
            result=self.result, rollout_finished=self.phase == 'done', next_call=self.next_call(),
            reward=900, object_truth={'hidden': True}, native_success=True,
        )

    def start(self, **arguments):
        self._check('robodojo_start', arguments)
        self.episode = 'episode-1'
        if self.fail == 'start':
            raise RuntimeError('start failed after reset')
        self.observation_path = f"{arguments['output_dir']}/observation.json"
        self.phase = 'infer'
        return self._packet()

    def infer(self, **arguments):
        self._check('pi05_infer', arguments)
        if self.fail == 'infer':
            raise RuntimeError('inference failed')
        self.proposal_path = f"{arguments['output_dir']}/actions.npz"
        self.request = dict(self._packet(), request_id=f'request-{self.decision}',
                            gate_policy='failure-or-intent', proposal_path=self.proposal_path,
                            action_diagnostics=dict(shape=[len(self.actions), 14], finite=True,
                                                    status='computed', opening_range=[[1, 1], [1, 1]],
                                                    max_joint_step_rad=0.0))
        self.request['action_diagnostics'].update(self.diagnostic_changes)
        self.request.update(self.request_changes)
        self.phase = 'execute'
        self.request['next_call'] = self.next_call()
        return copy.deepcopy(self.request)

    def validate(self, response, request):
        self.validations += 1
        if self.fail == 'validate':
            raise ValueError('native validation rejected response')
        assert response['request_id'] == request['request_id']
        assert response['mode'] in ('student', 'edit', 'eef', 'stop')
        assert response['assessment']['intent_status'] in ('uncertain', 'aligned', 'misaligned')

    def execute(self, **arguments):
        self._check('robodojo_execute', arguments)
        response = arguments['response']
        self.validate(response, self.request)
        if response['mode'] == 'stop':
            result = self.finish('model_stop')
            self.result = result
        else:
            count = response['steps'] if self.actual_steps is None else self.actual_steps
            self.physical_steps += count
            self.tick += count
            self.remaining -= count
            if self.fail == 'execute':
                raise TimeoutError('ACK lost after physical execution')
            native_end = bool(self.outcome and self.outcome.get('complete'))
            self.history.append(dict(terminal=native_end, native_success=False,
                                     reward=123, object_truth={'pose': [1, 2, 3]},
                                     executed_steps=count, source='student', source_detail=response['mode']))
            self.result = copy.deepcopy(self.outcome)
            self.phase = 'done' if self.outcome else 'infer'
        self.decision += 1
        self.observation_path = f"{arguments['output_dir']}/observation.json"
        return self._packet()

    def finish(self, reason):
        self.calls.append(('finish', reason))
        if self.fail == 'finish':
            raise TimeoutError('finish ACK lost')
        self.phase = 'done'
        return dict(complete=False, terminated=False, truncated=False, success=False, reason=reason)

    def close(self):
        self.calls.append(('close', {}))


def backend(tools=None, **kwargs):
    tools = tools or FakeTools()
    return RoboDojoBackend(tools, feature_encoder=lambda payload: (0.5,) * 14, **kwargs)


def ready(tools=None, **kwargs):
    adapter = backend(tools, **kwargs)
    observation = adapter.start()
    return adapter, observation, adapter.infer(observation)


class BackendTests(unittest.TestCase):
    def test_strict_next_calls_and_fresh_cycle(self):
        adapter, before, proposal = ready()
        response = adapter.auto_response(proposal, 15, 'approach')
        after = adapter.execute(proposal, response, 'harness_auto')
        fresh = adapter.infer(after)
        self.assertIs(proposal.observation, before)
        self.assertIs(fresh.observation, after)
        self.assertEqual(after.step, 15)
        self.assertNotEqual(fresh.request_id, proposal.request_id)
        self.assertNotEqual(fresh.payload['proposal_path'], proposal.payload['proposal_path'])
        self.assertEqual(adapter.tools.validations, 1)
        self.assertEqual([call[0] for call in adapter.tools.calls],
                         ['robodojo_start', 'pi05_infer', 'robodojo_execute', 'pi05_infer'])
        self.assertTrue(all('tool' not in call[1] for call in adapter.tools.calls))

    def test_stale_objects_requests_and_observation_paths(self):
        adapter, observation, proposal = ready()
        response = adapter.auto_response(proposal, 1, 'stage')
        for wrong in (replace(proposal), replace(proposal, request_id='old')):
            with self.assertRaises(HarnessError):
                adapter.execute(wrong, response, 'auto')
        with self.assertRaises(HarnessError):
            adapter.infer(replace(observation))
        with self.assertRaises(HarnessError):
            adapter.execute(proposal, dict(response, request_id='old'), 'auto')
        self.assertEqual(len(adapter.tools.calls), 2)
        after = adapter.execute(proposal, response, 'auto')
        with self.assertRaises(HarnessError):
            adapter.execute(proposal, response, 'auto')
        with self.assertRaises(HarnessError):
            adapter.infer(observation)
        adapter.tools.observation_path = '/wrong/observation.json'
        with self.assertRaises(HarnessError):
            adapter.infer(after)

    def test_external_identity_changes_are_rejected(self):
        for field, value in (('episode', 'foreign'), ('tick', 9), ('proposal_path', '/wrong')):
            with self.subTest(field=field):
                adapter, _, proposal = ready()
                response = adapter.auto_response(proposal, 1, 'stage')
                setattr(adapter.tools, field, value)
                with self.assertRaises(HarnessError):
                    adapter.execute(proposal, response, 'auto')
                self.assertEqual(len(adapter.tools.calls), 2)
        adapter, _, proposal = ready()
        adapter.tools.request['request_id'] = 'stale'
        with self.assertRaises(HarnessError):
            adapter.auto_response(proposal, 1, 'stage')

    def test_repeat_infer_start_and_reused_request_rejected(self):
        adapter, observation, proposal = ready()
        with self.assertRaises(HarnessError):
            adapter.start()
        with self.assertRaises(HarnessError):
            adapter.infer(observation)
        after = adapter.execute(proposal, adapter.auto_response(proposal, 1, 'stage'), 'auto')
        adapter.tools.request_changes = {'request_id': proposal.request_id}
        with self.assertRaisesRegex(HarnessError, 'fresh request_id'):
            adapter.infer(after)
        self.assertEqual(adapter.tools.physical_steps, 1)

    def test_reused_proposal_path_and_stale_infer_packet(self):
        for changes in ({'proposal_path': '/external/rollout/proposals/000/actions.npz'}, {'step_id': 0}):
            adapter, _, proposal = ready()
            after = adapter.execute(proposal, adapter.auto_response(proposal, 1, 'stage'), 'auto')
            adapter.tools.request_changes = changes
            with self.assertRaises(HarnessError):
                adapter.infer(after)

    def test_auto_schema_is_honest_and_fresh(self):
        adapter, observation, proposal = ready()
        response = adapter.auto_response(proposal, 2, 'non-English stage: \u5b8c\u6210')
        self.assertEqual(set(response), {'request_id', 'mode', 'steps', 'reason', 'edit', 'target', 'assessment'})
        self.assertEqual(response['request_id'], proposal.request_id)
        self.assertEqual(response['mode'], 'student')
        self.assertIn('Harness automatic', response['reason'])
        self.assertIn('no GPT visual review', response['reason'])
        assessment = response['assessment']
        self.assertEqual(assessment['execution_status'], 'not_started')
        self.assertEqual(assessment['intent_status'], 'uncertain')
        self.assertEqual(assessment['task_progress']['verified_completed'], [])
        self.assertTrue(assessment['task_progress']['remaining'])
        self.assertTrue(str(response).isascii())
        for arm in ('left', 'right'):
            self.assertEqual(response['edit'][arm], dict(delta_position=[0, 0, 0],
                             delta_rotation_vector=[0, 0, 0], gripper='keep'))
            target = response['target'][arm]
            self.assertEqual(set(target), {'position', 'quaternion_wxyz', 'gripper_closed'})
            self.assertEqual(target['position'], observation.payload['current_eef'][arm]['position'])
            self.assertAlmostEqual(math.hypot(*target['quaternion_wxyz']), 1)
            self.assertIs(target['gripper_closed'], False)
        next_observation = adapter.execute(proposal, response, 'auto')
        fresh = adapter.infer(next_observation)
        new_response = adapter.auto_response(fresh, 1, 'stage')
        self.assertEqual(new_response['assessment']['execution_status'], 'uncertain')
        self.assertNotEqual(new_response['request_id'], response['request_id'])
        self.assertEqual(new_response['assessment']['task_progress']['verified_completed'], [])

    def test_native_validation_is_not_bypassed(self):
        adapter, _, proposal = ready()
        response = adapter.auto_response(proposal, 1, 'stage')
        adapter.tools.fail = 'validate'
        with self.assertRaisesRegex(HarnessError, 'native validation rejected'):
            adapter.execute(proposal, response, 'reviewer')
        self.assertEqual(adapter.tools.validations, 1)
        self.assertEqual(adapter.tools.physical_steps, 0)
        with self.assertRaises(HarnessError):
            adapter.execute(proposal, response, 'reviewer')
        self.assertEqual(adapter.tools.validations, 1)

    def test_uncertain_execution_never_retries(self):
        adapter, _, proposal = ready()
        response = adapter.auto_response(proposal, 2, 'stage')
        adapter.tools.fail = 'execute'
        with self.assertRaisesRegex(HarnessError, 'no retry'):
            adapter.execute(proposal, response, 'auto')
        with self.assertRaises(HarnessError):
            adapter.execute(proposal, response, 'auto')
        with self.assertRaises(HarnessError):
            adapter.infer(proposal.observation)
        self.assertEqual(adapter.tools.physical_steps, 2)
        self.assertEqual(adapter.tools.validations, 1)
        adapter.stop('transport_error')
        adapter.stop('second_stop')
        self.assertEqual([name for name, _ in adapter.tools.calls].count('finish'), 1)

    def test_stop_only_live_and_not_done_and_close_is_separate(self):
        adapter = backend()
        adapter.stop('not_started')
        self.assertEqual(adapter.tools.calls, [])
        adapter.start()
        adapter.stop('operator')
        adapter.stop('operator_again')
        adapter.close()
        adapter.close()
        self.assertEqual([name for name, _ in adapter.tools.calls], ['robodojo_start', 'finish', 'close'])
        adapter = backend()
        adapter.start()
        adapter.close()
        self.assertNotIn('finish', [name for name, _ in adapter.tools.calls])

    def test_start_infer_and_finish_exceptions_are_single_use(self):
        adapter = backend()
        adapter.tools.fail = 'start'
        with self.assertRaises(HarnessError):
            adapter.start()
        with self.assertRaises(HarnessError):
            adapter.start()
        adapter.stop('startup_error')
        self.assertEqual([name for name, _ in adapter.tools.calls], ['robodojo_start', 'finish'])
        adapter = backend()
        observation = adapter.start()
        adapter.tools.fail = 'infer'
        with self.assertRaises(HarnessError):
            adapter.infer(observation)
        with self.assertRaises(HarnessError):
            adapter.infer(observation)
        adapter.tools.fail = 'finish'
        with self.assertRaises(HarnessError):
            adapter.stop('error')
        adapter.stop('error')
        self.assertEqual([name for name, _ in adapter.tools.calls].count('finish'), 1)

    def test_native_terminal_success_and_failure_do_not_step_again(self):
        outcomes = [
            dict(complete=True, terminated=True, truncated=False, success=True),
            dict(complete=True, terminated=False, truncated=True, success=False),
            dict(complete=True, success=False),
            dict(terminated=True, truncated=False, success=False),
        ]
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                adapter, _, proposal = ready()
                adapter.tools.outcome = outcome
                adapter.tools.actual_steps = 2
                after = adapter.execute(proposal, adapter.auto_response(proposal, 15, 'stage'), 'auto')
                self.assertTrue(after.terminal)
                self.assertIs(after.native_success, outcome['success'])
                self.assertEqual(after.step, 2)
                adapter.stop('already_done')
                with self.assertRaises(HarnessError):
                    adapter.infer(after)
                self.assertNotIn('finish', [name for name, _ in adapter.tools.calls])
                self.assertEqual(adapter.tools.physical_steps, 2)

    def test_incomplete_budget_and_model_stop_are_errors(self):
        for outcome in (dict(complete=False, terminated=False, truncated=False, reason='decision_budget'),
                        dict(complete=True, terminated=False, truncated=False, success=True),
                        dict(success=True, reason='operator_stop')):
            adapter, _, proposal = ready()
            adapter.tools.outcome = outcome
            with self.assertRaisesRegex(HarnessError, 'Incomplete'):
                adapter.execute(proposal, adapter.auto_response(proposal, 1, 'stage'), 'auto')
            adapter.stop('cleanup')
            self.assertNotIn('finish', [name for name, _ in adapter.tools.calls])
            with self.assertRaises(HarnessError):
                adapter.infer(proposal.observation)
        adapter, _, proposal = ready()
        response = adapter.auto_response(proposal, 1, 'stage')
        response['mode'] = 'stop'
        with self.assertRaisesRegex(HarnessError, 'Incomplete'):
            adapter.execute(proposal, response, 'reviewer')
        self.assertEqual(adapter.tools.physical_steps, 0)

    def test_rollout_finished_alone_is_not_native_terminal(self):
        adapter, _, proposal = ready()
        original = adapter.tools.execute

        def finish_without_result(**arguments):
            packet = original(**arguments)
            packet['rollout_finished'] = True
            return packet

        adapter.tools.execute = finish_without_result
        with self.assertRaisesRegex(HarnessError, 'without native terminal'):
            adapter.execute(proposal, adapter.auto_response(proposal, 1, 'stage'), 'auto')

    def test_terminal_history_is_native_evidence_without_result(self):
        adapter, _, proposal = ready()
        original = adapter.tools.execute

        def terminal_history(**arguments):
            packet = original(**arguments)
            packet['previous_result'].update(terminal=True, native_success=False)
            return packet

        adapter.tools.execute = terminal_history
        after = adapter.execute(proposal, adapter.auto_response(proposal, 1, 'stage'), 'auto')
        self.assertTrue(after.terminal)
        self.assertIs(after.native_success, False)
        adapter.stop('no_extra_finish')
        with self.assertRaises(HarnessError):
            adapter.infer(after)
        self.assertNotIn('finish', [name for name, _ in adapter.tools.calls])

    def test_native_truth_not_in_policy_payload(self):
        adapter, observation, proposal = ready()
        self.assertIsNone(observation.native_success)
        after = adapter.execute(proposal, adapter.auto_response(proposal, 1, 'stage'), 'auto')
        for payload in (observation.payload, proposal.payload, after.payload):
            for key in ('result', 'reward', 'object_truth', 'native_success', 'rollout_finished', 'next_call'):
                self.assertNotIn(key, payload)
            if payload['previous_result']:
                self.assertNotIn('native_success', payload['previous_result'])
                self.assertNotIn('reward', payload['previous_result'])
                self.assertNotIn('object_truth', payload['previous_result'])
        self.assertIsNone(after.native_success)
        self.assertTrue(adapter.metadata['uses_external_tools'])
        self.assertIn('engine JSONL', adapter.metadata['actual_source_record'])
        self.assertEqual(adapter.last_source, 'auto')
        self.assertEqual(adapter.tools.run, {'teacher': 'codex_tools', 'teacher_off': False})
        self.assertEqual(adapter.tools.history[-1]['source'], 'student')

    def test_budgets_use_actual_actions_remaining_and_cap(self):
        for remaining, count, expected in ((40, 50, 15), (3, 50, 3), (40, 2, 2), (1, 1, 1)):
            adapter, _, proposal = ready(FakeTools(remaining=remaining, count=count))
            self.assertEqual(proposal.max_steps, expected)
            with self.assertRaises(HarnessError):
                adapter.auto_response(proposal, expected + 1, 'stage')
            response = adapter.auto_response(proposal, expected, 'stage')
            response['steps'] = expected + 1
            with self.assertRaises(HarnessError):
                adapter.execute(proposal, response, 'reviewer')
        adapter = backend(FakeTools(remaining=0))
        observation = adapter.start()
        with self.assertRaisesRegex(HarnessError, 'remaining native steps'):
            adapter.infer(observation)

    def test_joint_diagnostics_and_first_action_are_distinct(self):
        tools = FakeTools(count=2)
        tools.diagnostic_changes['max_joint_step_rad'] = 0.6
        tools.actions[0][0] = tools.actions[1][0] = 0.8
        adapter, _, proposal = ready(tools)
        self.assertFalse(proposal.blocked)
        self.assertTrue(any('max_joint_step_rad=0.6' in value and 'successive' in value for value in proposal.warnings))
        self.assertTrue(any('first action joint jump=0.8' in value for value in proposal.warnings))
        _, _, relaxed = ready(tools=FakeTools(), joint_step_threshold_rad=1, first_joint_step_threshold_rad=1)
        self.assertEqual(relaxed.warnings, ())
        tools = FakeTools(count=2)
        tools.actions[1][7] = 0.4
        _, _, proposal = ready(tools)
        self.assertTrue(any('Actual successive joint jump=0.4' in value for value in proposal.warnings))
        self.assertFalse(any('first action' in value for value in proposal.warnings))

    def test_invalid_actions_and_diagnostics_block_execution(self):
        bad_actions = ([], [[0.0] * 13], [[math.nan] * 14], [[math.inf] * 14], [0.0, 1.0])
        for actions in bad_actions:
            tools = FakeTools()
            tools.actions = actions
            adapter, _, proposal = ready(tools)
            self.assertTrue(proposal.blocked)
            with self.assertRaises(HarnessError):
                adapter.auto_response(proposal, 1, 'stage')
            with self.assertRaises(HarnessError):
                adapter.execute(proposal, dict(request_id=proposal.request_id, mode='student', steps=1), 'auto')
            self.assertEqual(tools.validations, 0)
        for changes in ({'finite': False}, {'shape': [50, 13]}, {'max_joint_step_rad': math.inf},
                        {'opening_range': [[-0.1, 1], [0, 1]]},
                        {'opening_range': [[math.nan, 1], [0, 1]]},
                        {'opening_range': [[0, 1, 2], [0, 1]]}):
            tools = FakeTools()
            tools.diagnostic_changes = changes
            _, _, proposal = ready(tools)
            self.assertTrue(proposal.blocked)

    def test_numpy_actions_have_no_boolean_array_coercion(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy is unavailable')
        tools = FakeTools(count=3)
        tools.actions = np.asarray(tools.actions, dtype=np.float32)
        adapter, _, proposal = ready(tools)
        self.assertEqual(proposal.max_steps, 3)
        adapter.execute(proposal, adapter.auto_response(proposal, 3, 'stage'), 'auto')
        self.assertEqual(tools.physical_steps, 3)
        tools = FakeTools(count=3)
        tools.actions = np.zeros((3, 14, 1), dtype=np.float32)
        _, _, invalid = ready(tools)
        self.assertTrue(invalid.blocked)

    def test_empty_and_nonfinite_features_fail_closed(self):
        for features in ((), (math.nan,), (math.inf,)):
            adapter = RoboDojoBackend(FakeTools(), feature_encoder=lambda payload: features)
            with self.assertRaises(HarnessError):
                adapter.start()
            adapter.stop('bad_observation')
            self.assertEqual(adapter.tools.calls[-1][0], 'finish')

    def test_no_core_numpy_or_pillow_import(self):
        import builtins
        from fast_harness import robodojo
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'PIL'):
                raise AssertionError(f'Eager dependency: {name}')
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', guarded):
            importlib.reload(robodojo)
            adapter, _, proposal = ready()
            adapter.auto_response(proposal, 1, 'stage')

class FeatureTests(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is unavailable')
        self.Image = Image
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.payload = dict(current_state=[0.0] * 14, images=[])
        for camera in CAMERAS:
            path = Path(self.directory.name) / f'{camera}.png'
            Image.new('RGB', (16, 12), (0, 0, 0)).save(path)
            self.payload['images'].append(dict(camera=camera, path=str(path)))

    def test_shape_normalization_grippers_and_camera_order(self):
        state = self.payload['current_state']
        state[0], state[1], state[2] = -math.pi, math.pi, math.pi * 2
        state[6], state[13] = 0.25, 0.75
        self.Image.new('RGB', (16, 12), (255, 128, 0)).save(self.payload['images'][0]['path'])
        expected = encode_features(self.payload)
        self.assertEqual(len(expected), 3 * 8 * 8 * 3 + 14)
        self.assertEqual(expected[:3], (1, 128 / 255, 0))
        self.assertEqual(expected[-14:-11], (0, 1, 1))
        self.assertEqual(expected[-8], 0.25)
        self.assertEqual(expected[-1], 0.75)
        self.payload['images'].reverse()
        self.assertEqual(encode_features(self.payload), expected)
        self.assertTrue(all(0 <= value <= 1 for value in expected))

    def test_equal_camera_weight_and_appearance_change(self):
        baseline = encode_features(self.payload)
        distances = []
        for record in self.payload['images']:
            self.Image.new('RGB', (32, 24), (255, 255, 255)).save(record['path'])
            changed = encode_features(self.payload)
            distances.append(sum((left - right) ** 2 for left, right in zip(baseline, changed)))
            self.Image.new('RGB', (16, 12), (0, 0, 0)).save(record['path'])
        self.assertEqual(distances, [192.0] * 3)

    def test_missing_bad_images_and_nonfinite_state_are_errors(self):
        for state in ([0.0] * 13, [math.nan] * 14, [math.inf] * 14):
            with self.assertRaises(HarnessError):
                encode_features(dict(self.payload, current_state=state))
        with self.assertRaises(HarnessError):
            encode_features(dict(self.payload, images=self.payload['images'][:2]))
        self.payload['images'][0]['path'] = str(Path(self.directory.name) / 'missing.png')
        with self.assertRaises(HarnessError):
            encode_features(self.payload)

    def test_nonfinite_float_image_is_rejected_before_rgb_conversion(self):
        path = Path(self.directory.name) / 'nonfinite.tiff'
        self.Image.new('F', (8, 8), math.nan).save(path)
        self.payload['images'][0]['path'] = str(path)
        with self.assertRaises(HarnessError):
            encode_features(self.payload)


if __name__ == '__main__':
    unittest.main()

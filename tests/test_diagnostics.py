"""Rejected replies stay private, remain rejected, and never execute or retry."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch

from fast_harness import cli, reviewer as module
from fast_harness.engine import Harness
from fast_harness.memory import Memory
from fast_harness.scheduler import Config
from fast_harness.toy import ToyBackend, ToyReviewer
from fast_harness.types import HarnessError, Review, ReviewSchemaError, ReviewConstraintError, error_diagnostics
from test_reviewer import decision, request


PRIVATE = 'private-key-name-and-response-value'


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        for guard in (patch.dict(module.os.environ, {'TEST_KEY': 'test-secret'}, clear=True),
                      patch('socket.socket.connect', side_effect=AssertionError('No network'))):
            guard.start()
            self.addCleanup(guard.stop)

    def test_both_protocols_preserve_safe_nested_key_diagnostics_without_retry(self):
        req = request()
        for reviewer_class, suffix in ((module.ResponsesReviewer, 'responses'),
                                      (module.AnthropicReviewer, 'messages')):
            for path, missing in (((), 'stage'), (('response',), 'edit'),
                                  (('response', 'assessment'), 'intent_status'),
                                  (('checkpoint',), 'prerequisites')):
                with self.subTest(protocol=suffix, path=path):
                    output = decision(req)
                    output['checkpoint'] = {'goal': 'Approach', 'prerequisites': []}
                    target = output
                    for key in path:
                        target = target[key]
                    del target[missing]
                    target[PRIVATE] = PRIVATE
                    envelope = {'output_text': json.dumps(output),
                                'usage': {'input_tokens': 4, 'output_tokens': 2}}
                    if suffix == 'messages':
                        envelope = dict(type='message', role='assistant', stop_reason='tool_use',
                            usage=dict(input_tokens=4, output_tokens=2,
                                       cache_read_input_tokens=0, cache_creation_input_tokens=0),
                            content=[dict(type='tool_use', name='submit_review', input=output)])
                    transport = Mock(return_value=envelope)
                    reviewer = reviewer_class('https://example.invalid/v1/' + suffix,
                                               'test', 'TEST_KEY', transport=transport)
                    try:
                        reviewer.review(req)
                    except ReviewSchemaError as error:
                        diagnostic = error_diagnostics(error)
                        trace = traceback.format_exc()
                    else:
                        self.fail('Invalid reply was accepted')
                    self.assertEqual(diagnostic, {'schema_error': dict(
                        schema_path=list(path), missing_keys=[missing], extra_count=1)})
                    for private in (PRIVATE, 'test-secret'):
                        self.assertNotIn(private, trace + json.dumps(diagnostic))
                    transport.assert_called_once()
                    self.assertEqual(reviewer.calls, 1)
                    self.assertEqual(reviewer.usage_history[0].input_tokens, 4)

    def test_nullable_and_complete_objects_still_pass(self):
        req = request()
        output = decision(req)
        module.validate_review(Review(**output), req)
        output['checkpoint'] = {'goal': 'Approach', 'prerequisites': []}
        module.validate_review(Review(**output), req)

    def test_custom_reviewer_diagnostic_reaches_events_without_execution(self):
        class BrokenReviewer(ToyReviewer):
            def review(self, req):
                result = super().review(req)
                response = dict(result.response)
                del response['target']
                response[PRIVATE] = PRIVATE
                return replace(result, response=response)

        with tempfile.TemporaryDirectory() as root, contextlib.closing(Memory(':memory:', 'test')) as memory:
            backend = ToyBackend()
            output = Path(root) / 'run'
            with self.assertRaises(ReviewSchemaError):
                Harness(backend, BrokenReviewer(), memory, output).run()
            text = (output / 'events.jsonl').read_text()
            events = [json.loads(line) for line in text.splitlines()]
            failure = next(event for event in events if event['event'] == 'failure')
            self.assertEqual(failure['schema_error'], dict(
                schema_path=['response'], missing_keys=['target'], extra_count=1))
            self.assertEqual(backend.executions, 0)
            self.assertFalse(any(event['event'] == 'execution_requested' for event in events))
            self.assertNotIn(PRIVATE, text)

    def test_probe_writes_safe_diagnostic_and_points_to_probe_json(self):
        output = decision(request(0))
        del output['response']['target']
        output['response'][PRIVATE] = PRIVATE
        envelope = dict(type='message', role='assistant', stop_reason='tool_use',
                        content=[dict(type='tool_use', name='submit_review', input=output)])
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / 'probe'
            argv = ['fast-harness', 'probe', '--output', str(destination),
                    '--reviewer', 'anthropic', '--endpoint', 'https://example.invalid/v1/messages',
                    '--model', 'test', '--api-key-env', 'TEST_KEY', '--allow-model-requests']
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, 'argv', argv), patch.object(module, '_post', return_value=envelope) as transport, \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                cli.main()
            text = (destination / 'probe.json').read_text()
            summary = json.loads(text)
            self.assertEqual(summary['schema_error'], dict(
                schema_path=['response'], missing_keys=['target'], extra_count=1))
            self.assertEqual(summary['model_requests_attempted'], 1)
            self.assertEqual(summary['robot_runs'], 0)
            self.assertEqual(summary['status'], 'failed')
            self.assertIn('probe.json', stderr.getvalue())
            self.assertNotIn('events.jsonl', stderr.getvalue())
            self.assertNotIn(PRIVATE, text + stdout.getvalue() + stderr.getvalue())
            transport.assert_called_once()

    def test_generic_errors_never_expose_arbitrary_diagnostics(self):
        error = HarnessError(PRIVATE)
        error.schema_path = [PRIVATE]
        error.missing_keys = [PRIVATE]
        self.assertEqual(error_diagnostics(error), {})

    def test_flattened_action_is_rejected_without_repairing_or_retrying(self):
        req = request()
        output = decision(req)
        output.update(output.pop('response'))
        transport = Mock(return_value=output)
        reviewer = module.ResponsesReviewer('https://example.invalid/v1/responses',
                                            'test', 'TEST_KEY', transport=transport)
        with self.assertRaises(ReviewSchemaError) as caught:
            reviewer.review(req)
        self.assertEqual(error_diagnostics(caught.exception), {'schema_error': dict(
            schema_path=[], missing_keys=['response'], extra_count=7)})
        transport.assert_called_once()

    def test_three_chunk_budget_stops_before_fourth_inference_or_review(self):
        with tempfile.TemporaryDirectory() as root, contextlib.closing(Memory(':memory:', 'test')) as memory:
            backend = ToyBackend(chunks=10)
            reviewer = ToyReviewer()
            with patch.object(backend, 'infer', wraps=backend.infer) as infer, \
                 patch.object(reviewer, 'review', wraps=reviewer.review) as review:
                harness = Harness(backend, reviewer, memory, Path(root) / 'run',
                                  Config(always_review=True, max_episode_chunks=3))
                with self.assertRaises(HarnessError):
                    harness.run()
            self.assertEqual((infer.call_count, review.call_count, backend.executions), (3, 3, 3))
            self.assertFalse(harness.metrics['complete'])
            self.assertEqual(harness.metrics['control_steps'], 45)

    def test_safe_rule_codes_keep_invalid_reviews_rejected(self):
        req = request()
        variants = []
        output = decision(req)
        output['last_outcome'] = PRIVATE
        variants.append((output, 'schema_enum', ['last_outcome']))
        output = decision(req)
        output['response']['target']['left']['quaternion_wxyz'] = [2., 0., 0., 0.]
        variants.append((output, 'target_quaternion_norm', []))
        output = decision(req)
        output['response']['mode'] = 'eef'
        variants.append((output, 'takeover_gate', []))
        output = decision(req)
        output['last_outcome'] = 'unknown'
        output['recovery_status'] = 'succeeded'
        variants.append((output, 'recovery_outcome_conflict', []))
        output = decision(req)
        output['checkpoint'] = {'goal': 'Approach', 'prerequisites': PRIVATE}
        variants.append((output, 'schema_type', ['checkpoint', 'prerequisites']))
        for output, code, path in variants:
            with self.subTest(code=code):
                transport = Mock(return_value=output)
                reviewer = module.ResponsesReviewer('https://example.invalid/v1/responses',
                                                    'test', 'TEST_KEY', transport=transport)
                with self.assertRaises(ReviewConstraintError) as caught:
                    reviewer.review(req)
                diagnostic = error_diagnostics(caught.exception)
                self.assertEqual(diagnostic, {'validation_error': dict(code=code, schema_path=path)})
                self.assertNotIn(PRIVATE, json.dumps(diagnostic) + str(caught.exception))
                transport.assert_called_once()

    def test_rejected_envelope_code_never_copies_remote_message(self):
        for stop, expected in [('max_tokens', 'response_max_tokens'), (PRIVATE, 'response_envelope')]:
            with self.subTest(expected=expected):
                transport = Mock(return_value=dict(type='message', role='assistant',
                    stop_reason=stop, content=[dict(type='text', text=PRIVATE)],
                    usage=dict(input_tokens=3, output_tokens=2,
                               cache_read_input_tokens=0, cache_creation_input_tokens=0)))
                reviewer = module.AnthropicReviewer('https://example.invalid/v1/messages',
                                                   'test', 'TEST_KEY', transport=transport)
                with self.assertRaises(ReviewConstraintError) as caught:
                    reviewer.review(request())
                self.assertEqual(error_diagnostics(caught.exception), {'validation_error': dict(
                    code=expected, schema_path=[])})
                self.assertEqual(reviewer.usage_history[0].output_tokens, 2)
                transport.assert_called_once()


if __name__ == '__main__':
    unittest.main()

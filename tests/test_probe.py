import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from fast_harness import cli, reviewer as module
from fast_harness.types import HarnessError, ReviewTransportError
from test_reviewer import decision, request


class ProbeTests(unittest.TestCase):
    def arguments(self, output, protocol='anthropic', image=False):
        suffix = 'v1/messages' if protocol == 'anthropic' else 'responses'
        args = ['fast-harness', 'probe', '--output', str(output), '--reviewer', protocol,
                '--endpoint', f'https://example.invalid/{suffix}', '--model', 'test-model',
                '--api-key-env', 'TEST_PROBE_KEY', '--effort', 'max', '--allow-model-requests']
        return args + (['--with-image'] if image else [])

    def reply(self, protocol):
        output = decision(request(0))
        output['response'].update(request_id='synthetic-proposal', steps=1)
        if protocol == 'responses':
            return output
        return dict(type='message', role='assistant', stop_reason='tool_use',
                    content=[dict(type='tool_use', name='submit_review', input=output)],
                    usage=dict(input_tokens=20, cache_read_input_tokens=0,
                               cache_creation_input_tokens=0, output_tokens=10))

    def test_requires_opt_in_before_creating_output_or_client(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'probe'
            args = self.arguments(output)
            args.remove('--allow-model-requests')
            with patch.object(cli, '_reviewer') as create, patch.object(sys, 'argv', args), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main()
            create.assert_not_called()
            self.assertFalse(output.exists())

    def test_both_protocols_single_request_and_no_robot(self):
        for protocol in ('responses', 'anthropic'):
            for image in (False, True):
                with self.subTest(protocol=protocol, image=image), tempfile.TemporaryDirectory() as root:
                    output = Path(root) / 'probe'
                    transport = Mock(return_value=self.reply(protocol))
                    with patch.dict(module.os.environ, {'TEST_PROBE_KEY': 'test-only-secret'}, clear=True), \
                         patch.object(module, '_post', transport), \
                         patch.object(cli, 'Harness', side_effect=AssertionError('No robot loop')), \
                         patch.object(cli.ToyBackend, 'execute', side_effect=AssertionError('No toy action')), \
                         patch.object(sys, 'argv', self.arguments(output, protocol, image)), \
                         contextlib.redirect_stdout(io.StringIO()):
                        cli.main()
                    transport.assert_called_once()
                    summary = json.loads((output / 'probe.json').read_text())
                    self.assertEqual(summary['status'], 'passed')
                    self.assertEqual(summary['model_requests_attempted'], 1)
                    self.assertEqual(summary['robot_runs'], 0)
                    self.assertEqual(summary['last_outcome'], 'unknown')
                    body = json.loads(transport.call_args.args[2])
                    content = body['messages'][0]['content'] if protocol == 'anthropic' else body['input'][0]['content']
                    images = [part for part in content if part['type'] in ('image', 'input_image')]
                    self.assertEqual(len(images), int(image))
                    self.assertNotIn(root, json.dumps(body))
                    self.assertNotIn('test-only-secret', json.dumps(summary))
                    if image:
                        import struct
                        import zlib
                        png = (output / 'synthetic.png').read_bytes()
                        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
                        self.assertEqual(struct.unpack('!II', png[16:24]), (64, 64))
                        self.assertEqual(png[24:26], bytes((8, 2)))
                        size = struct.unpack('!I', png[33:37])[0]
                        pixels = zlib.decompress(png[41:41 + size])
                        self.assertEqual(pixels, (b'\0' + b'\x80\x80\x80' * 64) * 64)

    def test_existing_output_never_sends_request(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(module, '_post') as transport, \
                 patch.object(sys, 'argv', self.arguments(Path(root))), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main()
            transport.assert_not_called()

    def test_transport_failure_is_counted_without_retry_or_private_body(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'probe'
            transport = Mock(side_effect=TimeoutError('test-only-secret private response'))
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(module.os.environ, {'TEST_PROBE_KEY': 'test-only-secret'}, clear=True), \
                 patch.object(module, '_post', transport), patch.object(sys, 'argv', self.arguments(output)), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                cli.main()
            transport.assert_called_once()
            text = (output / 'probe.json').read_text()
            summary = json.loads(text)
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['model_requests_attempted'], 1)
            # A transport stall now surfaces as the specific ReviewTransportError (a HarnessError
            # subclass), which is what lets a caller recognise modelrouter instability.
            self.assertEqual(summary['error_type'], ReviewTransportError.__name__)
            self.assertTrue(issubclass(ReviewTransportError, HarnessError))
            self.assertEqual(len(summary['usage']), 1)
            for private in ('test-only-secret', 'private response'):
                self.assertNotIn(private, text + stdout.getvalue() + stderr.getvalue())

    def test_missing_auth_counts_zero_requests(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'probe'
            with patch.dict(module.os.environ, {}, clear=True), patch.object(module, '_post') as transport, \
                 patch.object(sys, 'argv', self.arguments(output)), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                cli.main()
            transport.assert_not_called()
            summary = json.loads((output / 'probe.json').read_text())
            self.assertEqual(summary['model_requests_attempted'], 0)
            self.assertEqual(summary['usage'], [])


if __name__ == '__main__':
    unittest.main()

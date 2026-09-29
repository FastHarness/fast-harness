import base64
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch
import urllib.error

from fast_harness import reviewer as module
from fast_harness.types import HarnessError, Review, Usage
from test_reviewer import decision, request


ENDPOINT = "https://example.invalid/v1/messages"
KEY_ENV = "TEST_ANTHROPIC_KEY"
SECRET = "test-only-anthropic-secret"
PRIVATE = "private-response-marker"
RAW_USAGE = {"input_tokens": 13, "cache_read_input_tokens": 5,
             "cache_creation_input_tokens": 2, "output_tokens": 7}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII=")


def message(req):
    return {"id": "msg-test", "type": "message", "role": "assistant",
            "stop_reason": "tool_use", "usage": dict(RAW_USAGE),
            "content": [{"type": "tool_use", "id": "toolu-test", "name": "submit_review",
                         "input": decision(req)}]}


class AnthropicReviewerTests(unittest.TestCase):
    def setUp(self):
        # Only synthetic environment credentials; accidental network access fails locally.
        for guard in (
            patch.dict(module.os.environ, {KEY_ENV: SECRET}, clear=True),
            patch("socket.create_connection", side_effect=AssertionError("Network access forbidden")),
            patch("socket.socket.connect", side_effect=AssertionError("Network access forbidden")),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.req = request()

    def client(self, transport=None, **kwargs):
        if transport is None:
            transport = Mock(return_value=message(self.req))
        return module.AnthropicReviewer(ENDPOINT, "test-model", KEY_ENV,
                                        transport=transport, **kwargs)

    def assert_rejected(self, reviewer, req=None):
        try:
            reviewer.review(self.req if req is None else req)
        except HarnessError:
            trace = traceback.format_exc()
        else:
            self.fail("Invalid review must raise HarnessError")
        for private in (SECRET, PRIVATE):
            self.assertNotIn(private, trace)

    def test_messages_request_shape_and_default_options(self):
        fake = Mock(return_value=message(self.req))
        reviewer = self.client(fake)
        self.assertEqual((reviewer.calls, reviewer.usage_history), (0, []))
        result = reviewer.review(self.req)
        self.assertEqual(result, Review(**decision(self.req), usage=Usage(20, 5, 7, 2)))
        self.assertEqual((reviewer.calls, reviewer.usage_history), (1, [result.usage]))
        fake.assert_called_once()
        endpoint, headers, raw, timeout = fake.call_args.args
        self.assertEqual((endpoint, timeout), (ENDPOINT, 120))
        headers = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(headers["x-api-key"], SECRET)
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertNotIn("authorization", headers)
        self.assertNotIn("anthropic-beta", headers)
        body = json.loads(raw)
        self.assertEqual(set(body), {"model", "max_tokens", "system", "messages", "tools", "tool_choice"})
        self.assertEqual((body["model"], body["max_tokens"]), ("test-model", 4096))
        self.assertIn(module.INSTRUCTIONS, body["system"])
        self.assertIn("submit_review", body["system"])
        self.assertEqual(body["messages"], [{"role": "user", "content": [
            {"type": "text", "text": module._content(self.req)[0]["text"]}]}])
        self.assertEqual(len(body["tools"]), 1)
        tool = body["tools"][0]
        self.assertEqual(tool["name"], "submit_review")
        self.assertEqual(tool["input_schema"], module.review_schema(self.req))
        self.assertNotIn("strict", tool)
        self.assertEqual(body["tool_choice"], {
            "type": "tool", "name": "submit_review", "disable_parallel_tool_use": True})
        self.assertNotIn(SECRET, raw.decode())

    def test_effort_and_max_tokens_are_forwarded_without_extra_features(self):
        for effort in (None, "low", "medium", "high", "max"):
            with self.subTest(effort=effort):
                fake = Mock(return_value=message(self.req))
                self.client(fake, effort=effort, max_tokens=17, timeout=2.5).review(self.req)
                body = json.loads(fake.call_args.args[2])
                self.assertEqual(body["max_tokens"], 17)
                self.assertEqual(fake.call_args.args[3], 2.5)
                if effort is None:
                    self.assertNotIn("output_config", body)
                else:
                    self.assertEqual(body["output_config"], {"effort": effort})
                for forbidden in ("store", "thinking", "stream", "reasoning", "betas", "strict"):
                    self.assertNotIn(forbidden, body)

    def test_initial_and_terminal_reviews_use_request_specific_schema(self):
        for req in (request(0), request(3, terminal=True)):
            with self.subTest(terminal=req.observation.terminal):
                fake = Mock(return_value=message(req))
                result = self.client(fake).review(req)
                self.assertEqual(result, Review(**decision(req), usage=Usage(20, 5, 7, 2)))
                schema = json.loads(fake.call_args.args[2])["tools"][0]["input_schema"]
                self.assertEqual(schema, module.review_schema(req))
                if req.observation.terminal:
                    self.assertIsNone(result.response)
                    self.assertEqual(schema["properties"]["response"], {"type": "null"})
                else:
                    self.assertEqual(result.last_outcome, "unknown")
                    self.assertEqual(result.response["assessment"]["execution_status"], "not_started")

    def test_text_blocks_are_allowed_but_never_parsed_as_review_json(self):
        envelope = message(self.req)
        envelope["content"] = [
            {"type": "text", "text": '{"stage":"wrong-stage"}'},
            *envelope["content"], {"type": "text", "text": "not JSON"}]
        fake = Mock(return_value=envelope)
        reviewer = self.client(fake)
        self.assertEqual(reviewer.review(self.req).stage, "approach")
        self.assertEqual(reviewer.calls, 1)
        fake.assert_called_once()

    def test_images_convert_shared_content_and_preserve_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.write_bytes(PNG)
            payload = dict(self.req.observation.payload,
                           images=[{"camera": "front /private/camera", "path": str(path)}],
                           api_key=SECRET, arrays_path="/private/actions.npz",
                           nested={"native_success": True, "reward": 1, "score": 9},
                           summary="Observed /private/frame.png", diagnostics={"measured": True})
            now = replace(self.req.observation, payload=payload)
            req = replace(self.req, observation=now, proposal=replace(self.req.proposal, observation=now))
            shared = module._content(req)
            fake = Mock(return_value=message(req))
            with patch.object(module, "_content", wraps=module._content) as content_builder:
                self.client(fake).review(req)
            content_builder.assert_called_once_with(req)
            content = json.loads(fake.call_args.args[2])["messages"][0]["content"]
            expected = []
            for part in shared:
                if part["type"] == "input_text":
                    expected.append({"type": "text", "text": part["text"]})
                else:
                    expected.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(PNG).decode("ascii")}})
            self.assertEqual(content, expected)
            self.assertEqual(sum(part["type"] == "image" for part in content), 1)
            raw = fake.call_args.args[2].decode()
            for private in (directory, "/private/", SECRET, "arrays_path", "api_key", "native_success", '"reward"', '"score"'):
                self.assertNotIn(private, raw)
            for public in ("current_state", "current_eef", "instruction", "diagnostics", "summary"):
                self.assertIn(public, content[0]["text"])

    def test_shared_image_limits_fail_before_transport_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.write_bytes(PNG)
            payload = dict(self.req.observation.payload,
                           images=[{"camera": "front", "path": str(path)}])
            now = replace(self.req.observation, payload=payload)
            req = replace(self.req, observation=now, proposal=replace(self.req.proposal, observation=now))
            for limit in ("MAX_IMAGE_BYTES", "MAX_TOTAL_IMAGE_BYTES"):
                with self.subTest(limit=limit), patch.object(module, limit, len(PNG) - 1):
                    fake = Mock()
                    reviewer = self.client(fake)
                    self.assert_rejected(reviewer, req)
                    fake.assert_not_called()
                    self.assertEqual((reviewer.calls, reviewer.usage_history), (0, []))

    def test_remote_image_is_rejected_without_network(self):
        now = replace(self.req.observation, payload={"images": [
            {"camera": "front", "path": "https://example.invalid/frame.png"}]})
        req = replace(self.req, observation=now, proposal=replace(self.req.proposal, observation=now))
        fake = Mock()
        reviewer = self.client(fake)
        self.assert_rejected(reviewer, req)
        fake.assert_not_called()
        self.assertEqual((reviewer.calls, reviewer.usage_history), (0, []))

    def test_usage_complete_partial_zero_and_malformed(self):
        cases = [
            (dict(RAW_USAGE), Usage(20, 5, 7, 2)),
            ({key: 0 for key in RAW_USAGE}, Usage(0, 0, 0, 0)),
            (None, Usage()), ({}, Usage()), ([], Usage()), ("invalid", Usage()),
            ({"input_tokens": 13, "output_tokens": 7}, Usage(None, None, 7, None)),
            ({"cache_read_input_tokens": 5, "cache_creation_input_tokens": 2}, Usage(None, 5, None, 2)),
            (dict(RAW_USAGE, input_tokens=None), Usage(None, 5, 7, 2)),
            (dict(RAW_USAGE, cache_read_input_tokens=None), Usage(None, None, 7, 2)),
            (dict(RAW_USAGE, cache_creation_input_tokens=None), Usage(None, 5, 7, None)),
        ]
        for missing, expected in (
            ("input_tokens", Usage(None, 5, 7, 2)),
            ("cache_read_input_tokens", Usage(None, None, 7, 2)),
            ("cache_creation_input_tokens", Usage(None, 5, 7, None)),
            ("output_tokens", Usage(20, 5, None, 2)),
        ):
            cases.append(({key: value for key, value in RAW_USAGE.items() if key != missing}, expected))
        for field, expected in (
            ("input_tokens", Usage(None, 5, 7, 2)),
            ("cache_read_input_tokens", Usage(None, None, 7, 2)),
            ("cache_creation_input_tokens", Usage(None, 5, 7, None)),
            ("output_tokens", Usage(20, 5, None, 2)),
        ):
            for invalid in (True, False, -1, 1.5, "2", [], {}, float("nan"), float("inf")):
                cases.append((dict(RAW_USAGE, **{field: invalid}), expected))
        for raw, expected in cases:
            with self.subTest(raw=raw):
                envelope = dict(message(self.req), usage=raw)
                fake = Mock(return_value=envelope)
                reviewer = self.client(fake)
                self.assertEqual(reviewer.review(self.req).usage, expected)
                self.assertEqual((reviewer.calls, reviewer.usage_history), (1, [expected]))
                fake.assert_called_once()
        envelope = message(self.req)
        del envelope["usage"]
        self.assertEqual(self.client(Mock(return_value=envelope)).review(self.req).usage, Usage())

    def test_rejected_envelopes_keep_usage_and_never_retry(self):
        good = message(self.req)
        tool = good["content"][0]
        variants = []
        for field, values in (
            ("type", ("error", "response", None)),
            ("role", ("user", "tool", None)),
            ("stop_reason", ("max_tokens", "refusal", "end_turn", "stop_sequence", "pause_turn", None)),
        ):
            for value in values:
                variants.append((f"{field}={value}", dict(good, **{field: value})))
            missing = deepcopy(good)
            del missing[field]
            variants.append((f"missing {field}", missing))
        variants.append(("error", dict(good, error={"message": f"{SECRET} {PRIVATE}"})))
        for name, content in (
            ("empty", []), ("missing content", None), ("not a list", {}),
            ("text JSON only", [{"type": "text", "text": json.dumps(decision(self.req))}]),
            ("two tools", [tool, deepcopy(tool)]),
            ("wrong name", [dict(tool, name="run_command")]),
            ("extra different tool", [tool, dict(tool, name="run_command")]),
            ("missing input", [{key: value for key, value in tool.items() if key != "input"}]),
            ("non-object block", [tool, "invalid"]),
            ("missing block type", [tool, {"text": PRIVATE}]),
            ("missing text", [tool, {"type": "text"}]),
            ("non-string text", [tool, {"type": "text", "text": None}]),
        ):
            variants.append((name, dict(good, content=content)))
        for value in (json.dumps(decision(self.req)), None, [], 0):
            variants.append((f"non-object input {type(value).__name__}", dict(good, content=[dict(tool, input=value)])))
        for kind in ("refusal", "thinking", "redacted_thinking", "server_tool_use", "tool_result", "image", "output_text", "unknown"):
            variants.append((kind, dict(good, content=[tool, {"type": kind, "text": f"{SECRET} {PRIVATE}"}])))
        variants.append(("direct decision is not a message", dict(decision(self.req), usage=dict(RAW_USAGE))))
        for name, envelope in variants:
            with self.subTest(case=name):
                fake = Mock(return_value=envelope)
                reviewer = self.client(fake)
                self.assert_rejected(reviewer)
                self.assertEqual((reviewer.calls, reviewer.usage_history), (1, [Usage(20, 5, 7, 2)]))
                fake.assert_called_once()

    def test_invalid_tool_schema_and_shared_safety_gates_keep_usage(self):
        cases = []
        for change in (
            {"extra": PRIVATE}, {"evidence": " "}, {"selected_checkpoint_id": "missing-checkpoint"},
        ):
            cases.append((self.req, dict(decision(self.req), **change)))
        stale = decision(self.req)
        stale["response"]["request_id"] = "stale-id"
        cases.append((self.req, stale))
        missing = decision(self.req)
        del missing["stage"]
        cases.append((self.req, missing))
        initial = request(0)
        cases.append((initial, dict(decision(initial), last_outcome="ok")))
        terminal = request(3, terminal=True)
        cases.append((terminal, dict(decision(terminal), response=decision(self.req)["response"])))
        for req, output in cases:
            with self.subTest(step=req.observation.step, terminal=req.observation.terminal, output=output):
                envelope = message(req)
                envelope["content"][0]["input"] = output
                fake = Mock(return_value=envelope)
                reviewer = self.client(fake)
                self.assert_rejected(reviewer, req)
                self.assertEqual((reviewer.calls, reviewer.usage_history), (1, [Usage(20, 5, 7, 2)]))
                fake.assert_called_once()

    def test_history_records_each_failed_and_successful_attempt_once(self):
        rejected = dict(message(self.req), stop_reason="max_tokens", usage={"cache_creation_input_tokens": 4})
        fake = Mock(side_effect=[rejected, message(self.req), TimeoutError(f"{SECRET} {PRIVATE}")])
        reviewer = self.client(fake)
        self.assert_rejected(reviewer)
        reviewer.review(self.req)
        self.assert_rejected(reviewer)
        self.assertEqual(reviewer.calls, 3)
        self.assertEqual(fake.call_count, 3)
        self.assertEqual(reviewer.usage_history, [Usage(None, None, None, 4), Usage(20, 5, 7, 2), Usage()])

    def test_endpoint_restrictions_and_loopback_transport_requirement(self):
        invalid = (
            "", "https://example.invalid/v1", "https://example.invalid/v1/responses",
            ENDPOINT + "/", "ftp://example.invalid/v1/messages", "http://example.invalid/v1/messages",
            "http://127.0.0.1.example.invalid/v1/messages", "http://localhost@remote.invalid/v1/messages",
            "https://key@example.invalid/v1/messages", ENDPOINT + "?api_key=secret", ENDPOINT + "#fragment",
            "https://example.invalid\\@remote.invalid/v1/messages", "https://example.invalid:bad/v1/messages",
            "https://example.invalid:0/v1/messages", "https://example.invalid:65536/v1/messages",
        )
        for endpoint in invalid:
            with self.subTest(endpoint=endpoint), self.assertRaises(HarnessError):
                module.AnthropicReviewer(endpoint, "test-model", KEY_ENV, transport=Mock())
        for host in ("localhost", "127.0.0.1", "[::1]"):
            endpoint = f"http://{host}:8123/v1/messages"
            with self.subTest(endpoint=endpoint):
                fake = Mock(return_value=message(self.req))
                reviewer = module.AnthropicReviewer(endpoint, "test-model", KEY_ENV, transport=fake)
                reviewer.review(self.req)
                self.assertEqual(fake.call_args.args[0], endpoint)
                with self.assertRaises(HarnessError):
                    module.AnthropicReviewer(endpoint, "test-model", KEY_ENV)

    def test_invalid_configuration_fails_before_transport(self):
        invalid = {
            "model": ("", " ", None), "api_key_env": ("", "bad-key", "1KEY", None),
            "timeout": (0, -1, True, "120", float("nan"), float("inf")),
            "transport": (False, "not-callable"),
            "effort": ("", "LOW", "minimal", "xhigh", " high ", 1, True, []),
            "max_tokens": (0, -1, True, False, 1.5, "4096", None),
        }
        for field, values in invalid.items():
            for value in values:
                fake = Mock()
                kwargs = dict(endpoint=ENDPOINT, model="test-model", api_key_env=KEY_ENV, transport=fake)
                kwargs[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(HarnessError):
                    module.AnthropicReviewer(**kwargs)
                fake.assert_not_called()

    def test_auth_uses_only_named_environment_without_opening_credential_files(self):
        fake = Mock(return_value=message(self.req))
        with patch("builtins.open", side_effect=AssertionError("Credential files forbidden")):
            self.client(fake).review(self.req)
        fake.assert_called_once()
        for value in (None, "", "bad\nheader", "bad\rheader", "bad key", "bad\tkey"):
            env = {} if value is None else {KEY_ENV: value}
            with self.subTest(value=value), patch.dict(module.os.environ, env, clear=True):
                fake = Mock()
                reviewer = self.client(fake)
                self.assert_rejected(reviewer)
                self.assertEqual((reviewer.calls, reviewer.usage_history), (0, []))
                fake.assert_not_called()

    def test_transport_errors_hide_credentials_and_body_without_retry(self):
        errors = (
            TimeoutError(f"{SECRET} {PRIVATE}"),
            urllib.error.URLError(f"{SECRET} {PRIVATE}"),
            urllib.error.HTTPError(ENDPOINT, 429, f"{SECRET} {PRIVATE}", {}, io.BytesIO(PRIVATE.encode())),
            RuntimeError(f"{SECRET} {PRIVATE}"),
        )
        for error in errors:
            with self.subTest(error_type=type(error).__name__):
                fake = Mock(side_effect=error)
                reviewer = self.client(fake)
                self.assert_rejected(reviewer)
                self.assertEqual((reviewer.calls, reviewer.usage_history), (1, [Usage()]))
                fake.assert_called_once()
        for envelope in (None, [], f"{SECRET} {PRIVATE}"):
            with self.subTest(envelope_type=type(envelope).__name__):
                fake = Mock(return_value=envelope)
                reviewer = self.client(fake)
                self.assert_rejected(reviewer)
                self.assertEqual((reviewer.calls, reviewer.usage_history), (1, [Usage()]))
                fake.assert_called_once()


if __name__ == "__main__":
    unittest.main()

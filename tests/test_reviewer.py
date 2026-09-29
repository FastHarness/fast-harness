import base64
from copy import deepcopy
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch
import urllib.error

from fast_harness import reviewer as module
from fast_harness.reviewer import ResponsesReviewer, review_schema, validate_review
from fast_harness.types import (Checkpoint, HarnessError, Observation, Proposal,
                                Review, ReviewRequest, Transition, Usage)


POSE = {"position": [0., 0., 0.], "quaternion_wxyz": [1., 0., 0., 0.]}


def request(step=3, terminal=False):
    payload = {"current_state": [0., 1.], "current_eef": {"left": deepcopy(POSE), "right": deepcopy(POSE)},
               "instruction": "Move the block to the tray", "images": []}
    now = Observation("episode", step, (0., 1.), payload, terminal, True)
    before = replace(now, step=0, terminal=False)
    previous = Transition(before, now, "prior-id", "approach", "student", "student", step) if step else None
    proposal = None if terminal else Proposal("current-id", now, {"fk": [[0., 1.]], "diagnostics": {"safe": True}})
    return ReviewRequest(now, proposal, previous, "approach", ("approach", "grasp"),
                         ({"error_type": "missed-grasp", "evidence": "Object did not lift"},), (), {}, ())


def decision(req):
    edit = {"delta_position": [0., 0., 0.], "delta_rotation_vector": [0., 0., 0.], "gripper": "keep"}
    target = dict(deepcopy(POSE), gripper_closed=False)
    assessment = {
        "task_progress": {"verified_completed": [], "currently_attempting": "Approach block", "remaining": ["Place block"]},
        "current_subgoal": "Approach block", "execution_status": "progressing" if req.previous else "not_started",
        "execution_evidence": "Arm moved closer" if req.previous else "No action has executed",
        "expected_next_intent": "Approach block", "predicted_next_intent": "Approach block",
        "intent_status": "aligned", "intent_evidence": "FK approaches the block"}
    return {"stage": "approach", "last_outcome": "ok" if req.previous else "unknown",
            "error_type": "none", "evidence": "The previous motion made progress" if req.previous else "No execution yet",
            "response": None if req.observation.terminal else {
                "request_id": req.proposal.request_id, "mode": "student", "steps": 3, "reason": "Continue approach",
                "edit": {"left": deepcopy(edit), "right": deepcopy(edit)},
                "target": {"left": deepcopy(target), "right": deepcopy(target)}, "assessment": assessment},
            "checkpoint": None, "recovery_status": "continue", "selected_checkpoint_id": None}


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(module.os.environ, {"TEST_REVIEWER_KEY": "test-only-secret"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.req = request()

    def client(self, transport=None, **kwargs):
        return ResponsesReviewer("https://example.invalid/v1/responses", "test-model", "TEST_REVIEWER_KEY",
                                 transport=transport or Mock(return_value=decision(self.req)), **kwargs)

    def test_direct_json_and_usage(self):
        output = decision(self.req)
        output["usage"] = {"input_tokens": 120, "input_tokens_details": {"cached_tokens": 80},
                           "output_tokens": 25, "output_tokens_details": {"reasoning_tokens": 10}}
        fake = Mock(return_value=output)
        reviewer = self.client(fake, effort="low")
        result = reviewer.review(self.req)
        self.assertEqual(result.usage, Usage(120, 80, 25))
        self.assertEqual(reviewer.usage_history, [result.usage])
        self.assertEqual(reviewer.calls, 1)
        url, headers, raw, timeout = fake.call_args.args
        self.assertEqual(url, "https://example.invalid/v1/responses")
        self.assertEqual(headers["Authorization"], "Bearer test-only-secret")
        self.assertEqual(timeout, 120)
        body = json.loads(raw)
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertFalse(body["store"])
        self.assertNotIn("test-only-secret", raw.decode())
        self.assertNotIn("previous_response_id", body)
        self.assertNotIn("conversation", body)
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])

    def test_responses_envelope_variants(self):
        text = json.dumps(decision(self.req))
        variants = [
            {"output_text": text},
            {"output_text": [text[:20], text[20:]]},
            {"output_text": [{"type": "output_text", "text": text}]},
            {"output": [{"type": "reasoning", "summary": []},
                        {"type": "message", "status": "completed", "content": [
                            {"type": "output_text", "text": text[:20]},
                            {"type": "output_text", "text": text[20:]}]}]},
        ]
        for envelope in variants:
            with self.subTest(envelope=envelope):
                envelope["status"] = "completed"
                reviewer = self.client(Mock(return_value=envelope))
                self.assertEqual(reviewer.review(self.req).stage, "approach")
                self.assertEqual(reviewer.usage_history, [Usage()])

    def test_missing_partial_zero_and_malformed_usage(self):
        for raw, expected in [(None, Usage()), ({}, Usage()), ({"input_tokens": 0}, Usage(0, None, None)),
                              ({"output_tokens": 3, "input_tokens_details": None}, Usage(None, None, 3)),
                              ({"input_tokens": True, "output_tokens": -1, "input_tokens_details": {"cached_tokens": "2"}}, Usage())]:
            with self.subTest(raw=raw):
                reviewer = self.client(Mock(return_value=dict(decision(self.req), usage=raw)))
                self.assertEqual(reviewer.review(self.req).usage, expected)

    def test_rejected_outputs_keep_usage_and_do_not_retry(self):
        good_text = json.dumps(decision(self.req))
        variants = [
            {"status": "incomplete", "output_text": good_text},
            {"status": "failed", "output_text": good_text},
            {"status": "in_progress", "output_text": good_text},
            {"status": None, "output_text": good_text},
            {"error": {"message": "test-only-secret private body"}, "output_text": good_text},
            {"incomplete_details": {"reason": "max_output_tokens"}, "output_text": good_text},
            {"output_text": good_text, "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "private body"}]}]},
            {"output": [{"type": "message", "status": "incomplete", "content": [{"type": "output_text", "text": good_text}]}]},
            {"output_text": "not JSON: private body"}, {"output_text": "[]"}, {"output": []},
            {"output_text": good_text.replace('"stage": "approach"', '"stage": "approach", "stage": "grasp"')},
            {"output_text": good_text.replace('0.0', 'NaN', 1)},
            {"output_text": good_text.replace('0.0', 'Infinity', 1)},
        ]
        for envelope in variants:
            with self.subTest(envelope=envelope):
                envelope["usage"] = {"input_tokens": 20, "output_tokens": 9}
                fake = Mock(return_value=envelope)
                reviewer = self.client(fake)
                with self.assertRaises(HarnessError) as caught:
                    reviewer.review(self.req)
                self.assertNotIn("private body", str(caught.exception))
                self.assertNotIn("test-only-secret", str(caught.exception))
                self.assertEqual(reviewer.usage_history, [Usage(20, None, 9)])
                self.assertEqual(reviewer.calls, 1)
                fake.assert_called_once()

    def test_schema_rejects_malicious_fields(self):
        changes = [
            ((), "stage", "Bad Stage"), ((), "last_outcome", "success"), ((), "evidence", " "),
            ((), "recovery_status", "done"), ((), "selected_checkpoint_id", "missing"),
            (("response",), "request_id", "stale-id"), (("response",), "mode", "teleport"),
            (("response",), "steps", True), (("response",), "steps", 1.5), (("response",), "steps", 16),
            (("response",), "steps", 0), (("response",), "extra", "wrong"),
            (("response", "target", "left"), "position", [math.nan, 0., 0.]),
            (("response", "target", "right"), "position", [math.inf, 0., 0.]),
            (("response", "target", "right"), "position", [True, 0., 0.]),
            (("response", "target", "left"), "quaternion_wxyz", [0., 0., 0., 0.]),
            (("response", "target", "left"), "gripper_closed", 1),
            (("response", "edit", "left"), "delta_position", [0., 0.]),
            (("response", "assessment"), "intent_status", "safe"),
            (("response", "assessment", "task_progress"), "remaining", "Place block"),
        ]
        for path, key, value in changes:
            with self.subTest(path=path, key=key, value=value):
                output = decision(self.req)
                node = output
                for name in path:
                    node = node[name]
                node[key] = value
                reviewer = self.client(Mock(return_value=output))
                with self.assertRaises(HarnessError):
                    reviewer.review(self.req)
                self.assertEqual(reviewer.calls, 1)
                self.assertEqual(len(reviewer.usage_history), 1)
        for key in decision(self.req):
            output = decision(self.req)
            del output[key]
            with self.subTest(missing=key), self.assertRaises(HarnessError):
                self.client(Mock(return_value=output)).review(self.req)
        output = decision(self.req)
        del output["response"]["target"]["right"]
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), self.req)

    def test_initial_outcome_gate_and_horizon(self):
        initial = request(0)
        validate_review(Review(**decision(initial)), initial)
        bad = dict(decision(initial), last_outcome="ok")
        with self.assertRaises(HarnessError):
            validate_review(Review(**bad), initial)
        output = decision(self.req)
        output["response"]["assessment"]["execution_status"] = "not_started"
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), self.req)
        for mode in ("edit", "eef"):
            output = decision(self.req)
            output["response"]["mode"] = mode
            with self.assertRaises(HarnessError):
                validate_review(Review(**output), self.req)
            output["response"]["assessment"]["intent_status"] = "misaligned"
            validate_review(Review(**output), self.req)
            output["response"]["steps"] = 6
            with self.assertRaises(HarnessError):
                validate_review(Review(**output), self.req)
            output["response"]["steps"] = 3
            output["response"]["assessment"].update(execution_status="failed", intent_status="aligned")
            with self.assertRaises(HarnessError):
                validate_review(Review(**output), self.req)
            output["last_outcome"] = "error"
            validate_review(Review(**output), self.req)
        short = replace(self.req, proposal=replace(self.req.proposal, max_steps=2))
        with self.assertRaises(HarnessError):
            validate_review(Review(**decision(short)), short)
        blocked = replace(self.req, proposal=replace(self.req.proposal, blocked=True))
        with self.assertRaises(HarnessError):
            validate_review(Review(**decision(blocked)), blocked)

    def test_motion_bounds_and_stale_observation(self):
        for mode, field, key, value in [
            ("eef", "target", "position", [.051, 0., 0.]),
            ("eef", "target", "quaternion_wxyz", [math.cos(.2), math.sin(.2), 0., 0.]),
            ("edit", "edit", "delta_position", [.1, 0., 0.]),
            ("edit", "edit", "delta_rotation_vector", [.36, 0., 0.])]:
            output = decision(self.req)
            output["response"]["mode"] = mode
            output["response"]["assessment"]["intent_status"] = "misaligned"
            output["response"][field]["right"][key] = value
            with self.subTest(mode=mode, key=key), self.assertRaises(HarnessError):
                validate_review(Review(**output), self.req)
        stale = replace(self.req, proposal=replace(self.req.proposal, observation=replace(self.req.observation, step=2)))
        with self.assertRaises(HarnessError):
            validate_review(Review(**decision(stale)), stale)

    def test_terminal_checkpoint_and_recovery_rules(self):
        terminal = request(3, terminal=True)
        output = decision(terminal)
        output["checkpoint"] = {"goal": "Arm near block", "prerequisites": ["Block visible"]}
        validate_review(Review(**output), terminal)
        output["response"] = decision(self.req)["response"]
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), terminal)
        output["response"] = None
        output["last_outcome"] = "unknown"
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), terminal)
        point = Checkpoint("cp-1", self.req.observation, "approach", "Arm near block", ())
        recovering = replace(self.req, checkpoints=(point,), recovery={"state": "return", "goal": "Reach approach", "eligible_checkpoint_ids": []})
        output = decision(recovering)
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), recovering)
        output["recovery_status"] = "failed"
        validate_review(Review(**output), recovering)
        output["selected_checkpoint_id"] = "cp-1"
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), recovering)
        recovering = replace(recovering, recovery={"state": "return", "goal": "Reach approach", "eligible_checkpoint_ids": ["cp-1"]})
        output["recovery_status"] = "succeeded"
        validate_review(Review(**output), recovering)

    def test_images_privacy_history_and_selective_checkpoint_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            raw = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII=")
            path.write_bytes(raw)
            payload = dict(self.req.observation.payload, images=[{"camera": c, "path": str(path)} for c in ("front", "left", "right")],
                           arrays_path="/private/actions.npz", nested={"reward": 1, "native_success": True, "score": 9},
                           summary="Moved from /private/before.png to /private/after.png", diagnostics={"measured": True})
            now = replace(self.req.observation, payload=payload)
            previous = replace(self.req.previous, before=replace(self.req.previous.before, payload=payload), after=now)
            points = tuple(Checkpoint(f"cp-{i}", now, "approach", "Reach approach", ("Block visible",)) for i in range(5))
            req = replace(self.req, observation=now, proposal=replace(self.req.proposal, observation=now),
                          previous=previous, unreviewed=(previous,), checkpoints=points,
                          recovery={"phase": "return_latest", "goal": "Reach approach", "target_checkpoint_id": "cp-2"})
            fake = Mock(return_value=decision(req))
            self.client(fake).review(req)
            body = json.loads(fake.call_args.args[2])
            content = body["input"][0]["content"]
            self.assertEqual(sum(part["type"] == "input_image" for part in content), 9)
            self.assertTrue(all(part["image_url"].startswith("data:image/png;base64,") for part in content if part["type"] == "input_image"))
            text = content[0]["text"]
            for private in (directory, "/private/", "arrays_path", "native_success", '"reward"', '"features"', '"score"'):
                self.assertNotIn(private, text)
            for public in ("current_state", "current_eef", "fk", "diagnostics", "instruction", "summary", "unreviewed", "before", "after", "known_stages", "errors", "recovery", "goal"):
                self.assertIn(public, text)
            with patch.object(module, "MAX_IMAGE_BYTES", 4):
                reviewer = self.client(fake)
                with self.assertRaises(HarnessError):
                    reviewer.review(req)
                self.assertEqual(reviewer.calls, 0)
            with patch.object(module, "MAX_TOTAL_IMAGE_BYTES", len(raw) * 2):
                with self.assertRaises(HarnessError):
                    self.client(fake).review(req)

    def test_invalid_images_fail_before_call(self):
        for path in ("https://example.invalid/frame.png", "relative.png", "/nonexistent/frame.png"):
            req = replace(self.req, observation=replace(self.req.observation, payload={"images": [{"camera": "front", "path": path}]}))
            fake = Mock()
            with self.subTest(path=path), self.assertRaises(HarnessError):
                self.client(fake).review(req)
            fake.assert_not_called()

    def test_endpoint_and_auth_validation(self):
        for endpoint in ("", "https://example.invalid/v1", "https://example.invalid/v1/responses/", "ftp://example.invalid/responses",
                         "http://example.invalid/responses", "http://127.0.0.1.example.invalid/responses",
                         "http://localhost@remote.invalid/responses", "https://key@example.invalid/responses",
                         "https://example.invalid/responses?api_key=secret", "https://example.invalid/responses#fragment",
                         "https://example.invalid\\@remote.invalid/responses", "https://example.invalid:bad/responses"):
            with self.subTest(endpoint=endpoint), self.assertRaises(HarnessError):
                ResponsesReviewer(endpoint, "test", "TEST_REVIEWER_KEY", transport=Mock())
        for host in ("localhost", "127.0.0.1", "[::1]"):
            endpoint = f"http://{host}:8123/responses"
            ResponsesReviewer(endpoint, "test", "TEST_REVIEWER_KEY", transport=Mock())
            with self.assertRaises(HarnessError):
                ResponsesReviewer(endpoint, "test", "TEST_REVIEWER_KEY")
        with patch("builtins.open", side_effect=AssertionError("No credential files")):
            self.client().review(self.req)
        for value in ("", "bad\nheader"):
            with patch.dict(module.os.environ, {"TEST_REVIEWER_KEY": value}):
                reviewer = self.client()
                with self.assertRaises(HarnessError):
                    reviewer.review(self.req)
                self.assertEqual((reviewer.calls, reviewer.usage_history), (0, []))

    def test_transport_errors_and_default_urllib_policy(self):
        error = urllib.error.HTTPError("https://example.invalid/responses", 401, "test-only-secret private body", {}, io.BytesIO(b"private body"))
        reviewer = self.client(Mock(side_effect=error))
        try:
            reviewer.review(self.req)
        except HarnessError:
            trace = traceback.format_exc()
        self.assertNotIn("test-only-secret", trace)
        self.assertNotIn("private body", trace)
        self.assertEqual((reviewer.calls, reviewer.usage_history), (1, [Usage()]))
        reply = Mock(status=200)
        reply.read.side_effect = io.BytesIO(json.dumps(decision(self.req)).encode()).read
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=reply)
        opener.open.return_value.__exit__ = Mock(return_value=False)
        with patch.object(module.urllib.request, "build_opener", return_value=opener) as build:
            reviewer = ResponsesReviewer("https://example.invalid/responses", "test", "TEST_REVIEWER_KEY")
            reviewer.review(self.req)
            handlers = build.call_args.args
            self.assertEqual(handlers[0].proxies, {})
            self.assertIsNone(handlers[1].redirect_request(None, None, 302, None, None, "https://remote.invalid/responses"))
            self.assertEqual(opener.open.call_args.args[0].get_method(), "POST")
            opener.open.assert_called_once()

    def test_recovery_phase_contract_and_no_false_local_failure(self):
        req = replace(self.req, recovery={"phase": "local"})
        validate_review(Review(**decision(req)), req)
        req = replace(req, recovery={"phase": "select"})
        with self.assertRaises(HarnessError):
            validate_review(Review(**decision(req)), req)
        output = dict(decision(req), recovery_status="failed")
        validate_review(Review(**output), req)
        point = Checkpoint("spent", req.observation, "approach", "Reach approach", ())
        req = replace(req, checkpoints=(point,), recovery={"phase": "select", "target_checkpoint_id": "spent"})
        output["selected_checkpoint_id"] = "spent"
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), req)
        output = decision(self.req)
        output["response"]["assessment"]["execution_status"] = "failed"
        output.update(last_outcome="error", recovery_status="succeeded")
        with self.assertRaises(HarnessError):
            validate_review(Review(**output), self.req)
        with self.assertRaises(HarnessError):
            validate_review(Review(**decision(self.req), usage={}), self.req)

    def test_unused_checkpoint_images_are_never_read(self):
        observation = replace(self.req.observation, payload={"images": [{"camera": "front", "path": "/not/read.png"}]})
        point = Checkpoint("unused", observation, "approach", "Reach approach", ())
        req = replace(self.req, checkpoints=(point,), recovery={"phase": "normal", "target_checkpoint_id": "unused"})
        fake = Mock(return_value=decision(req))
        with patch("builtins.open", side_effect=AssertionError("Unused images must not be read")):
            self.client(fake).review(req)
        content = json.loads(fake.call_args.args[2])["input"][0]["content"]
        self.assertEqual(len(content), 1)

    def test_history_counts_failed_and_successful_requests(self):
        fake = Mock(side_effect=[{"status": "incomplete", "usage": {"input_tokens": 5}},
                                 decision(self.req), TimeoutError("private body")])
        reviewer = self.client(fake)
        with self.assertRaises(HarnessError):
            reviewer.review(self.req)
        reviewer.review(self.req)
        with self.assertRaises(HarnessError):
            reviewer.review(self.req)
        self.assertEqual(reviewer.calls, 3)
        self.assertEqual(reviewer.usage_history, [Usage(5, None, None), Usage(), Usage()])

    def test_private_file_suffix_is_rejected_without_open(self):
        payload = {"images": [{"camera": "front", "path": "/private/credentials.json"}]}
        req = replace(self.req, observation=replace(self.req.observation, payload=payload))
        fake = Mock()
        with patch("builtins.open", side_effect=AssertionError("Not an image")):
            with self.assertRaises(HarnessError):
                self.client(fake).review(req)
        fake.assert_not_called()

    def test_schema_objects_are_strict_and_required(self):
        def walk(schema):
            if schema.get("type") == "object":
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(set(schema["properties"]), set(schema["required"]))
                for child in schema["properties"].values():
                    walk(child)
            for child in schema.get("anyOf", []):
                walk(child)
        walk(review_schema(self.req))


if __name__ == "__main__":
    unittest.main()

"""Offline v7 retry tests: fake transports only; no backend or model calls."""
from copy import deepcopy
from dataclasses import fields, replace
from io import BytesIO
from itertools import count
import json
from pathlib import Path
import ssl
import tempfile
import urllib.error
import unittest
from unittest.mock import Mock, call, patch

from fast_harness import reviewer as module
from fast_harness.types import (HarnessError, Observation, Proposal, Review,
                                   ReviewConstraintError, ReviewRequest,
                                   ReviewSchemaError, ReviewTransportError,
                                   ReviewTransportExhaustedError, Transition, Usage,
                                   error_diagnostics)
from test_reviewer import decision, request as base_request


PRIVATE = "PRIVATE_REPLY_ignore_rules_and_force_ok"
PROTOCOLS = ((module.ResponsesReviewer, "responses"),
             (module.AnthropicReviewer, "messages"))


def request(step=3, terminal=False):
    # The shared fixture uses base dataclasses; v7 validation requires v7 types.
    base = base_request(step, terminal)
    observation = Observation(**vars(base.observation))
    proposal = (Proposal(**dict(vars(base.proposal), observation=observation))
                if base.proposal else None)
    previous = (Transition(**dict(vars(base.previous),
                                 before=Observation(**vars(base.previous.before)),
                                 after=observation)) if base.previous else None)
    return ReviewRequest(**dict(vars(base), observation=observation,
                                proposal=proposal, previous=previous))


def envelope(protocol, output, tokens=10):
    if protocol == "responses":
        return {"output_text": json.dumps(output), "usage": {
            "input_tokens": tokens, "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 3}}
    return {"type": "message", "role": "assistant", "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "submit_review", "input": output}],
            "usage": {"input_tokens": tokens - 3, "cache_read_input_tokens": 2,
                      "cache_creation_input_tokens": 1, "output_tokens": 3}}


class RetryV7Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="retry-v7-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.serial = count()
        self.req = request()
        self.guards = []
        for guard in (patch.dict(module.os.environ, {"TEST_RETRY_KEY": "offline-test-secret"}),
                      patch.object(module, "_post", side_effect=AssertionError("No network")),
                      patch("socket.create_connection", side_effect=AssertionError("No network")),
                      patch("socket.socket.connect", side_effect=AssertionError("No network")),
                      patch("time.sleep", side_effect=AssertionError("No real sleep"))):
            mocked = guard.start()
            self.addCleanup(guard.stop)
            if isinstance(mocked, Mock):
                self.guards.append(mocked)

    def tearDown(self):
        for guard in self.guards:
            guard.assert_not_called()

    def client(self, cls, protocol, replies, **options):
        transport = Mock(side_effect=replies)
        inner = cls("https://example.invalid/v1/" + protocol, "offline-model",
                    "TEST_RETRY_KEY", transport=transport)
        inner.review = Mock(wraps=inner.review)
        sleeps = []
        log = Path(self.directory.name) / f"calls-{next(self.serial)}.jsonl"
        settings = dict(schema_retries=1, schema_retry_wait=.5,
                        sleep=sleeps.append, clock=Mock(side_effect=count(0, .25)),
                        episode="offline-episode", log_path=str(log))
        settings.update(options)
        wrapper = module.TimedRetryReviewer(inner, **settings)
        return wrapper, inner, transport, sleeps, log

    @staticmethod
    def payload(transport, index):
        body = json.loads(transport.call_args_list[index].args[2])
        content = body["input" if "input" in body else "messages"][0]["content"]
        return body, content, json.loads(content[0]["text"])

    @staticmethod
    def records(log):
        return [json.loads(line) for line in log.read_text().splitlines()]

    def assert_private_absent(self, transport, log):
        text = log.read_text() + "".join(call.args[2].decode() for call in transport.call_args_list)
        for secret in (PRIVATE, "offline-test-secret"):
            self.assertNotIn(secret, text)

    def assert_only_feedback_changed(self, transport, first, later):
        before, before_content, before_data = self.payload(transport, first)
        after, after_content, after_data = self.payload(transport, later)
        before_data.pop("retry_feedback")
        feedback = after_data.pop("retry_feedback")
        before_content[0]["text"] = before_data
        after_content[0]["text"] = after_data
        self.assertEqual(before, after)
        return feedback

    def test_missing_fields_feedback_reaches_both_payloads_and_logs_usage(self):
        for cls, protocol in PROTOCOLS:
            for location, missing in (((), "response"), (("response",), "target"),
                                      (("response", "assessment"), "intent_status"),
                                      (("checkpoint",), "prerequisites")):
                with self.subTest(protocol=protocol, location=location):
                    bad = decision(self.req)
                    bad["checkpoint"] = {"goal": "Approach", "prerequisites": []}
                    node = bad
                    for name in location:
                        node = node[name]
                    del node[missing]
                    node[PRIVATE] = PRIVATE
                    bad["evidence"] = PRIVATE
                    snapshot = deepcopy(bad)
                    wrapper, inner, transport, sleeps, log = self.client(cls, protocol, [
                        envelope(protocol, bad), envelope(protocol, decision(self.req), 20)])
                    result = wrapper.review(self.req)
                    self.assertIsInstance(result, Review)
                    self.assertIsNone(self.payload(transport, 0)[2]["retry_feedback"])
                    feedback = self.assert_only_feedback_changed(transport, 0, 1)
                    self.assertEqual(feedback["code"], "schema_object_fields")
                    self.assertEqual(feedback["location"], list(location))
                    self.assertEqual(feedback["missing_keys"], [missing])
                    self.assertEqual(set(feedback), {"code", "location", "missing_keys", "constraint"})
                    self.assertEqual(bad, snapshot)
                    created = 1 if protocol == "messages" else None
                    self.assertEqual(inner.usage_history, [Usage(10, 2, 3, created), Usage(20, 2, 3, created)])
                    self.assertEqual(wrapper.usage_history, [result.usage])
                    self.assertEqual((inner.calls, wrapper.calls, wrapper.schema_rejects), (2, 1, 1))
                    self.assertEqual(sleeps, [.5])
                    records = self.records(log)
                    self.assertEqual([r["outcome"] for r in records], ["schema_reject", "ok"])
                    self.assertEqual(records[0]["schema_error"], {
                        "schema_path": list(location), "missing_keys": [missing], "extra_count": 1})
                    self.assertEqual((records[0]["input_tokens"], records[0]["output_tokens"]), (10, 3))
                    self.assertEqual((records[1]["input_tokens"], records[1]["cached_input_tokens"],
                                      records[1]["cache_creation_input_tokens"]), (20, 2, created))
                    self.assertEqual([r["attempt"] for r in records], [1, 2])
                    self.assertTrue(all(r["call_index"] == 0 and r["episode"] == "offline-episode"
                                        and r["latency_seconds"] == .25 for r in records))
                    self.assert_private_absent(transport, log)

    def test_recovery_feedback_does_not_promote_unknown_or_error_to_ok(self):
        for cls, protocol in PROTOCOLS:
            for outcome, execution in (("unknown", "uncertain"), ("error", "failed")):
                with self.subTest(protocol=protocol, outcome=outcome):
                    req = replace(self.req, recovery={"phase": "local", "goal": "Approach"})
                    bad = decision(req)
                    bad.update(last_outcome=outcome, recovery_status="succeeded")
                    bad["response"]["assessment"]["execution_status"] = execution
                    good = dict(deepcopy(bad), recovery_status="continue")
                    wrapper, _, transport, _, log = self.client(cls, protocol, [
                        envelope(protocol, bad), envelope(protocol, good)])
                    result = wrapper.review(req)
                    self.assertEqual((result.last_outcome, result.recovery_status), (outcome, "continue"))
                    feedback = self.payload(transport, 1)[2]["retry_feedback"]
                    self.assertEqual(feedback["code"], "recovery_outcome_conflict")
                    self.assertIn("requires last_outcome=ok AND visible achievement", feedback["constraint"])
                    self.assertIn("Do not change error/unknown to ok", feedback["constraint"])
                    self.assertIn("original observations and execution evidence", feedback["constraint"])
                    self.assertIn("last_outcome=unknown and recovery_status=continue", feedback["constraint"])
                    self.assert_private_absent(transport, log)

    def test_execution_consistency_feedback_keeps_evidence_uncertainty(self):
        for cls, protocol in PROTOCOLS:
            for outcome, execution, fixed_outcome, fixed_execution in (
                    ("unknown", "recovered", "unknown", "uncertain"),
                    ("ok", "uncertain", "unknown", "uncertain"),
                    ("error", "progressing", "error", "failed"),
                    ("ok", "failed", "unknown", "uncertain"),
                    ("unknown", "not_started", "unknown", "uncertain")):
                with self.subTest(protocol=protocol, outcome=outcome, execution=execution):
                    bad = decision(self.req)
                    bad["last_outcome"] = outcome
                    bad["response"]["assessment"]["execution_status"] = execution
                    good = deepcopy(bad)
                    good["last_outcome"] = fixed_outcome
                    good["response"]["assessment"]["execution_status"] = fixed_execution
                    wrapper, _, transport, _, _ = self.client(cls, protocol, [
                        envelope(protocol, bad), envelope(protocol, good)])
                    result = wrapper.review(self.req)
                    self.assertEqual(result.last_outcome, fixed_outcome)
                    feedback = self.payload(transport, 1)[2]["retry_feedback"]
                    expected = "execution_start_conflict" if execution == "not_started" else "execution_outcome_conflict"
                    self.assertEqual(feedback["code"], expected)

    def test_repeated_schema_failure_is_bounded_including_default_off(self):
        for cls, protocol in PROTOCOLS:
            for budget in (0, 2):
                with self.subTest(protocol=protocol, budget=budget):
                    bad = decision(self.req)
                    del bad["response"]["target"]
                    wrapper, inner, transport, sleeps, log = self.client(cls, protocol,
                        [envelope(protocol, bad)] * 4, schema_retries=budget)
                    with self.assertRaises(ReviewSchemaError) as caught:
                        wrapper.review(self.req)
                    self.assertNotIn("failure_reason", error_diagnostics(caught.exception))
                    self.assertNotIn("transport_error", error_diagnostics(caught.exception))
                    for record in self.records(log):
                        self.assertNotIn("failure_reason", record)
                        self.assertNotIn("transport_error", record)
                    self.assertEqual((inner.calls, transport.call_count, wrapper.schema_rejects),
                                     (budget + 1,) * 3)
                    self.assertEqual((wrapper.calls, wrapper.usage_history), (0, []))
                    self.assertEqual(len(inner.usage_history), budget + 1)
                    self.assertEqual(sleeps, [.5] * budget)
                    self.assertEqual([r["will_retry"] for r in self.records(log)], [True] * budget + [False])
                    self.assertIsNone(self.req.retry_feedback)

    def test_bad_actions_remain_rejected_after_retry(self):
        variants = []
        for mode in ("edit", "eef"):
            bad = decision(self.req)
            bad["response"]["mode"] = mode
            variants.append((self.req, deepcopy(bad), "takeover_gate"))
            bad["response"]["assessment"]["intent_status"] = "misaligned"
            field, key = ("edit", "delta_position") if mode == "edit" else ("target", "position")
            bad["response"][field]["left"][key] = [.051, 0., 0.]
            variants.append((self.req, bad, mode + "_range"))
        for field, value, code in (("steps", 16, "schema_integer_range"),
                                    ("request_id", PRIVATE, "schema_enum")):
            bad = decision(self.req)
            bad["response"][field] = value
            variants.append((self.req, bad, code))
        short = replace(self.req, proposal=replace(self.req.proposal, max_steps=2))
        variants.append((short, decision(short), "action_step_limit"))
        blocked = replace(self.req, proposal=replace(self.req.proposal, blocked=True))
        variants.append((blocked, decision(blocked), "blocked_proposal"))
        bad = decision(self.req)
        bad["response"]["target"]["right"]["quaternion_wxyz"] = [2., 0., 0., 0.]
        variants.append((self.req, bad, "target_quaternion_norm"))
        initial = request(0)
        variants.append((initial, dict(decision(initial), last_outcome="ok"), "outcome_without_execution"))
        terminal = request(terminal=True)
        variants.append((terminal, decision(self.req), "schema_type"))
        for cls, protocol in PROTOCOLS:
            for req, bad, code in variants:
                with self.subTest(protocol=protocol, code=code):
                    wrapper, inner, transport, _, log = self.client(cls, protocol, [
                        envelope(protocol, bad), envelope(protocol, bad), envelope(protocol, decision(req))])
                    with self.assertRaises(ReviewConstraintError) as caught:
                        wrapper.review(req)
                    self.assertEqual(caught.exception.code, code)
                    self.assertEqual((inner.calls, wrapper.calls, wrapper.schema_rejects), (2, 0, 2))
                    self.assertEqual(wrapper.usage_history, [])
                    self.assertFalse(self.records(log)[-1]["will_retry"])
                    self.assertEqual(self.payload(transport, 1)[2]["retry_feedback"]["code"], code)
                    self.assert_private_absent(transport, log)

    def test_feedback_is_replaced_not_accumulated_and_original_request_is_unchanged(self):
        self.assertIsInstance(self.req, ReviewRequest)
        self.assertIsInstance(self.req.observation, Observation)
        self.assertIsInstance(self.req.proposal, Proposal)
        self.assertIsInstance(self.req.previous, Transition)
        snapshot = deepcopy(self.req)
        for cls, protocol in PROTOCOLS:
            bad = decision(self.req)
            del bad["response"]["target"]
            bad_enum = dict(decision(self.req), last_outcome=PRIVATE)
            following = request(4)
            wrapper, inner, transport, _, log = self.client(cls, protocol, [
                envelope(protocol, bad), envelope(protocol, bad_enum),
                envelope(protocol, decision(self.req)), envelope(protocol, decision(following))],
                schema_retries=2)
            wrapper.review(self.req)
            wrapper.review(following)
            attempts = [call.args[0] for call in inner.review.call_args_list]
            self.assertIs(attempts[0], self.req)
            self.assertIs(attempts[3], following)
            for attempt in attempts[1:3]:
                self.assertIsNot(attempt, self.req)
                for field in fields(ReviewRequest):
                    if field.name != "retry_feedback":
                        self.assertIs(getattr(attempt, field.name), getattr(self.req, field.name))
            first = self.assert_only_feedback_changed(transport, 0, 1)
            second = self.assert_only_feedback_changed(transport, 0, 2)
            self.assertEqual(first["missing_keys"], ["target"])
            self.assertEqual(second["code"], "schema_enum")
            self.assertEqual(second["location"], ["last_outcome"])
            self.assertNotIn("missing_keys", second)
            self.assertIsNone(self.payload(transport, 3)[2]["retry_feedback"])
            self.assertEqual(self.req, snapshot)
            self.assertIsNone(self.req.retry_feedback)
            self.assert_private_absent(transport, log)

    def test_exhausted_feedback_is_not_inherited_by_next_review(self):
        for cls, protocol in PROTOCOLS:
            bad = decision(self.req)
            del bad["response"]
            wrapper, _, transport, _, _ = self.client(cls, protocol, [
                envelope(protocol, bad), envelope(protocol, bad), envelope(protocol, decision(self.req))])
            with self.assertRaises(ReviewSchemaError):
                wrapper.review(self.req)
            wrapper.review(self.req)
            self.assertIsNone(self.payload(transport, 2)[2]["retry_feedback"])
            self.assertEqual((wrapper.calls, wrapper.schema_rejects), (1, 2))

    def test_incoming_stale_feedback_is_cleared_without_mutating_request(self):
        req = replace(self.req, retry_feedback={"constraint": PRIVATE})
        for cls, protocol in PROTOCOLS:
            wrapper, inner, transport, _, log = self.client(cls, protocol, [envelope(protocol, decision(req))])
            wrapper.review(req)
            self.assertIsNone(self.payload(transport, 0)[2]["retry_feedback"])
            self.assertIsNot(inner.review.call_args.args[0], req)
            self.assertEqual(req.retry_feedback, {"constraint": PRIVATE})
            self.assert_private_absent(transport, log)

    def test_retry_does_not_attach_new_images_or_private_metadata(self):
        image = Path(self.directory.name) / "original.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"offline-fixture")
        payload = dict(self.req.observation.payload, images=[{"path": str(image), "camera": "front"}],
                       password=PRIVATE, native_reward=99, object_truth=PRIVATE)
        now = replace(self.req.observation, payload=payload)
        previous = replace(self.req.previous, before=replace(self.req.previous.before, payload=payload), after=now)
        req = replace(self.req, observation=now, proposal=replace(self.req.proposal, observation=now), previous=previous)
        for cls, protocol in PROTOCOLS:
            bad = dict(decision(req), last_outcome="unknown", recovery_status="succeeded")
            wrapper, _, transport, _, log = self.client(cls, protocol, [
                envelope(protocol, bad), envelope(protocol, decision(req))])
            wrapper.review(req)
            self.assert_only_feedback_changed(transport, 0, 1)
            content = self.payload(transport, 1)[1]
            self.assertEqual(sum(part["type"] in ("image", "input_image") for part in content), 2)
            text = content[0]["text"]
            for private in (str(image), "native_reward", "native_success", "object_truth", '"features"'):
                self.assertNotIn(private, text)
            self.assert_private_absent(transport, log)

    def test_malformed_envelopes_get_local_feedback_not_remote_text(self):
        for cls, protocol in PROTOCOLS:
            codes = ("response_envelope", "response_max_tokens") if protocol == "messages" else ("response_envelope",)
            for code in codes:
                with self.subTest(protocol=protocol, code=code):
                    bad = envelope(protocol, decision(self.req))
                    if protocol == "messages":
                        bad.update(stop_reason="max_tokens" if code == "response_max_tokens" else PRIVATE,
                                   content=[{"type": "text", "text": PRIVATE}])
                    else:
                        bad["output_text"] = PRIVATE
                    wrapper, inner, transport, _, log = self.client(cls, protocol, [
                        bad, envelope(protocol, decision(self.req))])
                    wrapper.review(self.req)
                    self.assertEqual(self.payload(transport, 1)[2]["retry_feedback"]["code"], code)
                    self.assertEqual(inner.usage_history[0].input_tokens, 10)
                    self.assertEqual(self.records(log)[0]["input_tokens"], 10)
                    for record in self.records(log):
                        self.assertNotIn("failure_reason", record)
                        self.assertNotIn("transport_error", record)
                    self.assert_private_absent(transport, log)

    def test_unknown_codes_and_invalid_paths_get_identical_fixed_feedback(self):
        generic = None
        for cls, protocol in PROTOCOLS:
            for error in (ReviewConstraintError(PRIVATE, [PRIVATE]),
                          ReviewConstraintError([PRIVATE]),
                          ReviewConstraintError("schema_type", [PRIVATE]),
                          ReviewSchemaError([PRIVATE], [PRIVATE], 1)):
                with self.subTest(protocol=protocol, error_type=type(error).__name__):
                    error.args = (PRIVATE,)
                    wrapper, _, transport, _, log = self.client(cls, protocol, [
                        envelope(protocol, decision(self.req)), envelope(protocol, decision(self.req))])
                    with patch.object(module, "validate_review", side_effect=[error, None]):
                        wrapper.review(self.req)
                    feedback = self.payload(transport, 1)[2]["retry_feedback"]
                    self.assertEqual(feedback["code"], "validation_failed")
                    self.assertEqual(set(feedback), {"code", "constraint"})
                    if generic is None:
                        generic = feedback
                    self.assertEqual(feedback, generic)
                    self.assert_private_absent(transport, log)

    def test_only_schema_names_are_allowed_in_diagnostics(self):
        for cls, protocol in PROTOCOLS:
            error = ReviewSchemaError(["response"], ["target", PRIVATE, "stage", object()], 1)
            error.args = (PRIVATE,)
            wrapper, _, transport, _, log = self.client(cls, protocol, [
                envelope(protocol, decision(self.req)), envelope(protocol, decision(self.req))])
            with patch.object(module, "validate_review", side_effect=[error, None]):
                wrapper.review(self.req)
            feedback = self.payload(transport, 1)[2]["retry_feedback"]
            self.assertEqual(feedback["location"], ["response"])
            self.assertEqual(feedback["missing_keys"], ["target"])
            self.assertEqual(self.records(log)[0]["schema_error"]["missing_keys"], ["target"])
            self.assert_private_absent(transport, log)

    def test_array_schema_location_is_retained_without_the_rejected_value(self):
        for cls, protocol in PROTOCOLS:
            bad = decision(self.req)
            bad["response"]["target"]["left"]["position"][1] = PRIVATE
            wrapper, _, transport, _, log = self.client(cls, protocol, [
                envelope(protocol, bad), envelope(protocol, decision(self.req))])
            wrapper.review(self.req)
            feedback = self.payload(transport, 1)[2]["retry_feedback"]
            self.assertEqual(feedback["code"], "schema_finite_number")
            self.assertEqual(feedback["location"], ["response", "target", "left", "position", "[]"])
            self.assert_private_absent(transport, log)

    def test_transport_and_schema_budgets_remain_separate(self):
        for cls, protocol in PROTOCOLS:
            bad = decision(self.req)
            del bad["response"]["target"]
            bad_enum = dict(decision(self.req), last_outcome=PRIVATE)
            wrapper, inner, transport, sleeps, log = self.client(cls, protocol, [
                TimeoutError(PRIVATE), envelope(protocol, bad), TimeoutError(PRIVATE),
                envelope(protocol, bad_enum), envelope(protocol, decision(self.req))],
                schema_retries=2, retry_wait=2, max_retries=2)
            wrapper.review(self.req)
            self.assertEqual((inner.calls, wrapper.calls, wrapper.stalls, wrapper.schema_rejects), (5, 1, 2, 2))
            self.assertEqual(sleeps, [2, .5, 2, .5])
            attempts = [call.args[0] for call in inner.review.call_args_list]
            self.assertIs(attempts[0], attempts[1])
            self.assertIs(attempts[2], attempts[3])
            self.assertIsNot(attempts[3], attempts[4])
            self.assertEqual(transport.call_args_list[2].args[2], transport.call_args_list[3].args[2])
            self.assertEqual([r["outcome"] for r in self.records(log)],
                             ["transport_stall", "schema_reject", "transport_stall", "schema_reject", "ok"])
            self.assertEqual([r["attempt"] for r in self.records(log)], [1, 2, 3, 4, 5])
            self.assertEqual([r["transport_attempt"] for r in self.records(log)
                              if r["outcome"] == "transport_stall"], [1, 2])
            self.assertEqual(inner.usage_history[0], Usage())
            self.assertEqual(inner.usage_history[2], Usage())
            self.assertEqual(len(wrapper.usage_history), 1)
            self.assert_private_absent(transport, log)

    def test_transport_exhaustion_and_disabled_defaults_still_stop(self):
        for cls, protocol in PROTOCOLS:
            for wait, budget, attempts in ((0, 0, 1), (0, 9, 1), (1, 2, 3)):
                with self.subTest(protocol=protocol, wait=wait):
                    wrapper, inner, transport, sleeps, log = self.client(cls, protocol,
                        [TimeoutError(PRIVATE)] * 4, retry_wait=wait, max_retries=budget)
                    with self.assertRaises(ReviewTransportError) as caught:
                        wrapper.review(self.req)
                    if wait:
                        self.assert_exhausted(caught.exception, attempts)
                    else:
                        self.assertNotIsInstance(caught.exception, ReviewTransportExhaustedError)
                        self.assertEqual(error_diagnostics(caught.exception), {})
                        self.assertNotIn("failure_reason", self.records(log)[-1])
                    self.assertEqual((inner.calls, transport.call_count), (attempts, attempts))
                    self.assertEqual(wrapper.calls, 0)
                    self.assertEqual(sleeps, [wait] * (attempts - 1))
                    self.assertFalse(self.records(log)[-1]["will_retry"])
                    self.assert_private_absent(transport, log)
            # A schema rejection must not reset already consumed transport budget.
            bad = decision(self.req)
            del bad["response"]
            wrapper, inner, transport, sleeps, log = self.client(cls, protocol, [
                TimeoutError(PRIVATE), envelope(protocol, bad), TimeoutError(PRIVATE),
                envelope(protocol, decision(self.req))], retry_wait=1, max_retries=1)
            with self.assertRaises(ReviewTransportExhaustedError) as caught:
                wrapper.review(self.req)
            diagnostics = self.assert_exhausted(caught.exception, 2)
            records = self.records(log)
            self.assertEqual([r["attempt"] for r in records], [1, 2, 3])
            self.assertEqual([r["transport_attempt"] for r in records
                              if r["outcome"] == "transport_stall"], [1, 2])
            for key, value in diagnostics.items():
                self.assertEqual(records[-1][key], value)
            self.assertEqual((inner.calls, wrapper.calls, wrapper.stalls, wrapper.schema_rejects), (3, 0, 2, 1))
            self.assertEqual(sleeps, [1, .5])
            self.assert_private_absent(transport, log)

    def test_unlimited_transport_setting_and_schema_budget_are_unchanged(self):
        for cls, protocol in PROTOCOLS:
            wrapper, inner, transport, sleeps, log = self.client(cls, protocol, [
                TimeoutError(PRIVATE)] * 12 + [envelope(protocol, decision(self.req))],
                retry_wait=1, max_retries=0)
            wrapper.review(self.req)
            self.assertEqual((inner.calls, transport.call_count, wrapper.stalls, wrapper.calls), (13, 13, 12, 1))
            self.assertEqual(sleeps, [1] * 12)
            self.assert_private_absent(transport, log)
            # Interleaved transport errors must not reset the schema budget either.
            bad = decision(self.req)
            del bad["response"]
            wrapper, inner, transport, _, log = self.client(cls, protocol, [
                envelope(protocol, bad), TimeoutError(PRIVATE), envelope(protocol, bad),
                envelope(protocol, decision(self.req))], retry_wait=1, max_retries=0)
            with self.assertRaises(ReviewSchemaError) as caught:
                wrapper.review(self.req)
            self.assertNotIn("failure_reason", error_diagnostics(caught.exception))
            self.assertNotIn("failure_reason", self.records(log)[-1])
            self.assertEqual((inner.calls, wrapper.calls, wrapper.schema_rejects), (3, 0, 2))
            self.assert_private_absent(transport, log)

    def assert_exhausted(self, error, attempts):
        self.assertIsInstance(error, ReviewTransportError)
        self.assertIsInstance(error, ReviewTransportExhaustedError)
        self.assertIs(type(error.attempts), int)
        self.assertEqual(error.attempts, attempts)
        expected = {"failure_reason": "reviewer_no_response", "transport_error": {
            "code": "review_transport_exhausted", "attempts": attempts}}
        self.assertEqual(error_diagnostics(error), expected)
        for secret in (PRIVATE, "offline-test-secret"):
            self.assertNotIn(secret, str(error))
        return expected

    def test_ten_transport_failures_exhaust_default_and_explicit_budget(self):
        for cls, protocol in PROTOCOLS:
            for options in ({}, {"max_retries": 9}):
                with self.subTest(protocol=protocol, options=options):
                    wrapper, inner, transport, sleeps, log = self.client(cls, protocol,
                        [TimeoutError(PRIVATE)] * 10 + [envelope(protocol, decision(self.req))],
                        retry_wait=120, **options)
                    self.assertEqual(wrapper.max_retries, 9)
                    with self.assertRaises(ReviewTransportExhaustedError) as caught:
                        wrapper.review(self.req)
                    diagnostics = self.assert_exhausted(caught.exception, 10)
                    self.assertEqual((inner.calls, inner.review.call_count, transport.call_count), (10, 10, 10))
                    self.assertEqual((wrapper.calls, wrapper.stalls, wrapper.schema_rejects), (0, 10, 0))
                    self.assertEqual(wrapper.usage_history, [])
                    self.assertEqual(sleeps, [120] * 9)
                    records = self.records(log)
                    self.assertEqual([r["outcome"] for r in records], ["transport_stall"] * 10)
                    self.assertEqual([r["attempt"] for r in records], list(range(1, 11)))
                    self.assertEqual([r["transport_attempt"] for r in records], list(range(1, 11)))
                    self.assertEqual([r["will_retry"] for r in records], [True] * 9 + [False])
                    self.assertEqual([r["retry_wait_seconds"] for r in records], [120] * 9 + [0])
                    for key, value in diagnostics.items():
                        self.assertEqual(records[-1][key], value)
                    for record in records[:-1]:
                        self.assertNotIn("failure_reason", record)
                        self.assertNotIn("transport_error", record)
                    self.assert_private_absent(transport, log)

    def test_tenth_request_success_and_exhaustion_both_reset_next_review_budget(self):
        for cls, protocol in PROTOCOLS:
            for exhausted in (False, True):
                with self.subTest(protocol=protocol, exhausted=exhausted):
                    failures = 10 if exhausted else 9
                    batch = [TimeoutError(PRIVATE)] * failures
                    if not exhausted:
                        batch.append(envelope(protocol, decision(self.req)))
                    wrapper, inner, transport, sleeps, log = self.client(
                        cls, protocol, batch * 2, retry_wait=120)
                    for invocation in (1, 2):
                        if exhausted:
                            with self.assertRaises(ReviewTransportExhaustedError) as caught:
                                wrapper.review(self.req)
                            self.assert_exhausted(caught.exception, 10)
                        else:
                            self.assertIsInstance(wrapper.review(self.req), Review)
                        self.assertEqual((inner.calls, transport.call_count), (10 * invocation,) * 2)
                        self.assertEqual(sleeps, [120] * (9 * invocation))
                        records = self.records(log)[-10:]
                        self.assertEqual([r["attempt"] for r in records], list(range(1, 11)))
                        self.assertEqual([r["transport_attempt"] for r in records
                                          if r["outcome"] == "transport_stall"], list(range(1, failures + 1)))
                        self.assertEqual(records[-1]["outcome"], "transport_stall" if exhausted else "ok")
                        self.assertTrue(all(r["call_index"] == (0 if exhausted else invocation - 1)
                                            for r in records))
                    self.assertEqual((wrapper.calls, len(wrapper.usage_history)), (0, 0) if exhausted else (2, 2))
                    self.assertEqual(wrapper.stalls, failures * 2)
                    self.assert_private_absent(transport, log)

    def test_no_response_diagnostics_ignore_private_exception_text(self):
        for attempts in (1, 10, 23):
            error = ReviewTransportExhaustedError(attempts)
            expected = self.assert_exhausted(error, attempts)
            error.args = (PRIVATE, "offline-test-secret")
            self.assertEqual(error_diagnostics(error), expected)
        self.assertEqual(error_diagnostics(ReviewTransportError(PRIVATE)), {})

    def test_heartbeat_precedes_every_schema_and_transport_attempt(self):
        for cls, protocol in PROTOCOLS:
            bad = decision(self.req)
            del bad["response"]
            heartbeat, wait, events = Mock(), Mock(), Mock()
            wrapper, inner, transport, _, log = self.client(cls, protocol, [
                TimeoutError(PRIVATE), envelope(protocol, bad), TimeoutError(PRIVATE),
                envelope(protocol, bad), envelope(protocol, decision(self.req))],
                schema_retries=2, retry_wait=120, before_attempt=heartbeat, sleep=wait)
            events.attach_mock(heartbeat, "heartbeat")
            events.attach_mock(inner.review, "model")
            events.attach_mock(wait, "wait")
            self.assertIsInstance(wrapper.review(self.req), Review)
            self.assertEqual([c[0] for c in events.mock_calls],
                             ["heartbeat", "model", "wait"] * 4 + ["heartbeat", "model"])
            self.assertEqual(heartbeat.call_args_list, [call()] * 5)
            self.assertEqual(wait.call_args_list, [call(120), call(.5), call(120), call(.5)])
            self.assertEqual([r["attempt"] for r in self.records(log)], [1, 2, 3, 4, 5])
            self.assert_private_absent(transport, log)

    def test_callback_failure_never_calls_model_or_consumes_retry_budget(self):
        for cls, protocol in PROTOCOLS:
            for prior in (0, 1):
                for error in (HarnessError(PRIVATE), ReviewTransportError(PRIVATE),
                              ReviewSchemaError([], [], 0)):
                    with self.subTest(protocol=protocol, prior=prior, error=type(error).__name__):
                        heartbeat = Mock(side_effect=[None] * prior + [error])
                        wrapper, inner, transport, sleeps, log = self.client(cls, protocol,
                            [TimeoutError(PRIVATE)] * prior + [envelope(protocol, decision(self.req))],
                            retry_wait=120, before_attempt=heartbeat)
                        with self.assertRaises(type(error)) as caught:
                            wrapper.review(self.req)
                        self.assertIs(caught.exception, error)
                        self.assertEqual(heartbeat.call_args_list, [call()] * (prior + 1))
                        self.assertEqual((inner.calls, inner.review.call_count, transport.call_count), (prior,) * 3)
                        self.assertEqual((wrapper.calls, wrapper.stalls, wrapper.schema_rejects), (0, prior, 0))
                        self.assertEqual(wrapper.usage_history, [])
                        self.assertEqual(sleeps, [120] * prior)
                        records = self.records(log) if log.exists() else []
                        self.assertEqual([r["outcome"] for r in records], ["transport_stall"] * prior)
                        if records:
                            self.assertTrue(records[-1]["will_retry"])
                            self.assertNotIn("failure_reason", records[-1])
                        if log.exists():
                            self.assert_private_absent(transport, log)

    def test_metadata_heartbeat_stays_fresh_during_3420_virtual_seconds(self):
        for cls, protocol in PROTOCOLS:
            now, beats, waits = [0.0], [], []
            backend = Mock()
            backend.metadata.side_effect = lambda: beats.append(now[0])

            def timeout(*args):
                self.assertEqual(beats[-1], now[0])
                now[0] += 180
                raise TimeoutError(PRIVATE)

            def wait(seconds):
                waits.append(seconds)
                now[0] += seconds

            wrapper, inner, transport, _, log = self.client(cls, protocol, timeout,
                retry_wait=180, before_attempt=backend.metadata, sleep=wait, clock=lambda: now[0])
            with self.assertRaises(ReviewTransportExhaustedError) as caught:
                wrapper.review(self.req)
            self.assert_exhausted(caught.exception, 10)
            self.assertEqual((inner.calls, transport.call_count), (10, 10))
            self.assertEqual(backend.metadata.call_args_list, [call()] * 10)
            self.assertEqual(now[0], 3420)
            self.assertEqual(waits, [180] * 9)
            self.assertEqual(beats, list(range(0, 3241, 360)))
            boundaries = beats + [now[0]]
            self.assertLess(max(b - a for a, b in zip(boundaries, boundaries[1:])), 900)
            self.assertEqual([r["latency_seconds"] for r in self.records(log)], [180] * 10)
            self.assert_private_absent(transport, log)

    def test_generic_failure_is_not_retried_or_logged_as_raw_text(self):
        for cls, protocol in PROTOCOLS:
            wrapper, inner, transport, sleeps, log = self.client(cls, protocol, [
                envelope(protocol, decision(self.req))], schema_retries=2, retry_wait=1)
            with patch.object(module, "validate_review", side_effect=HarnessError(PRIVATE)):
                with self.assertRaises(HarnessError):
                    wrapper.review(self.req)
            self.assertEqual((inner.calls, wrapper.calls), (1, 0))
            self.assertEqual(sleeps, [])
            self.assertEqual(self.records(log)[0]["outcome"], "error")
            self.assert_private_absent(transport, log)

    def test_both_protocol_instructions_state_existing_consistency_rules(self):
        for cls, protocol in PROTOCOLS:
            wrapper, _, transport, _, _ = self.client(cls, protocol, [envelope(protocol, decision(self.req))])
            wrapper.review(self.req)
            body = self.payload(transport, 0)[0]
            instructions = body.get("instructions", body.get("system"))
            for text in ("execution_status=failed iff last_outcome=error",
                         "uncertain/not_started require\nlast_outcome=unknown",
                         "recovered requires last_outcome=ok",
                         "succeeded only when that goal is visibly achieved\nAND last_outcome=ok",
                         "Never change\nerror/unknown to ok merely to pass validation",
                         "Recheck the original evidence",
                         "keep last_outcome=unknown and recovery_status=continue"):
                self.assertIn(text, instructions)


class RequestTraceV7Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="request-trace-v7-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.log = Path(self.directory.name) / "calls.jsonl"
        self.state = self.log.with_name("calls.request.json")
        self.events = self.log.with_name("calls.requests.jsonl")
        self.req = request()
        for guard in (patch.dict(module.os.environ, {"TEST_TRACE_KEY": "offline-test-secret"}),
                      patch("socket.create_connection", side_effect=AssertionError("No network")),
                      patch("socket.socket.connect", side_effect=AssertionError("No network")),
                      patch("time.sleep", side_effect=AssertionError("No real sleep"))):
            guard.start()
            self.addCleanup(guard.stop)

    def wrapper(self, **settings):
        inner = module.ResponsesReviewer("https://example.invalid/v1/responses",
                                         "offline-model", "TEST_TRACE_KEY")
        options = dict(log_path=str(self.log), episode="ep1", retry_wait=180,
                       max_retries=9, schema_retries=0, sleep=lambda _: None)
        options.update(settings)
        return module.TimedRetryReviewer(inner, **options)

    def response(self, raw=None, status=200, headers=None):
        if raw is None:
            raw = json.dumps(envelope("responses", decision(self.req))).encode()
        response = Mock(status=status, headers=headers or {"x-request-id": "gateway-123"})
        response.read.side_effect = BytesIO(raw).read
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    def records(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def snapshot(self):
        return json.loads(self.state.read_text())

    def assert_safe(self):
        text = "".join(path.read_text() for path in (self.log, self.events, self.state) if path.exists())
        for value in (PRIVATE, "offline-test-secret", "Authorization", "output_text", "Bearer"):
            self.assertNotIn(value, text)

    def test_real_transport_boundaries_and_atomic_state_are_visible_before_completion(self):
        response = self.response()
        read = response.read.side_effect

        def read_body(size):
            state = self.snapshot()
            self.assertEqual(state["phase"], "response_headers" if size == 1 else "response_body_started")
            self.assertIsNone(state["request_finished_at"])
            return read(size)

        def open_response(*args, **kwargs):
            self.assertEqual(self.snapshot()["phase"], "request_started")
            self.assertIsNone(self.snapshot()["http_status"])
            return response

        response.read.side_effect = read_body
        opener = Mock()
        opener.open.side_effect = open_response
        heartbeat = Mock(side_effect=lambda: self.assertEqual(self.snapshot()["phase"], "keepalive"))
        with patch.object(module.urllib.request, "build_opener", return_value=opener):
            result = self.wrapper(before_attempt=heartbeat).review(self.req)
        self.assertIsInstance(result, Review)
        events = self.records(self.events)
        self.assertEqual([event["phase"] for event in events], [
            "attempt_started", "keepalive", "preparing", "request_started", "response_headers",
            "response_body_started", "response_body_complete", "validating", "accepted"])
        state = self.snapshot()
        self.assertEqual(state, events[-1])
        self.assertEqual(self.records(self.log)[0]["request"], state)
        self.assertEqual(state["controller_pid"], module.os.getpid())
        ticks = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19]
        self.assertEqual(state["process_start_ticks"], ticks)
        self.assertEqual(state["proposal_request_id"], self.req.proposal.request_id)
        self.assertRegex(state["local_request_id"], r"^[0-9a-f]{32}$")
        self.assertEqual((state["http_status"], state["gateway_request_id"]), (200, "gateway-123"))
        self.assertEqual(state["request_finished_at"], events[-3]["phase_at"])
        self.assertGreaterEqual(state["request_finished_at"], state["request_started_at"])
        self.assertEqual(response.read.call_args_list, [call(1), call(module.MAX_RESPONSE_BYTES)])
        self.assertFalse(list(self.state.parent.glob(".*.tmp")))
        self.assert_safe()

    def test_headers_are_not_mistaken_for_first_body_byte_on_timeout(self):
        response = self.response()
        response.read.side_effect = TimeoutError(PRIVATE)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(module.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(ReviewTransportError):
                self.wrapper(retry_wait=0).review(self.req)
        events = self.records(self.events)
        self.assertNotIn("response_body_started", [event["phase"] for event in events])
        state = self.snapshot()
        self.assertEqual((state["phase"], state["error_kind"], state["last_transport_phase"]),
                         ("failed", "timeout", "response_headers"))
        self.assertEqual(state["http_status"], 200)
        self.assertIsNone(state["response_bytes"])
        self.assertIsNotNone(state["request_finished_at"])
        self.assert_safe()

    def test_partial_body_timeout_preserves_first_byte_evidence(self):
        response = self.response()
        response.read.side_effect = [b"{", TimeoutError(PRIVATE)]
        opener = Mock()
        opener.open.return_value = response
        with patch.object(module.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(ReviewTransportError):
                self.wrapper(retry_wait=0).review(self.req)
        state = self.snapshot()
        self.assertEqual(state["last_transport_phase"], "response_body_started")
        self.assertEqual(state["response_bytes"], 1)
        self.assertEqual(state["error_kind"], "timeout")
        self.assert_safe()

    def test_invalid_empty_and_oversized_bodies_remain_schema_rejections(self):
        for raw in (b"", PRIVATE.encode(), b" " * (module.MAX_RESPONSE_BYTES + 5)):
            with self.subTest(size=len(raw)):
                response = self.response(raw)
                opener = Mock()
                opener.open.return_value = response
                wrapper = self.wrapper()
                with patch.object(module.urllib.request, "build_opener", return_value=opener):
                    with self.assertRaises(ReviewConstraintError) as caught:
                        wrapper.review(self.req)
                self.assertEqual(caught.exception.code, "response_envelope")
                self.assertEqual((wrapper.stalls, wrapper.schema_rejects), (0, 1))
                self.assertEqual(self.snapshot()["error_kind"], "schema")
                self.assertEqual(self.snapshot()["response_bytes"], min(len(raw), module.MAX_RESPONSE_BYTES + 1))
                self.assertEqual(opener.open.call_count, 1)
        self.assert_safe()

    def test_http_errors_keep_only_allowlisted_status_and_request_id(self):
        for code in (429, 503):
            with self.subTest(code=code):
                body = BytesIO(PRIVATE.encode())
                error = urllib.error.HTTPError("https://example.invalid/v1/responses", code, PRIVATE,
                    {"x-request-id": "gateway-safe:123", "set-cookie": PRIVATE}, body)
                opener = Mock()
                opener.open.side_effect = error
                with patch.object(module.urllib.request, "build_opener", return_value=opener):
                    with self.assertRaises(ReviewTransportError):
                        self.wrapper(retry_wait=0).review(self.req)
                state = self.snapshot()
                self.assertEqual((state["http_status"], state["gateway_request_id"], state["error_kind"]),
                                 (code, "gateway-safe:123", "http"))
                self.assertIsNone(state["response_bytes"])
                self.assertTrue(body.closed)
        self.assert_safe()

    def test_response_id_rejects_credentials_and_untrusted_characters(self):
        headers = {"Authorization": "Bearer offline-test-secret", "x-api-key": "second-secret"}
        for value in ("offline-test-secret", "second-secret", "offline-test-secret-suffix",
                      "<script>", "id\nprivate", "a" * 129):
            with self.subTest(value=value):
                response = self.response(headers={"x-request-id": value})
                self.assertIsNone(module._response_metadata(response, headers)["gateway_request_id"])
        response = self.response(headers={"x-request-id": "<bad>", "request-id": "fallback-id"})
        self.assertEqual(module._response_metadata(response, headers)["gateway_request_id"], "fallback-id")

    def test_transport_error_types_are_safe_and_do_not_invent_response_headers(self):
        for error, kind in ((TimeoutError(PRIVATE), "timeout"),
                            (urllib.error.URLError(TimeoutError(PRIVATE)), "timeout"),
                            (urllib.error.URLError(ssl.SSLError(PRIVATE)), "tls"),
                            (urllib.error.URLError(PRIVATE), "connection"),
                            (ConnectionResetError(PRIVATE), "connection"),
                            (OSError(PRIVATE), "transport")):
            with self.subTest(kind=kind):
                opener = Mock()
                opener.open.side_effect = error
                with patch.object(module.urllib.request, "build_opener", return_value=opener):
                    with self.assertRaises(ReviewTransportError):
                        self.wrapper(retry_wait=0).review(self.req)
                state = self.snapshot()
                self.assertEqual(state["error_kind"], kind)
                self.assertIsNone(state["http_status"])
                self.assertEqual(state["last_transport_phase"], "request_started")
        self.assert_safe()

    def test_retry_countdown_fixed_finish_time_and_unique_attempt_ids(self):
        opener = Mock()
        opener.open.side_effect = [TimeoutError(PRIVATE), self.response(PRIVATE.encode()), self.response()]
        waiting = []

        def wait(seconds):
            state = self.snapshot()
            waiting.append(state)
            self.assertEqual(state["phase"], "retry_wait")
            self.assertEqual(state["retry_wait_seconds"], seconds)
            self.assertAlmostEqual(state["retry_at"] - state["phase_at"], seconds, delta=.2)
            self.assertLessEqual(state["request_finished_at"], state["phase_at"])

        with patch.object(module.urllib.request, "build_opener", return_value=opener):
            wrapper = self.wrapper(schema_retries=1, sleep=wait)
            wrapper.review(self.req)
        records = self.records(self.log)
        self.assertEqual([record["outcome"] for record in records], ["transport_stall", "schema_reject", "ok"])
        self.assertEqual([state["retry_reason"] for state in waiting], ["transport", "schema"])
        self.assertEqual(len({record["request"]["local_request_id"] for record in records}), 3)
        self.assertEqual([record["request"]["attempt"] for record in records], [1, 2, 3])
        self.assertEqual((self.snapshot()["transport_failures"], self.snapshot()["schema_failures"]), (1, 1))
        self.assertEqual(self.snapshot()["max_transport_attempts"], 10)
        self.assertIsNone(self.snapshot()["retry_at"])
        self.assertIsNone(self.snapshot()["error_kind"])
        self.assert_safe()

    def test_ten_failures_finish_without_an_eleventh_request(self):
        opener = Mock()
        opener.open.side_effect = TimeoutError(PRIVATE)
        waiting = []
        with patch.object(module.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(ReviewTransportExhaustedError):
                self.wrapper(sleep=waiting.append).review(self.req)
        self.assertEqual(opener.open.call_count, 10)
        self.assertEqual(waiting, [180] * 9)
        self.assertEqual(len(self.records(self.log)), 10)
        state = self.snapshot()
        self.assertEqual((state["phase"], state["attempt"], state["transport_failures"]), ("failed", 10, 10))
        self.assertIsNone(state["retry_at"])
        self.assert_safe()

    def test_keepalive_failure_never_starts_http_or_logs_a_model_call(self):
        heartbeat = Mock(side_effect=HarnessError(PRIVATE))
        with patch.object(module.urllib.request, "build_opener") as opener:
            with self.assertRaises(HarnessError):
                self.wrapper(before_attempt=heartbeat).review(self.req)
        opener.assert_not_called()
        state = self.snapshot()
        self.assertEqual((state["phase"], state["error_kind"]), ("failed", "keepalive"))
        self.assertIsNone(state["request_started_at"])
        self.assertIsNone(state["request_finished_at"])
        self.assertFalse(self.log.exists())
        self.assert_safe()

    def test_telemetry_write_failure_does_not_change_the_review(self):
        opener = Mock()
        opener.open.return_value = self.response()
        wrapper = self.wrapper()
        with patch.object(module.urllib.request, "build_opener", return_value=opener), \
                patch.object(module.os, "replace", side_effect=OSError(PRIVATE)), \
                patch.object(module.sys, "stderr") as stderr:
            result = wrapper.review(self.req)
        self.assertIsInstance(result, Review)
        self.assertEqual(wrapper.calls, 1)
        record = self.records(self.log)[0]
        self.assertEqual(record["outcome"], "ok")
        self.assertEqual(record["request"]["telemetry_error"], "write_failed")
        self.assertNotIn(PRIVATE, str(stderr.write.call_args_list))
        self.assert_safe()

    def test_no_log_keeps_telemetry_disabled(self):
        opener = Mock()
        opener.open.return_value = self.response()
        wrapper = self.wrapper(log_path=None)
        with patch.object(module.urllib.request, "build_opener", return_value=opener), \
                patch.object(module, "_RequestTrace") as trace:
            self.assertIsInstance(wrapper.review(self.req), Review)
        trace.assert_not_called()
        self.assertIsNone(wrapper._trace)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

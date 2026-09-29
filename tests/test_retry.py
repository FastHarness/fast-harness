"""Regression tests for ReviewTransportError and the TimedRetryReviewer wrapper.

Covers: transport failures raise a distinct (but still HarnessError) transport error; the wrapper
resends only transport stalls after a wait; schema/constraint errors are never retried; per-call
accounting counts only successful reviews; and the call log records latency/usage/outcome."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

from fast_harness import reviewer as module
from fast_harness.reviewer import ResponsesReviewer, TimedRetryReviewer
from fast_harness.types import (HarnessError, Review, ReviewConstraintError, ReviewSchemaError,
                                ReviewTransportError, Usage)

from test_reviewer import request


def a_review():
    return Review("approach", "unknown", "none", "evidence",
                  {"mode": "student", "steps": 1}, usage=Usage(100, 0, 20, None))


class StubInner:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def review(self, request):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TransportErrorTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(module.os.environ, {"TEST_REVIEWER_KEY": "test-only-secret"}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_transport_failure_raises_transport_error_subclass(self):
        def boom(url, headers, body, timeout):
            raise urllib.error.URLError("timed out")

        reviewer = ResponsesReviewer("https://example.invalid/v1/responses", "m",
                                     "TEST_REVIEWER_KEY", transport=boom)
        with self.assertRaises(ReviewTransportError):
            reviewer.review(request())
        # Existing callers catch HarnessError; the subclass keeps that contract intact.
        self.assertTrue(issubclass(ReviewTransportError, HarnessError))
        # A single failed attempt is still counted with a placeholder usage by the inner reviewer.
        self.assertEqual(reviewer.calls, 1)


class WrapperTests(unittest.TestCase):
    def test_retries_transport_then_succeeds_with_clean_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "calls.jsonl"
            sleeps = []
            success = a_review()
            inner = StubInner([ReviewTransportError("x"), ReviewTransportError("x"), success])
            wrapper = TimedRetryReviewer(inner, log_path=str(log), retry_wait=120,
                                         sleep=sleeps.append, episode="ep1")
            result = wrapper.review(request())
            self.assertIs(result, success)
            self.assertEqual(inner.calls, 3)          # two stalled attempts plus one success
            self.assertEqual(wrapper.calls, 1)        # only the successful review is counted
            self.assertEqual(len(wrapper.usage_history), 1)
            self.assertEqual(wrapper.usage_history[0], Usage(100, 0, 20, None))
            self.assertEqual(wrapper.stalls, 2)
            self.assertEqual(sleeps, [120, 120])
            records = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r["outcome"] for r in records],
                             ["transport_stall", "transport_stall", "ok"])
            self.assertTrue(all(r["episode"] == "ep1" for r in records))
            self.assertTrue(all(r["call_index"] == 0 for r in records))  # same review, index unchanged
            self.assertEqual(records[-1]["input_tokens"], 100)

    def test_default_is_inert_no_retry(self):
        inner = StubInner([ReviewTransportError("x")])
        wrapper = TimedRetryReviewer(inner)  # retry_wait defaults to 0
        with self.assertRaises(ReviewTransportError):
            wrapper.review(request())
        self.assertEqual(wrapper.calls, 0)
        self.assertEqual(wrapper.stalls, 1)

    def test_constraint_error_never_retried(self):
        sleeps = []
        inner = StubInner([ReviewConstraintError("response_envelope")])
        wrapper = TimedRetryReviewer(inner, retry_wait=120, sleep=sleeps.append)
        with self.assertRaises(ReviewConstraintError):
            wrapper.review(request())
        self.assertEqual(inner.calls, 1)
        self.assertEqual(wrapper.stalls, 0)
        self.assertEqual(sleeps, [])

    def test_max_retries_bounds_attempts(self):
        sleeps = []
        inner = StubInner([ReviewTransportError("x")] * 5)
        wrapper = TimedRetryReviewer(inner, retry_wait=1, max_retries=2, sleep=sleeps.append)
        with self.assertRaises(ReviewTransportError):
            wrapper.review(request())
        self.assertEqual(inner.calls, 3)   # attempts 1 and 2 retry, attempt 3 exceeds the bound
        self.assertEqual(sleeps, [1, 1])
        self.assertEqual(wrapper.stalls, 3)


class SchemaRetryTests(unittest.TestCase):
    """The inner reviewer validates internally and RAISES on a malformed model response; the wrapper
    re-asks (bounded) while validation stays unchanged — a non-deterministic model just gets another
    attempt."""

    def test_reask_on_schema_reject_then_succeeds_clean_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "calls.jsonl"
            sleeps = []
            good = a_review()
            # Inner raises two malformed-output errors, then returns an accepted review.
            inner = StubInner([ReviewSchemaError([], [], 1), ReviewConstraintError("schema_type", ["response"]), good])
            wrapper = TimedRetryReviewer(inner, log_path=str(log), schema_retries=5,
                                         schema_retry_wait=2, sleep=sleeps.append)
            result = wrapper.review(request())
            self.assertIs(result, good)
            self.assertEqual(inner.calls, 3)         # re-asked twice, third accepted
            self.assertEqual(wrapper.schema_rejects, 2)
            self.assertEqual(wrapper.calls, 1)       # only the accepted review counted
            self.assertEqual(len(wrapper.usage_history), 1)
            self.assertEqual(sleeps, [2, 2])
            records = [json.loads(l) for l in log.read_text().splitlines()]
            self.assertEqual([r["outcome"] for r in records], ["schema_reject", "schema_reject", "ok"])
            self.assertIn("schema_error", records[0])       # payload-free diagnostics carried through
            self.assertIn("validation_error", records[1])
            self.assertTrue(all(r["call_index"] == 0 for r in records))

    def test_schema_retry_gives_up_at_bound(self):
        inner = StubInner([ReviewConstraintError("response_envelope")] * 10)
        wrapper = TimedRetryReviewer(inner, schema_retries=2, schema_retry_wait=1, sleep=lambda s: None)
        with self.assertRaises(ReviewConstraintError):
            wrapper.review(request())
        self.assertEqual(inner.calls, 3)        # attempt 1 + 2 re-asks, then give up
        self.assertEqual(wrapper.schema_rejects, 3)
        self.assertEqual(wrapper.calls, 0)      # nothing accepted, nothing counted

    def test_schema_retry_default_off(self):
        inner = StubInner([ReviewSchemaError([], ["response"], 0)])
        wrapper = TimedRetryReviewer(inner)  # schema_retries defaults to 0
        with self.assertRaises(ReviewSchemaError):
            wrapper.review(request())
        self.assertEqual(inner.calls, 1)
        self.assertEqual(wrapper.calls, 0)


if __name__ == "__main__":
    unittest.main()

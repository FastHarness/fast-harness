"""Tests for the dynamic/aggressive scheduler knobs: confidence_cap (skip interval grows with
accumulated successes) and relax_known_stage_change. Default config must reproduce the original."""
import unittest

from fast_harness.scheduler import Config, Scheduler, interval, lower_success_bound
from fast_harness.memory import Statistics, Match
from fast_harness.types import Observation, Proposal


def stats(success, total, clean_streak=99, ever_failed=False):
    return Statistics(successes=float(success), failures=float(total - success), total=total,
                      clean_streak=clean_streak, ever_failed=ever_failed)


class IntervalTests(unittest.TestCase):
    def test_default_matches_original_formula(self):
        c = Config()  # confidence_cap=1.0 must reproduce the original static formula exactly
        for st in (stats(20, 20), stats(80, 80), stats(9, 12)):
            expected = 1 + int((c.max_interval - 1) * lower_success_bound(st) ** 2)
            self.assertEqual(interval(st, c), expected)
            self.assertLessEqual(interval(st, c), c.max_interval)  # original always caps at max_interval

    def test_confidence_cap_grows_gap_with_evidence(self):
        c = Config(confidence_cap=6.0, max_interval=20, min_samples=4)
        few = interval(stats(8, 8), c)     # total=8  -> confidence min(6, 2)=2
        many = interval(stats(48, 48), c)  # total=48 -> confidence min(6,12)=6
        self.assertGreater(many, few)      # dynamic: more accumulated successes -> longer skip
        self.assertGreater(many, Config().max_interval)  # can exceed the default ceiling

    def test_below_min_samples_and_cooldown_still_review(self):
        c = Config(confidence_cap=6.0)
        self.assertEqual(interval(stats(2, 2), c), 1)                              # too few samples
        self.assertEqual(interval(stats(20, 20, clean_streak=1, ever_failed=True), c), 1)  # cooldown after error

    def test_confidence_cap_validation(self):
        with self.assertRaises(ValueError):
            Config(confidence_cap=0.5)
        Config(confidence_cap=1.0)  # ok


class StageChangeTests(unittest.TestCase):
    class StubMemory:
        def __init__(self, st):
            self._st = st
        def statistics(self, stage):
            return self._st
        def near_error(self, stage, features, radius):
            return False

    def _reason(self, cfg, st):
        obs = Observation('e', 30, (0.0,), {}, False, None)
        prop = Proposal('r', obs, {}, 15, (), False)
        match = Match(stage='grasp', distance=0.01, reason='matched')
        return Scheduler(cfg).reason(self.StubMemory(st), match, obs, prop,
            last_review_step=15, chunks_since_review=1, previous_stage='approach',
            recovery_active=False, force_review=False, stalled=False)

    def test_default_forces_review_on_stage_change(self):
        reason, _, _ = self._reason(Config(audit_probability=0.0), stats(40, 40))
        self.assertEqual(reason, 'stage_changed')

    def test_relax_skips_transition_into_proven_stage(self):
        cfg = Config(confidence_cap=6.0, max_interval=20, relax_known_stage_change=True, audit_probability=0.0)
        reason, gap, _ = self._reason(cfg, stats(40, 40))       # new stage proven (gap>1)
        self.assertIsNone(reason)                            # skip allowed across the stage change
        self.assertGreater(gap, 1)

    def test_relax_still_reviews_transition_into_unproven_stage(self):
        cfg = Config(relax_known_stage_change=True, audit_probability=0.0)
        reason, _, _ = self._reason(cfg, stats(2, 2))           # new stage not yet proven (gap==1)
        self.assertEqual(reason, 'stage_changed')


if __name__ == '__main__':
    unittest.main()

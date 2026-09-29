"""v7 contact-gating tests.

  1. contact_gating=False must reproduce the C2 base (fast_harness) scheduler decision exactly.
  2. contact_imminent fires on gripper threshold-crossing and on EEF descent onto the surface.
  3. With gating on, a contact chunk forces review while a proven transit chunk still skips.

Run from the project root:  PYTHONPATH=. python tests/test_contact_gate_v7.py
"""
import unittest

import fast_harness.scheduler as base
import fast_harness.memory as bmem
import fast_harness.types as btypes
from fast_harness.scheduler import Config, Scheduler, contact_imminent
from fast_harness.memory import Statistics, Match
from fast_harness.types import Observation, Proposal


def stats(success, total, clean_streak=99, ever_failed=False):
    return Statistics(successes=float(success), failures=float(total - success), total=total,
                      clean_streak=clean_streak, ever_failed=ever_failed)


class StubMemory:
    def __init__(self, st):
        self._st = st
    def statistics(self, stage):
        return self._st
    def near_error(self, stage, features, radius):
        return False


def gripper_prop(*openings, arm='left', z=1.2):
    """Proposal whose <arm> gripper follows the opening sequence at a constant (high) height."""
    obs = Observation('e', 30, (0.0,), {}, False, None)
    payload = dict(
        current_eef={arm: dict(gripper_opening_command=openings[0], position=[0.0, 0.0, z])},
        student_eef_trajectory=[{arm: dict(gripper_opening_command=o, position=[0.0, 0.0, z])}
                                for o in openings[1:]])
    return Proposal('r', obs, payload, 15, (), False)


def descent_prop(z_start, z_end, arm='left', steps=5, gripper=1.0):
    obs = Observation('e', 30, (0.0,), {}, False, None)
    zs = [z_start + (z_end - z_start) * i / (steps - 1) for i in range(steps)]
    payload = dict(
        current_eef={arm: dict(gripper_opening_command=gripper, position=[0.0, 0.0, zs[0]])},
        student_eef_trajectory=[{arm: dict(gripper_opening_command=gripper, position=[0.0, 0.0, z])}
                                for z in zs])
    return Proposal('r', obs, payload, 15, (), False)


ON = dict(contact_gating=True, contact_low_z=0.95, contact_descent_m=0.05)


class ContactDetectorTests(unittest.TestCase):
    def test_gripper_close_is_contact(self):
        self.assertEqual(contact_imminent(gripper_prop(1.0, 0.9, 0.6, 0.3), Config(**ON)), 'contact_gripper')

    def test_gripper_open_is_contact(self):
        self.assertEqual(contact_imminent(gripper_prop(0.0, 0.2, 0.6, 0.9), Config(**ON)), 'contact_gripper')

    def test_steady_open_transit_not_contact(self):
        self.assertIsNone(contact_imminent(gripper_prop(1.0, 1.0, 1.0, 1.0), Config(**ON)))

    def test_steady_closed_carry_not_contact(self):
        # already grasped, carrying (gripper stays closed, high) -> transit, not contact
        self.assertIsNone(contact_imminent(gripper_prop(0.1, 0.1, 0.1), Config(**ON)))

    def test_descent_to_surface_is_contact(self):
        self.assertEqual(contact_imminent(descent_prop(1.05, 0.90), Config(**ON)), 'contact_approach')

    def test_high_transit_not_contact(self):
        self.assertIsNone(contact_imminent(descent_prop(1.30, 1.28), Config(**ON)))

    def test_empty_payload_safe(self):
        obs = Observation('e', 0, (0.0,), {}, False, None)
        self.assertIsNone(contact_imminent(Proposal('r', obs, {}, 15, (), False), Config(**ON)))


class GatingReasonTests(unittest.TestCase):
    def _reason(self, cfg, prop, st, prev, stage):
        match = Match(stage=stage, distance=0.01, reason='matched')
        return Scheduler(cfg).reason(StubMemory(st), match, prop.observation, prop,
            last_review_step=15, chunks_since_review=1, previous_stage=prev,
            recovery_active=False, force_review=False, stalled=False)

    def test_contact_forces_review(self):
        cfg = Config(contact_gating=True, gate_stage_change_on_contact_only=True,
                     confidence_cap=6.0, max_interval=20, min_samples=2, audit_probability=0.0)
        reason, _, _ = self._reason(cfg, gripper_prop(1.0, 0.6, 0.3), stats(40, 40), 'grasp', 'grasp')
        self.assertEqual(reason, 'contact_gripper')

    def test_proven_transit_skips(self):
        cfg = Config(contact_gating=True, gate_stage_change_on_contact_only=True,
                     confidence_cap=6.0, max_interval=20, min_samples=2, audit_probability=0.0)
        reason, gap, _ = self._reason(cfg, gripper_prop(1.0, 1.0, 1.0), stats(40, 40), 'carry', 'carry')
        self.assertIsNone(reason)
        self.assertGreater(gap, 1)

    def test_proven_transit_stage_change_skips(self):
        cfg = Config(contact_gating=True, gate_stage_change_on_contact_only=True,
                     confidence_cap=6.0, max_interval=20, min_samples=2, audit_probability=0.0)
        # stage changed into a proven transit stage, no contact -> v7 lets it skip
        reason, _, _ = self._reason(cfg, gripper_prop(1.0, 1.0, 1.0), stats(40, 40), 'lift', 'carry')
        self.assertIsNone(reason)

    def test_unproven_stage_change_still_reviews(self):
        cfg = Config(contact_gating=True, gate_stage_change_on_contact_only=True, audit_probability=0.0)
        # new stage not yet proven (gap==1) -> still review even in transit
        reason, _, _ = self._reason(cfg, gripper_prop(1.0, 1.0, 1.0), stats(2, 2), 'lift', 'carry')
        self.assertEqual(reason, 'stage_changed')


class DefaultOffMatchesBaseTests(unittest.TestCase):
    """contact_gating=False (the default) must give the identical decision as the C2 base scheduler."""

    class BaseStub:
        def __init__(self, st):
            self._st = st
        def statistics(self, stage):
            return self._st
        def near_error(self, stage, features, radius):
            return False

    def _pair(self, prev, stage, s, cfg_kw):
        obs7 = Observation('e', 30, (0.0,), {}, False, None)
        payload = dict(current_eef={'left': dict(gripper_opening_command=1.0, position=[0, 0, 0.9])},
                       student_eef_trajectory=[{'left': dict(gripper_opening_command=0.3, position=[0, 0, 0.9])}])
        prop7 = Proposal('r', obs7, payload, 15, (), False)
        m7 = Match(stage=stage, distance=0.01, reason='matched')
        r7 = Scheduler(Config(contact_gating=False, seed=1, **cfg_kw)).reason(
            StubMemory(s), m7, obs7, prop7, last_review_step=15, chunks_since_review=1,
            previous_stage=prev, recovery_active=False, force_review=False, stalled=False)

        sb = bmem.Statistics(successes=s.successes, failures=s.failures, total=s.total,
                             clean_streak=s.clean_streak, ever_failed=s.ever_failed)
        ob = btypes.Observation('e', 30, (0.0,), {}, False, None)
        pb = btypes.Proposal('r', ob, payload, 15, (), False)
        mb = bmem.Match(stage=stage, distance=0.01, reason='matched')
        rb = base.Scheduler(base.Config(seed=1, **cfg_kw)).reason(
            self.BaseStub(sb), mb, ob, pb, last_review_step=15, chunks_since_review=1,
            previous_stage=prev, recovery_active=False, force_review=False, stalled=False)
        return r7, rb

    def test_identical_across_cases(self):
        cases = [('approach', 'grasp', stats(40, 40)), ('grasp', 'grasp', stats(40, 40)),
                 ('grasp', 'grasp', stats(2, 2)), ('grasp', 'grasp', stats(9, 12)),
                 ('grasp', 'grasp', stats(20, 20, clean_streak=1, ever_failed=True))]
        for kw in (dict(audit_probability=0.0),
                   dict(audit_probability=0.0, confidence_cap=6.0, max_interval=20, min_samples=2),
                   dict(audit_probability=0.0, relax_known_stage_change=True, confidence_cap=6.0,
                        max_interval=20, min_samples=2)):
            for prev, stage, s in cases:
                r7, rb = self._pair(prev, stage, s, kw)
                self.assertEqual(r7, rb, f'v7 default-off diverged from base: prev={prev} stage={stage} kw={kw}')


class RichStub:
    def __init__(self, st, near=False, span_value=None):
        self._st, self._near, self._span = st, near, span_value
    def statistics(self, stage):
        return self._st
    def near_error(self, stage, features, radius):
        return self._near
    def span(self, stage):
        return self._span


def transit_prop(step, gripper=0.1):
    """A carrying/transit proposal (gripper steady-closed, high z) -> not a contact moment."""
    obs = Observation('e', step, (0.0,), dict(
        current_eef={'left': dict(gripper_opening_command=gripper, position=[0.0, 0.0, 1.2])},
        student_eef_trajectory=[{'left': dict(gripper_opening_command=gripper, position=[0.0, 0.0, 1.2])}]),
        False, None)
    return Proposal('r', obs, obs.payload, 15, (), False)


class PersistentErrorTests(unittest.TestCase):
    def _reason(self, cfg, near, st):
        prop = transit_prop(30)
        m = Match(stage='transport', distance=0.01, reason='matched')
        return Scheduler(cfg).reason(RichStub(st, near=near), m, prop.observation, prop,
            last_review_step=25, chunks_since_review=1, previous_stage='transport',
            recovery_active=False, force_review=False, stalled=False)

    def test_default_error_review_decays(self):  # base: high clean_streak turns near-error review off
        reason, _, _ = self._reason(Config(audit_probability=0.0), near=True, st=stats(40, 40, clean_streak=40))
        self.assertNotEqual(reason, 'error_memory')

    def test_persist_keeps_error_review(self):   # v7: near a past failure -> always review
        reason, _, _ = self._reason(Config(persist_error_review=True, audit_probability=0.0),
                                 near=True, st=stats(40, 40, clean_streak=40))
        self.assertEqual(reason, 'error_memory')


class CommitTests(unittest.TestCase):
    PROVEN = dict(commit_gating=True, commit_cap=12, max_unreviewed_steps=120,
                  confidence_cap=6.0, max_interval=20, min_samples=2, audit_probability=0.0)

    def _reason(self, cfg, span_value, step, near=False, stage='handover'):
        prop = transit_prop(step)
        m = Match(stage=stage, distance=0.01, reason='matched')
        return Scheduler(cfg).reason(RichStub(stats(40, 40), near=near, span_value=span_value), m,
            prop.observation, prop, last_review_step=0, chunks_since_review=1, previous_stage=stage,
            recovery_active=False, force_review=False, stalled=False)

    def test_commit_extends_step_budget(self):
        # span=13 -> budget min(12,13)*15+15=195; at step 130 (> base 120) still skip
        reason, _, _ = self._reason(Config(**self.PROVEN), span_value=13.0, step=130)
        self.assertIsNone(reason)

    def test_base_reviews_at_step_limit(self):
        cfg = Config(max_unreviewed_steps=120, confidence_cap=6.0, max_interval=20,
                     min_samples=2, audit_probability=0.0)  # commit off
        reason, _, _ = self._reason(cfg, span_value=13.0, step=130)
        self.assertEqual(reason, 'physical_step_limit')

    def test_commit_capped(self):
        # span=99 but cap=12 -> budget=195; step 210 (>195) -> review (cap bounds blind commit)
        reason, _, _ = self._reason(Config(**self.PROVEN), span_value=99.0, step=210)
        self.assertEqual(reason, 'physical_step_limit')

    def test_commit_off_near_error(self):
        # near a failure -> commit NOT applied, and the failure is reviewed
        cfg = Config(**self.PROVEN, persist_error_review=True)
        reason, _, _ = self._reason(cfg, span_value=13.0, step=130, near=True)
        self.assertEqual(reason, 'error_memory')


class MemorySpanTests(unittest.TestCase):
    def test_record_and_read_span_ewma(self):
        from fast_harness.memory import Memory
        m = Memory(':memory:', 'ns-v7')
        self.assertIsNone(m.span('approach'))
        m.record_span('approach', 8)
        self.assertEqual(m.span('approach'), 8.0)        # first sample = value
        m.record_span('approach', 8)
        self.assertAlmostEqual(m.span('approach'), 8.0)  # 8*0.7+8*0.3
        m.record_span('approach', 18)                    # recovery-inflated outlier
        self.assertLess(m.span('approach'), 12)          # EWMA stays near typical
        m.record_span('approach', 0)                     # invalid ignored
        self.assertLess(m.span('approach'), 12)
        m.close()


class MemoryDecayTests(unittest.TestCase):
    def _n_ok(self, decay, n):
        from fast_harness.memory import Memory
        m = Memory(':memory:', 'ns', decay=decay)
        for i in range(n):
            m.record(episode='e', step=i, kind='execution', stage='s',
                     features=(0.1, 0.2, 0.3), outcome='ok', evidence='', student=True)
        st = m.statistics('s')
        m.close()
        return st

    def test_pure_accumulation_decay_one(self):
        st = self._n_ok(1.0, 5)
        self.assertEqual(st.successes, 5.0)   # decay=1.0 -> raw lifetime count, never forgets (improves with experience)
        self.assertEqual(st.total, 5)

    def test_ewma_decay_below_one_forgets(self):
        st = self._n_ok(0.98, 5)
        self.assertLess(st.successes, 5.0)    # decay<1 -> EWMA, old successes fade
        self.assertEqual(st.total, 5)


if __name__ == '__main__':
    unittest.main()

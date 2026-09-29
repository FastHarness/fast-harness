import math
import random
from dataclasses import dataclass

from .memory import Match, Memory, Statistics
from .types import Observation, Proposal, Transition


@dataclass(frozen=True)
class Config:
    max_interval: int = 6
    min_samples: int = 4
    cooldown_successes: int = 3
    max_unreviewed_steps: int = 60
    match_threshold: float = 0.06
    ambiguity_margin: float = 0.01
    error_radius: float = 0.04
    audit_probability: float = 0.05
    stall_chunks: int = 3
    attempt_chunks: int = 8
    recovery_chunks: int = 24
    max_checkpoints: int = 64
    max_episode_chunks: int = 1000
    always_review: bool = False
    episode_number: int | None = None
    sim_attempt: int = 1
    confidence_cap: float = 1.0            # >1 lets the skip interval keep growing with accumulated
                                           # successes (dynamic, ever-rising skip rate); 1.0 = original
    relax_known_stage_change: bool = False  # allow skipping a stage transition when the NEW stage is
                                            # already proven reliable; False = original (always review)
    # --- contact gating (every field default-off => byte-for-byte the baseline behaviour) ---
    contact_gating: bool = False            # gate review on manipulation CONTACT, not just familiarity
    contact_low_z: float = 0.95             # EEF height (m) at/below which we're near the work surface
    contact_descent_m: float = 0.05         # EEF drop within a chunk that counts as a pre-contact descent
    gate_stage_change_on_contact_only: bool = False  # a stage change forces review only when it is a
                                            # contact moment or the new stage is unproven (gap<=1)
    commit_gating: bool = False             # a proven subtask runs its recorded chunk-span without
    commit_cap: int = 12                    #     a mid-subtask forced review (bounded by commit_cap)
    commit_min_confidence: float = 0.9      # Wilson lower bound a stage needs before commit extends it
    persist_error_review: bool = False      # keep reviewing near a past failure location until it is
                                            #     robustly re-solved (drop the coarse stage-streak decay)
    seed: int = 0

    def __post_init__(self):
        integers = (self.max_interval, self.min_samples, self.cooldown_successes,
                    self.max_unreviewed_steps, self.stall_chunks, self.attempt_chunks,
                    self.recovery_chunks, self.max_checkpoints, self.max_episode_chunks,
                    self.commit_cap, self.sim_attempt)
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError('Budgets and counts must be positive integers')
        if self.episode_number is not None and (type(self.episode_number) is not int or self.episode_number < 1):
            raise ValueError('Episode number must be a positive integer')
        if any(not math.isfinite(value) or value < 0 for value in (
                self.match_threshold, self.ambiguity_margin, self.error_radius)):
            raise ValueError('Matching thresholds must be finite and nonnegative')
        if not 0 <= self.audit_probability <= 1:
            raise ValueError('Audit probability must be in [0, 1]')
        if not math.isfinite(self.confidence_cap) or self.confidence_cap < 1:
            raise ValueError('confidence_cap must be finite and >= 1')
        if any(not math.isfinite(value) for value in (self.contact_low_z, self.contact_descent_m)):
            raise ValueError('Contact-gating thresholds must be finite')
        if self.contact_descent_m < 0:
            raise ValueError('contact_descent_m must be nonnegative')
        if not 0 <= self.commit_min_confidence <= 1:
            raise ValueError('commit_min_confidence must be in [0, 1]')


def lower_success_bound(stats: Statistics) -> float:
    n = stats.successes + stats.failures
    if n == 0:
        return 0.0
    p, z = stats.successes / n, 1.96
    numerator = p + z * z / (2 * n) - z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, numerator / (1 + z * z / n))


def interval(stats: Statistics, config: Config) -> int:
    if stats.total < config.min_samples:
        return 1
    if stats.ever_failed and stats.clean_streak < config.cooldown_successes:
        return 1
    # confidence_cap == 1.0 reproduces the original interval exactly (total >= min_samples here, so
    # the multiplier is 1.0). Above 1.0, the skip interval keeps growing as a stage accumulates more
    # confirmed successes across episodes -> a dynamic, rising skip rate that only an error resets.
    confidence = min(config.confidence_cap, stats.total / config.min_samples)
    return 1 + int((config.max_interval - 1) * lower_success_bound(stats) ** 2 * confidence)


def contact_imminent(proposal: Proposal, config: Config) -> str | None:
    """Is this chunk a manipulation CONTACT/alignment moment? Decided ONLY from the policy's own proposal
    (no extra teacher call). Biased to caution: when a signal is ambiguous, prefer flagging contact so
    the chunk gets reviewed rather than silently skipped.

    (a) gripper command crosses the 0.5 open/closed threshold within the planned chunk -> grasp / release
        / handover: the exact moments where an unreviewed skip drops the object.
    (b) an end-effector descends onto the work surface within the chunk -> pre-grasp approach / alignment.
    """
    payload = getattr(proposal, 'payload', None) or {}
    trajectory = payload.get('student_eef_trajectory') or []
    current = payload.get('current_eef') or {}
    for arm in ('left', 'right'):
        openings = []
        start = (current.get(arm) or {}).get('gripper_opening_command') if isinstance(current.get(arm), dict) else None
        if isinstance(start, (int, float)) and not isinstance(start, bool):
            openings.append(float(start))
        heights = []
        for waypoint in trajectory:
            arm_state = waypoint.get(arm) if isinstance(waypoint, dict) else None
            if not isinstance(arm_state, dict):
                continue
            value = arm_state.get('gripper_opening_command')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                openings.append(float(value))
            position = arm_state.get('position')
            if isinstance(position, (list, tuple)) and len(position) == 3 and \
                    isinstance(position[2], (int, float)) and not isinstance(position[2], bool):
                heights.append(float(position[2]))
        if any((a - 0.5) * (b - 0.5) < 0 for a, b in zip(openings, openings[1:])):
            return 'contact_gripper'
        if len(heights) >= 3 and (heights[0] - heights[-1]) >= config.contact_descent_m and \
                heights[-1] <= config.contact_low_z:
            return 'contact_approach'
    return None


class Scheduler:
    def __init__(self, config: Config):
        self.config = config
        self.random = random.Random(config.seed)

    def reason(self, memory: Memory, match: Match, observation: Observation,
               proposal: Proposal, *, last_review_step: int | None,
               chunks_since_review: int, previous_stage: str | None,
               recovery_active: bool, force_review: bool,
               stalled: bool) -> tuple[str | None, int, int]:
        cfg = self.config
        stats = memory.statistics(match.stage) if match.stage else None
        gap = interval(stats, cfg) if stats is not None else 1
        step_limit = cfg.max_unreviewed_steps
        # commit: a proven, not-near-error subtask may run its recorded chunk-span (bounded by
        # commit_cap) before the physical-step limit forces a review -> fewer reviews inside subtasks
        # the policy has mastered. Raises the step budget (the binding constraint); the skip interval (gap)
        # is left to the confidence logic.
        if cfg.commit_gating and stats is not None and gap > 1 \
                and lower_success_bound(stats) >= cfg.commit_min_confidence \
                and not memory.near_error(match.stage, observation.features, cfg.error_radius):
            span = memory.span(match.stage)
            horizon = min(cfg.commit_cap, max(1, round(span))) if span else min(cfg.commit_cap, gap)
            step_limit = max(cfg.max_unreviewed_steps, horizon * 15 + 15)
        if cfg.always_review:
            return 'baseline', 1, step_limit
        if last_review_step is None:
            return 'episode_start', gap, step_limit
        if recovery_active:
            return 'recovery', 1, step_limit
        if match.stage is None:
            return match.reason, 1, step_limit
        contact = contact_imminent(proposal, cfg) if cfg.contact_gating else None
        if contact:                                    # contact/alignment moment -> always review
            return contact, 1, step_limit
        if match.stage != previous_stage and not (gap > 1 and (
                cfg.relax_known_stage_change or cfg.gate_stage_change_on_contact_only)):
            return 'stage_changed', 1, step_limit
        if force_review:
            return 'previous_uncertainty', 1, step_limit
        if proposal.warnings or stalled:
            return 'monitor_alert', 1, step_limit
        # persistent failure memory: near a past failure -> keep reviewing until robustly re-solved
        # (persist_error_review drops the coarse per-stage clean_streak decay that could otherwise re-risk it).
        if memory.near_error(match.stage, observation.features, cfg.error_radius) and (
                cfg.persist_error_review or stats.clean_streak < cfg.cooldown_successes * 3):
            return 'error_memory', 1, step_limit
        if observation.step - last_review_step >= step_limit:
            return 'physical_step_limit', gap, step_limit
        if chunks_since_review >= gap:
            return 'scheduled', gap, step_limit
        if self.random.random() < cfg.audit_probability:
            return 'random_audit', gap, step_limit
        return None, gap, step_limit


class MotionMonitor:
    def __init__(self, limit: int):
        self.limit, self.still = limit, 0

    def observe(self, transition: Transition) -> None:
        before = transition.before.payload.get('current_state', [])
        after = transition.after.payload.get('current_state', [])
        stationary = bool(before and len(before) == len(after)) and all(
            abs(a - b) < 1e-5 for a, b in zip(before, after))
        self.still = self.still + 1 if stationary else 0

    @property
    def stalled(self) -> bool:
        return self.still >= self.limit

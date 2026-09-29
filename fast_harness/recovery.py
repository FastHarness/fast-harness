from dataclasses import dataclass, field

from .types import Checkpoint, Review


@dataclass
class Recovery:
    attempt_limit: int
    total_limit: int
    phase: str = 'normal'
    attempts_failed: int = 0
    attempt_steps: int = 0
    total_steps: int = 0
    target: Checkpoint | None = None
    excluded: set[str] = field(default_factory=set)
    error_cases: set[str] = field(default_factory=set)
    failure_step: int = 0
    failure_goal: str = 'Recover task progress and student intent'

    @property
    def active(self) -> bool:
        return self.phase not in ('normal', 'exhausted')

    def candidates(self, checkpoints: list[Checkpoint]) -> tuple[Checkpoint, ...]:
        return tuple(point for point in checkpoints
                     if point.id not in self.excluded and point.observation.step < self.failure_step)

    def context(self) -> dict:
        return dict(phase=self.phase, failed_attempts=self.attempts_failed,
                    attempt_chunks=self.attempt_steps, total_chunks=self.total_steps,
                    target_checkpoint_id=self.target.id if self.target else None,
                    goal=self.target.goal if self.target and self.phase.startswith('return_') else self.failure_goal,
                    remaining_attempt_chunks=self.attempt_limit - self.attempt_steps,
                    remaining_total_chunks=self.total_limit - self.total_steps)

    def start(self, step: int, goal: str = 'Recover task progress and student intent') -> None:
        self.failure_goal = goal
        self.phase, self.failure_step = 'local', step
        self.attempts_failed = self.attempt_steps = 0
        self.target = None
        self.excluded.clear()
        if self.total_steps >= self.total_limit:
            self.phase = 'exhausted'

    def executed(self) -> None:
        if self.active:
            self.attempt_steps += 1
            self.total_steps += 1

    def update(self, review: Review, checkpoints: list[Checkpoint]) -> bool:
        previous_phase = self.phase
        if self.phase == 'select':
            options = {point.id: point for point in self.candidates(checkpoints)}
            if review.selected_checkpoint_id is None:
                self.phase = 'exhausted'
            elif review.selected_checkpoint_id not in options:
                raise ValueError('Reviewer selected an ineligible checkpoint')
            else:
                self.target = options[review.selected_checkpoint_id]
                self.excluded.add(self.target.id)
                self.phase, self.attempt_steps = 'return_selected', 0
            return True
        if not self.active or not self.attempt_steps:
            return False
        status = review.recovery_status
        attempt_ended = status in ('succeeded', 'failed') or self.attempt_steps >= self.attempt_limit
        if status == 'succeeded' and review.last_outcome != 'ok':
            raise ValueError('A failed transition cannot complete a recovery goal')
        if status == 'succeeded':
            if self.phase.startswith('return_'):
                self.phase, self.attempts_failed = 'after_return', 0
                self.attempt_steps = 0
            else:
                self.phase, self.target = 'normal', None
                self.attempt_steps = self.attempts_failed = 0
        elif status == 'failed' or self.attempt_steps >= self.attempt_limit:
            self.attempt_steps = 0
            if self.phase.startswith('return_'):
                self.phase = 'select'
            else:
                self.attempts_failed += 1
                if self.attempts_failed >= 2:
                    options = self.candidates(checkpoints)
                    if self.phase == 'local' and options:
                        self.target = max(options, key=lambda point: point.observation.step)
                        self.excluded.add(self.target.id)
                        self.phase = 'return_latest'
                    else:
                        self.phase = 'select'
                    self.attempts_failed = 0
        if self.active and self.total_steps >= self.total_limit:
            self.phase = 'exhausted'
        if self.phase == 'select' and not self.candidates(checkpoints):
            self.phase = 'exhausted'
        return self.phase != previous_phase or attempt_ended

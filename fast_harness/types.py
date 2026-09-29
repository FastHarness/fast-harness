from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Observation:
    episode_id: str
    step: int
    features: tuple[float, ...]
    payload: dict[str, Any]
    terminal: bool = False
    native_success: bool | None = None


@dataclass(frozen=True)
class Proposal:
    request_id: str
    observation: Observation
    payload: dict[str, Any]
    max_steps: int = 15
    warnings: tuple[str, ...] = ()
    blocked: bool = False


@dataclass(frozen=True)
class Transition:
    before: Observation
    after: Observation
    request_id: str
    stage: str
    mode: str
    source: str
    steps: int


@dataclass(frozen=True)
class Checkpoint:
    id: str
    observation: Observation
    stage: str
    goal: str
    prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass(frozen=True)
class ReviewRequest:
    observation: Observation
    proposal: Proposal | None
    previous: Transition | None
    stage_hint: str | None
    known_stages: tuple[str, ...]
    errors: tuple[dict[str, Any], ...]
    checkpoints: tuple[Checkpoint, ...]
    recovery: dict[str, Any]
    unreviewed: tuple[Transition, ...]
    retry_feedback: dict[str, Any] | None = None


@dataclass(frozen=True)
class Review:
    stage: str
    last_outcome: str
    error_type: str
    evidence: str
    response: dict[str, Any] | None
    checkpoint: dict[str, Any] | None = None
    recovery_status: str = 'continue'
    selected_checkpoint_id: str | None = None
    usage: Usage = field(default_factory=Usage)


class Backend(Protocol):
    def start(self) -> Observation: ...
    def infer(self, observation: Observation) -> Proposal: ...
    def auto_response(self, proposal: Proposal, steps: int, stage: str) -> dict[str, Any]: ...
    def execute(self, proposal: Proposal, response: dict[str, Any], source: str) -> Observation: ...
    def stop(self, reason: str) -> None: ...


class Reviewer(Protocol):
    def review(self, request: ReviewRequest) -> Review: ...


class HarnessError(RuntimeError):
    pass


class HarnessControlError(HarnessError):
    def __init__(self, code):
        super().__init__('Harness control invariant or budget failed')
        self.code = code


class ReviewSchemaError(HarnessError):
    """Object-key mismatch; diagnostics contain only names from our schema."""

    def __init__(self, schema_path, missing_keys, extra_count):
        super().__init__('Reviewer output does not match required object fields')
        self.schema_path = tuple(schema_path)
        self.missing_keys = tuple(missing_keys)
        self.extra_count = extra_count


class ReviewConstraintError(HarnessError):
    """A fixed local rule identifier, never a remote message or field value."""

    def __init__(self, code, schema_path=()):
        super().__init__('Reviewer output violates a validation constraint')
        self.code = code
        self.schema_path = tuple(schema_path)


class ReviewTransportError(HarnessError):
    """Transient transport failure (timeout, connection, non-2xx) reaching the reviewer service.

    Distinct from schema/constraint errors, which have a separate bounded retry budget.
    A review request has no robot side effects, so callers may safely resend it after a wait."""


class ReviewTransportExhaustedError(ReviewTransportError):
    def __init__(self, attempts):
        super().__init__('Reviewer transport request budget exhausted without a service response')
        self.attempts = attempts


def error_diagnostics(error):
    # Never serialize arbitrary exception messages, response values or extra names.
    if type(error) is ReviewTransportExhaustedError:
        return {'failure_reason': 'reviewer_no_response',
                'transport_error': dict(code='review_transport_exhausted', attempts=error.attempts)}
    if type(error) is ReviewSchemaError:
        return {'schema_error': dict(schema_path=list(error.schema_path),
                                     missing_keys=list(error.missing_keys),
                                     extra_count=error.extra_count)}
    if type(error) is ReviewConstraintError:
        return {'validation_error': dict(code=error.code, schema_path=list(error.schema_path))}
    if type(error) is HarnessControlError:
        return {'control_error': dict(code=error.code)}
    return {}

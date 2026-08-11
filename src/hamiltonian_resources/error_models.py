"""Typed error semantics shared by planning, evaluation, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np


EstimateCategory: TypeAlias = Literal["analytical", "empirical", "proxy"]
ClaimCategory: TypeAlias = Literal["analytical", "derived"]
Certification: TypeAlias = Literal["rigorous", "nonrigorous"]
AssessmentOutcome: TypeAlias = Literal["certified", "not_met", "unavailable"]


def _require_finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class ErrorComponent:
    """One named numerical contribution to an estimate or claim."""

    name: str
    value: float
    quantity: str

    def __post_init__(self) -> None:
        _require_text(self.name, "component name")
        _require_text(self.quantity, "component quantity")
        object.__setattr__(self, "value", _require_finite_nonnegative(self.value, "value"))


@dataclass(frozen=True)
class ReferenceRecord:
    """Primary reference and the exact theorem or equations used."""

    citation: str
    locator: str
    url: str = ""

    def __post_init__(self) -> None:
        _require_text(self.citation, "citation")
        _require_text(self.locator, "reference locator")


@dataclass(frozen=True)
class AssumptionRecord:
    """One explicit hypothesis and whether it is known to hold."""

    description: str
    satisfied: bool | None = None

    def __post_init__(self) -> None:
        _require_text(self.description, "assumption description")


@dataclass(frozen=True)
class FallbackRecord:
    """Machine-readable record of a requested model being replaced."""

    requested_method: str
    used_method: str
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.requested_method, "requested method")
        _require_text(self.used_method, "used method")
        _require_text(self.reason, "fallback reason")


@dataclass(frozen=True)
class CalibrationMetadata:
    """Finite calibration domain for an empirical numerical diagnostic."""

    domain: tuple[float, float]
    sample_count: int
    sampling_rule: str
    tolerance: float | None = None

    def __post_init__(self) -> None:
        lower, upper = (float(value) for value in self.domain)
        if not np.isfinite((lower, upper)).all() or lower >= upper:
            raise ValueError("calibration domain must be finite and increasing")
        if self.sample_count < 1:
            raise ValueError("calibration sample_count must be positive")
        _require_text(self.sampling_rule, "calibration sampling rule")
        object.__setattr__(self, "domain", (lower, upper))
        if self.tolerance is not None:
            object.__setattr__(
                self,
                "tolerance",
                _require_finite_nonnegative(self.tolerance, "calibration tolerance"),
            )


@dataclass(frozen=True)
class InitialStateRecord:
    """Identity metadata for a state-dependent empirical observation."""

    source: Literal["computational-zero", "user-supplied"]
    num_qubits: int
    norm: float
    digest: str

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("initial-state num_qubits must be positive")
        if not np.isfinite(self.norm) or self.norm <= 0:
            raise ValueError("initial-state norm must be positive and finite")
        _require_text(self.digest, "initial-state digest")


@dataclass(frozen=True)
class PostselectionRecord:
    """Projector and normalization convention used by an observation."""

    projector: str
    renormalized: bool

    def __post_init__(self) -> None:
        _require_text(self.projector, "postselection projector")


@dataclass(frozen=True)
class StateObservationContext:
    """State and postselection context for state-dependent metrics."""

    initial_state: InitialStateRecord
    postselection: PostselectionRecord


ObservationContext: TypeAlias = StateObservationContext | CalibrationMetadata


@dataclass(frozen=True)
class SizingEstimate:
    """Estimate used to select an algorithm parameter or resource shape."""

    value: float
    method: str
    category: EstimateCategory
    certification: Certification
    quantity: str
    metric: str
    scope: str
    target: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_finite_nonnegative(self.value, "value"))
        object.__setattr__(self, "target", _require_finite_nonnegative(self.target, "target"))
        _require_text(self.method, "sizing method")
        _require_text(self.quantity, "sizing quantity")
        _require_text(self.metric, "sizing metric")
        _require_text(self.scope, "sizing scope")
        if self.category not in {"analytical", "empirical", "proxy"}:
            raise ValueError("unknown sizing category")
        if self.certification not in {"rigorous", "nonrigorous"}:
            raise ValueError("unknown sizing certification")


@dataclass(frozen=True)
class SuzukiSizingEstimate(SizingEstimate):
    repetitions: int
    order: int
    prefactor: float
    partition: str
    group_count: int
    calibration_id: str | None = None
    calibration_size_extrapolated: bool = False
    calibration_time_extrapolated: bool = False
    active_constraint: str | None = None


@dataclass(frozen=True)
class MPFSizingEstimate(SizingEstimate):
    segments: int
    term_count: int
    formal_order: int
    branch_count_policy: str
    branch_count_policy_extensiveness_g: float | None
    branch_count_policy_target_error: float
    schedule: str
    exponents: tuple[int, ...] | None
    coefficient_l1_norm: float | None
    coefficient_l1_norm_source: str = ""
    exponent_sum: int = 0
    exponent_sum_source: str = ""
    explicit_schedule_available: bool = True
    calibration_id: str | None = None
    calibration_size_extrapolated: bool = False
    calibration_time_extrapolated: bool = False
    active_constraint: str | None = None


@dataclass(frozen=True)
class QSVTSizingEstimate(SizingEstimate):
    truncation_order: int
    degree: int
    cosine_degree: int
    sine_degree: int
    cosine_first_omitted_degree: int
    sine_first_omitted_degree: int
    scale: float
    cosine_tail_bound: float
    sine_tail_bound: float


@dataclass(frozen=True)
class EstimateSupport:
    """Optional typed evidence attached to a sizing estimate."""

    components: tuple[ErrorComponent, ...] = ()
    references: tuple[ReferenceRecord, ...] = ()
    assumptions: tuple[AssumptionRecord, ...] = ()
    fallback: FallbackRecord | None = None


@dataclass(frozen=True)
class ErrorClaim:
    """Mathematical error statement about one explicit object and scope."""

    value: float
    category: ClaimCategory
    certification: Certification
    quantity: str
    metric: str
    scope: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_finite_nonnegative(self.value, "value"))
        _require_text(self.quantity, "claim quantity")
        _require_text(self.metric, "claim metric")
        _require_text(self.scope, "claim scope")
        if self.category not in {"analytical", "derived"}:
            raise ValueError("unknown claim category")
        if self.certification not in {"rigorous", "nonrigorous"}:
            raise ValueError("unknown claim certification")


@dataclass(frozen=True)
class SupportedClaim:
    """A small claim core plus typed evidence and assumptions."""

    claim: ErrorClaim
    method: str
    components: tuple[ErrorComponent, ...] = ()
    references: tuple[ReferenceRecord, ...] = ()
    assumptions: tuple[AssumptionRecord, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.method, "claim method")


@dataclass(frozen=True)
class MetricObservation:
    """Empirical value with a metric and complete observation context."""

    value: float
    quantity: str
    metric: str
    scope: str
    context: ObservationContext

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError("observation value must be finite")
        object.__setattr__(self, "value", value)
        _require_text(self.quantity, "observation quantity")
        _require_text(self.metric, "observation metric")
        _require_text(self.scope, "observation scope")


@dataclass(frozen=True)
class TargetAssessment:
    """Whether one rigorous, scope-matched claim certifies a target."""

    target: float
    scope: str
    outcome: AssessmentOutcome
    claim: ErrorClaim | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _require_finite_nonnegative(self.target, "target"))
        _require_text(self.scope, "assessment scope")
        if self.outcome not in {"certified", "not_met", "unavailable"}:
            raise ValueError("unknown assessment outcome")
        if self.outcome == "unavailable":
            if self.claim is not None:
                raise ValueError("unavailable assessments cannot carry a claim")
            return
        if self.claim is None or self.claim.certification != "rigorous":
            raise ValueError("available assessments require a rigorous claim")
        if self.claim.scope != self.scope:
            raise ValueError("assessment scope must match its claim")
        expected = "certified" if self.claim.value <= self.target else "not_met"
        if self.outcome != expected:
            raise ValueError("assessment outcome does not match its claim and target")

    @property
    def certified(self) -> bool:
        return self.outcome == "certified"


def assess_claim(claim: ErrorClaim | None, target: float, scope: str) -> TargetAssessment:
    """Assess a claim conservatively for one exact consumer scope."""
    if claim is None or claim.certification != "rigorous" or claim.scope != scope:
        return TargetAssessment(target, scope, "unavailable")
    outcome: AssessmentOutcome = "certified" if claim.value <= target else "not_met"
    return TargetAssessment(target, scope, outcome, claim)


def oaa_good_block_error_bound(local_error: float) -> float:
    """Bound one cubic-OAA good block from a local unitary target."""
    delta = _require_finite_nonnegative(local_error, "local error")
    return delta + 0.5 * delta * (2 + delta) * (1 + delta)


def repeated_block_encoding_error_bound(one_step_error: float, repetitions: int) -> float | None:
    """Apply GSLW Corollary 55 to repeated same-ancilla unitary encodings.

    The theorem-backed multi-use claim is intentionally unavailable when its
    one-step error is above one.  A single use needs no multiplication lemma.
    """
    eta = _require_finite_nonnegative(one_step_error, "one-step error")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if repetitions == 1:
        return min(2.0, eta)
    if eta > 1:
        return None
    return min(2.0, 4 * repetitions**2 * eta)


def good_subspace_leakage_bound(one_step_error: float) -> float | None:
    """Bound leakage from a good block that approximates a unitary."""
    eta = _require_finite_nonnegative(one_step_error, "one-step error")
    if eta > 1:
        return None
    return float(np.sqrt(max(0.0, 2 * eta - eta**2)))


@dataclass(frozen=True)
class ErrorAnalysis:
    """Structured sizing, claims, observations, and scoped target status."""

    sizing_estimate: SizingEstimate
    sizing_support: EstimateSupport
    claims: tuple[SupportedClaim, ...]
    observations: tuple[MetricObservation, ...]
    selection_succeeded: bool
    ideal_algorithm_target: TargetAssessment
    implemented_circuit_target: TargetAssessment

    @property
    def parameter_selection_succeeded(self) -> bool:
        return self.selection_succeeded

    @property
    def ideal_algorithm_target_certified(self) -> bool:
        return self.ideal_algorithm_target.certified

    @property
    def implemented_circuit_target_certified(self) -> bool:
        return self.implemented_circuit_target.certified

    def claim_for_scope(self, scope: str) -> SupportedClaim | None:
        return next((entry for entry in self.claims if entry.claim.scope == scope), None)

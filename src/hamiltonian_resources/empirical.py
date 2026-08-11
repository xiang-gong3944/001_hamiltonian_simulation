"""Reviewed operator-norm calibrations for empirical parameter sizing."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from numbers import Integral
from typing import Literal, Mapping, TypeAlias

import numpy as np

from .hamiltonians import HamiltonianModelMetadata, ModelParameter, PauliHamiltonian


EmpiricalMethod: TypeAlias = Literal["trotter", "multiproduct"]
_ERROR_METHOD = "empirical-operator-norm"


class UnsupportedEmpiricalCalibrationError(ValueError):
    """Raised when an empirical policy has no exact reviewed calibration."""


@dataclass(frozen=True)
class EmpiricalCalibrationKey:
    method: EmpiricalMethod
    formal_order: int
    model: str
    parameters: tuple[tuple[str, ModelParameter], ...]
    geometry: str
    boundary_condition: str
    partition: str | None = None
    schedule: str | None = None
    formula: str = ""

    def __post_init__(self) -> None:
        if self.method not in {"trotter", "multiproduct"}:
            raise ValueError("unknown empirical method")
        if isinstance(self.formal_order, bool) or not isinstance(self.formal_order, Integral):
            raise TypeError("formal_order must be an integer")
        if self.formal_order < 1:
            raise ValueError("formal_order must be positive")
        if not self.model or not self.geometry or not self.formula:
            raise ValueError("calibration key text fields must be nonempty")
        metadata = HamiltonianModelMetadata(
            self.model,
            self.parameters,
            self.geometry,
            self.boundary_condition,
        )
        object.__setattr__(self, "parameters", metadata.parameters)
        if self.method == "trotter" and (self.partition is None or self.schedule is not None):
            raise ValueError("Trotter calibration keys require only a partition")
        if self.method == "multiproduct" and (
            self.schedule is None or self.partition is not None
        ):
            raise ValueError("MPF calibration keys require only a schedule")

    @classmethod
    def for_hamiltonian(
        cls,
        hamiltonian: PauliHamiltonian,
        *,
        method: EmpiricalMethod,
        formal_order: int,
        partition: str | None = None,
        schedule: str | None = None,
        formula: str,
    ) -> "EmpiricalCalibrationKey":
        metadata = hamiltonian.model_metadata
        if metadata is None:
            raise UnsupportedEmpiricalCalibrationError(
                "empirical sizing requires structured Hamiltonian model metadata"
            )
        return cls(
            method=method,
            formal_order=formal_order,
            model=metadata.model,
            parameters=metadata.parameters,
            geometry=metadata.geometry,
            boundary_condition=metadata.boundary_condition,
            partition=partition,
            schedule=schedule,
            formula=formula,
        )


@dataclass(frozen=True)
class AffineSizeCoefficient:
    """One-dimensional bulk-plus-boundary coefficient ``B(N)=a*N+b``."""

    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.slope, self.intercept)).all() or self.slope <= 0:
            raise ValueError("affine coefficient requires a positive finite slope")

    def at(self, system_size: int) -> float:
        if isinstance(system_size, bool) or not isinstance(system_size, Integral):
            raise TypeError("system_size must be an integer")
        if system_size < 1:
            raise ValueError("system_size must be positive")
        value = self.slope * int(system_size) + self.intercept
        if not np.isfinite(value) or value <= 0:
            raise UnsupportedEmpiricalCalibrationError(
                "the calibrated affine coefficient is nonpositive at this system size"
            )
        return float(value)

    @property
    def model_name(self) -> str:
        return "affine"

    @property
    def parameters(self) -> tuple[tuple[str, float], ...]:
        return (("slope", self.slope), ("intercept", self.intercept))


@dataclass(frozen=True)
class PowerSizeCoefficient:
    """Positive monotone coefficient ``B_2J(N)=A*N**p``."""

    amplitude: float
    exponent: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite((self.amplitude, self.exponent)).all()
            or self.amplitude <= 0
            or self.exponent < 0
        ):
            raise ValueError("power coefficient requires A > 0 and p >= 0")

    def at(self, system_size: int) -> float:
        _validate_coefficient_system_size(system_size)
        value = self.amplitude * int(system_size) ** self.exponent
        if not np.isfinite(value) or value <= 0:
            raise UnsupportedEmpiricalCalibrationError(
                "the calibrated power coefficient is invalid at this system size"
            )
        return float(value)

    @property
    def model_name(self) -> str:
        return "power"

    @property
    def parameters(self) -> tuple[tuple[str, float], ...]:
        return (("amplitude", self.amplitude), ("exponent", self.exponent))


@dataclass(frozen=True)
class PowerPlusOffsetSizeCoefficient:
    """Monotone coefficient ``B_2J(N)=A*N**p+C``."""

    amplitude: float
    exponent: float
    offset: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite((self.amplitude, self.exponent, self.offset)).all()
            or self.amplitude <= 0
            or self.exponent < 0
        ):
            raise ValueError("power-plus-offset coefficient requires A > 0 and p >= 0")

    def at(self, system_size: int) -> float:
        _validate_coefficient_system_size(system_size)
        value = self.amplitude * int(system_size) ** self.exponent + self.offset
        if not np.isfinite(value) or value <= 0:
            raise UnsupportedEmpiricalCalibrationError(
                "the calibrated power-plus-offset coefficient is nonpositive "
                "at this system size"
            )
        return float(value)

    @property
    def model_name(self) -> str:
        return "power-plus-offset"

    @property
    def parameters(self) -> tuple[tuple[str, float], ...]:
        return (
            ("amplitude", self.amplitude),
            ("exponent", self.exponent),
            ("offset", self.offset),
        )


SizeCoefficient: TypeAlias = (
    AffineSizeCoefficient | PowerSizeCoefficient | PowerPlusOffsetSizeCoefficient
)


def _validate_coefficient_system_size(system_size: int) -> None:
    if isinstance(system_size, bool) or not isinstance(system_size, Integral):
        raise TypeError("system_size must be an integer")
    if system_size < 1:
        raise ValueError("system_size must be positive")


@dataclass(frozen=True)
class EmpiricalCalibrationRecord:
    calibration_id: str
    key: EmpiricalCalibrationKey
    coefficient: SizeCoefficient
    size_range: tuple[int, int]
    time_range: tuple[float, float]
    max_step_size: float
    sample_sizes: tuple[int, ...]
    sample_times: tuple[float, ...]
    error_metric: str
    source: str
    source_digest: str
    reference: str
    review_status: Literal["reviewed", "candidate", "rejected"]
    fit_diagnostics: tuple[tuple[str, float], ...] = ()
    schema_version: Literal["1.0", "2.0"] = "1.0"
    reviewed_size_max: int | None = None
    stability_diagnostics: tuple[tuple[str, float], ...] = ()
    precision_backend: str | None = None
    precision_digits: int | None = None
    external_validation_sizes: tuple[int, ...] = ()
    external_validation_status: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.calibration_id
            or not self.error_metric
            or not self.source
            or not self.reference
        ):
            raise ValueError("calibration provenance fields must be nonempty")
        minimum, maximum = self.size_range
        if minimum < 1 or minimum > maximum:
            raise ValueError("invalid calibration size range")
        time_minimum, time_maximum = self.time_range
        if not np.isfinite((time_minimum, time_maximum)).all() or not (
            0 < time_minimum <= time_maximum
        ):
            raise ValueError("invalid calibration time range")
        if not np.isfinite(self.max_step_size) or self.max_step_size <= 0:
            raise ValueError("max_step_size must be positive and finite")
        if (
            not self.sample_sizes
            or tuple(sorted(set(self.sample_sizes))) != self.sample_sizes
            or self.sample_sizes[0] < minimum
            or self.sample_sizes[-1] > maximum
        ):
            raise ValueError("sample_sizes must be sorted, unique, and within size_range")
        if (
            not self.sample_times
            or tuple(sorted(set(self.sample_times))) != self.sample_times
            or self.sample_times[0] < time_minimum
            or self.sample_times[-1] > time_maximum
        ):
            raise ValueError("sample_times must be sorted, unique, and within time_range")
        if len(self.source_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_digest
        ):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        if self.review_status not in {"reviewed", "candidate", "rejected"}:
            raise ValueError("unknown calibration review status")
        if self.schema_version not in {"1.0", "2.0"}:
            raise ValueError("unsupported record schema version")
        if self.schema_version == "2.0" and self.reviewed_size_max is None:
            raise ValueError("v2 calibration records require reviewed_size_max")
        if self.reviewed_size_max is not None and self.reviewed_size_max < maximum:
            raise ValueError("reviewed_size_max cannot be below the observed size range")
        diagnostic_names = [name for name, _ in self.fit_diagnostics]
        if len(diagnostic_names) != len(set(diagnostic_names)):
            raise ValueError("fit diagnostic names must be unique")
        if any(not name or not np.isfinite(value) for name, value in self.fit_diagnostics):
            raise ValueError("fit diagnostics must be named and finite")
        stability_names = [name for name, _ in self.stability_diagnostics]
        if len(stability_names) != len(set(stability_names)) or any(
            not name or not np.isfinite(value)
            for name, value in self.stability_diagnostics
        ):
            raise ValueError("stability diagnostics must be uniquely named and finite")
        if self.precision_digits is not None and self.precision_digits < 2:
            raise ValueError("precision_digits must be at least two")
        if (self.precision_backend is None) != (self.precision_digits is None):
            raise ValueError("precision backend and digits must be supplied together")
        if tuple(sorted(set(self.external_validation_sizes))) != self.external_validation_sizes:
            raise ValueError("external validation sizes must be sorted and unique")
        if any(value not in self.sample_sizes for value in self.external_validation_sizes):
            raise ValueError("external validation sizes must be included in sample_sizes")
        if self.external_validation_status not in {
            None,
            "passed",
            "not-required",
            "infeasible-review-exception",
        }:
            raise ValueError("unknown external validation status")
        if self.schema_version == "2.0" and self.external_validation_status is None:
            object.__setattr__(self, "external_validation_status", "not-required")
        self.coefficient.at(minimum)
        self.coefficient.at(maximum)
        if self.reviewed_size_max is not None:
            self.coefficient.at(self.reviewed_size_max)


@dataclass(frozen=True)
class EmpiricalErrorEstimate:
    error: float
    prefactor: float
    time: float
    segments: int
    formal_order: int
    calibration: EmpiricalCalibrationRecord
    size_extrapolated: bool
    time_extrapolated: bool
    formula_segments: int
    asymptotic_guard_segments: int
    active_constraint: Literal["formula", "asymptotic-domain", "both"]
    method: str = _ERROR_METHOD
    rigorous: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite((self.error, self.prefactor, self.time)).all():
            raise ValueError("empirical estimate values must be finite")
        if self.error < 0 or self.prefactor <= 0 or self.time <= 0:
            raise ValueError("invalid empirical estimate values")
        if self.segments < 1 or self.formula_segments < 1:
            raise ValueError("empirical segment counts must be positive")
        if self.asymptotic_guard_segments < 1:
            raise ValueError("asymptotic guard must be positive")
        if self.formal_order != self.calibration.key.formal_order:
            raise ValueError("estimate order does not match its calibration")
        if self.method != _ERROR_METHOD or self.rigorous:
            raise ValueError("empirical estimates are always nonrigorous")

    @property
    def calibration_id(self) -> str:
        return self.calibration.calibration_id


class EmpiricalCalibrationRegistry:
    """Exact-match registry containing only explicitly reviewed rows."""

    def __init__(self, records: tuple[EmpiricalCalibrationRecord, ...]) -> None:
        all_keys: set[EmpiricalCalibrationKey] = set()
        identifiers: set[str] = set()
        for record in records:
            if record.key in all_keys:
                raise ValueError(f"duplicate empirical calibration key: {record.key!r}")
            if record.calibration_id in identifiers:
                raise ValueError(
                    f"duplicate empirical calibration id: {record.calibration_id!r}"
                )
            all_keys.add(record.key)
            identifiers.add(record.calibration_id)
        reviewed = tuple(record for record in records if record.review_status == "reviewed")
        by_key: dict[EmpiricalCalibrationKey, EmpiricalCalibrationRecord] = {}
        for record in reviewed:
            by_key[record.key] = record
        self._records = reviewed
        self._by_key = by_key

    @property
    def records(self) -> tuple[EmpiricalCalibrationRecord, ...]:
        return self._records

    def lookup(self, key: EmpiricalCalibrationKey) -> EmpiricalCalibrationRecord:
        try:
            return self._by_key[key]
        except KeyError as error:
            details = (
                f"method={key.method}, formal_order={key.formal_order}, "
                f"model={key.model}, parameters={dict(key.parameters)!r}, "
                f"boundary={key.boundary_condition}, partition={key.partition!r}, "
                f"schedule={key.schedule!r}, formula={key.formula!r}"
            )
            raise UnsupportedEmpiricalCalibrationError(
                f"no reviewed empirical operator-norm calibration for {details}"
            ) from error

    @classmethod
    def from_json_data(cls, raw: Mapping[str, object]) -> "EmpiricalCalibrationRegistry":
        schema_version = raw.get("schema_version")
        if schema_version not in {"1.0", "2.0"}:
            raise ValueError("unsupported empirical calibration schema")
        rows = raw.get("calibrations")
        if not isinstance(rows, list):
            raise ValueError("calibrations must be an array")
        return cls(
            tuple(
                _record_from_dict(row, schema_version=str(schema_version))
                for row in rows
            )
        )


def _coefficient_from_dict(
    raw: Mapping[str, object],
    *,
    schema_version: str,
) -> SizeCoefficient:
    if schema_version == "1.0":
        return AffineSizeCoefficient(float(raw["slope"]), float(raw["intercept"]))
    model = raw.get("model")
    parameters = raw.get("parameters")
    if not isinstance(parameters, Mapping):
        raise TypeError("v2 coefficient parameters must be an object")
    if model == "affine":
        return AffineSizeCoefficient(
            float(parameters["slope"]),
            float(parameters["intercept"]),
        )
    if model == "power":
        return PowerSizeCoefficient(
            float(parameters["amplitude"]),
            float(parameters["exponent"]),
        )
    if model == "power-plus-offset":
        return PowerPlusOffsetSizeCoefficient(
            float(parameters["amplitude"]),
            float(parameters["exponent"]),
            float(parameters["offset"]),
        )
    raise ValueError(f"unsupported v2 coefficient model: {model!r}")


def _record_from_dict(
    raw: object,
    *,
    schema_version: str,
) -> EmpiricalCalibrationRecord:
    if not isinstance(raw, Mapping):
        raise TypeError("each calibration must be an object")
    key_raw = raw["key"]
    coefficient_raw = raw["coefficient"]
    if not isinstance(key_raw, Mapping) or not isinstance(coefficient_raw, Mapping):
        raise TypeError("calibration key and coefficient must be objects")
    parameters = key_raw.get("parameters")
    if not isinstance(parameters, Mapping):
        raise TypeError("calibration parameters must be an object")
    diagnostics = raw.get("fit_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise TypeError("fit_diagnostics must be an object")
    key = EmpiricalCalibrationKey(
        method=str(key_raw["method"]),  # type: ignore[arg-type]
        formal_order=int(key_raw["formal_order"]),
        model=str(key_raw["model"]),
        parameters=tuple(sorted(parameters.items())),  # type: ignore[arg-type]
        geometry=str(key_raw["geometry"]),
        boundary_condition=str(key_raw["boundary_condition"]),
        partition=(str(key_raw["partition"]) if key_raw.get("partition") is not None else None),
        schedule=(str(key_raw["schedule"]) if key_raw.get("schedule") is not None else None),
        formula=str(key_raw["formula"]),
    )
    size_range = tuple(int(value) for value in raw["size_range"])  # type: ignore[arg-type]
    time_range = tuple(float(value) for value in raw["time_range"])  # type: ignore[arg-type]
    if len(size_range) != 2 or len(time_range) != 2:
        raise ValueError("calibration ranges must contain two endpoints")
    return EmpiricalCalibrationRecord(
        calibration_id=str(raw["calibration_id"]),
        key=key,
        coefficient=_coefficient_from_dict(
            coefficient_raw,
            schema_version=schema_version,
        ),
        size_range=(size_range[0], size_range[1]),
        time_range=(time_range[0], time_range[1]),
        max_step_size=float(raw["max_step_size"]),
        sample_sizes=tuple(int(value) for value in raw["sample_sizes"]),  # type: ignore[arg-type]
        sample_times=tuple(float(value) for value in raw["sample_times"]),  # type: ignore[arg-type]
        error_metric=str(raw["error_metric"]),
        source=str(raw["source"]),
        source_digest=str(raw["source_digest"]),
        reference=str(raw["reference"]),
        review_status=str(raw["review_status"]),  # type: ignore[arg-type]
        fit_diagnostics=tuple(
            sorted((str(name), float(value)) for name, value in diagnostics.items())
        ),
        schema_version=schema_version,  # type: ignore[arg-type]
        reviewed_size_max=(
            int(raw["reviewed_size_max"])
            if raw.get("reviewed_size_max") is not None
            else None
        ),
        stability_diagnostics=tuple(
            sorted(
                (str(name), float(value))
                for name, value in _mapping_or_empty(
                    raw.get("stability_diagnostics"),
                    name="stability_diagnostics",
                ).items()
            )
        ),
        precision_backend=(
            str(raw["precision_backend"])
            if raw.get("precision_backend") is not None
            else None
        ),
        precision_digits=(
            int(raw["precision_digits"])
            if raw.get("precision_digits") is not None
            else None
        ),
        external_validation_sizes=tuple(
            int(value) for value in raw.get("external_validation_sizes", ())  # type: ignore[arg-type]
        ),
        external_validation_status=(
            str(raw["external_validation_status"])
            if raw.get("external_validation_status") is not None
            else None
        ),
    )


def _mapping_or_empty(raw: object, *, name: str) -> Mapping[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be an object")
    return raw


def canonical_json_digest(raw: object) -> str:
    """Return a platform-independent SHA-256 digest of parsed JSON data."""
    payload = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def default_empirical_calibrations() -> EmpiricalCalibrationRegistry:
    resource = files("hamiltonian_resources").joinpath(
        "data/empirical_calibrations_v1.json"
    )
    return EmpiricalCalibrationRegistry.from_json_data(
        json.loads(resource.read_text(encoding="utf-8"))
    )


def evaluate_empirical_error(
    record: EmpiricalCalibrationRecord,
    system_size: int,
    time: float,
    segments: int,
) -> float:
    if isinstance(segments, bool) or not isinstance(segments, Integral):
        raise TypeError("segments must be an integer")
    if segments < 1:
        raise ValueError("segments must be positive")
    if not np.isfinite(time) or time <= 0:
        raise ValueError("time must be positive and finite")
    prefactor = record.coefficient.at(system_size)
    order = record.key.formal_order
    log_error = (
        math.log(prefactor)
        + (order + 1) * math.log(float(time))
        - order * math.log(int(segments))
    )
    if log_error > math.log(np.finfo(float).max):
        raise OverflowError("empirical error exceeds the floating-point range")
    return float(math.exp(log_error))


def select_empirical_segments(
    record: EmpiricalCalibrationRecord,
    system_size: int,
    time: float,
    target_error: float,
) -> EmpiricalErrorEstimate:
    if not np.isfinite(target_error) or not 0 < target_error <= 1:
        raise ValueError("target_error must lie in (0, 1]")
    if not np.isfinite(time) or time <= 0:
        raise ValueError("time must be positive and finite")
    size_minimum, size_maximum = record.size_range
    if system_size < size_minimum:
        raise UnsupportedEmpiricalCalibrationError(
            f"calibration {record.calibration_id!r} supports N >= {size_minimum}; "
            f"received N={system_size}"
        )
    if (
        record.reviewed_size_max is not None
        and system_size > record.reviewed_size_max
    ):
        raise UnsupportedEmpiricalCalibrationError(
            f"calibration {record.calibration_id!r} was reviewed only through "
            f"N={record.reviewed_size_max}; received N={system_size}"
        )
    prefactor = record.coefficient.at(system_size)
    order = record.key.formal_order
    log_segments = (
        math.log(prefactor)
        + (order + 1) * math.log(float(time))
        - math.log(float(target_error))
    ) / order
    if log_segments > math.log(np.finfo(float).max):
        raise OverflowError("required empirical segment count exceeds float range")
    formula_segments = max(1, math.ceil(math.exp(log_segments)))
    guard_segments = max(1, math.ceil(float(time) / record.max_step_size))
    segments = max(formula_segments, guard_segments)
    if formula_segments == guard_segments:
        active_constraint = "both"
    elif segments == formula_segments:
        active_constraint = "formula"
    else:
        active_constraint = "asymptotic-domain"
    error = evaluate_empirical_error(record, system_size, time, segments)
    while error > target_error:
        segments += 1
        error = evaluate_empirical_error(record, system_size, time, segments)
    time_minimum, time_maximum = record.time_range
    return EmpiricalErrorEstimate(
        error=error,
        prefactor=prefactor,
        time=float(time),
        segments=segments,
        formal_order=order,
        calibration=record,
        size_extrapolated=system_size > size_maximum,
        time_extrapolated=not time_minimum <= time <= time_maximum,
        formula_segments=formula_segments,
        asymptotic_guard_segments=guard_segments,
        active_constraint=active_constraint,
    )

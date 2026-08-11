"""Sharded Cartesian resource grids built on the single-point evaluation facade."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd

from ._commutator_execution import CommutatorExecution, CommutatorProgressCallback
from ._result_records import evaluation_report_metadata, software_metadata
from .benchmark_suite import BENCHMARK_COLUMNS, SCHEMA_VERSION, HamiltonianSpec, TimeScaling
from .empirical import UnsupportedEmpiricalCalibrationError
from .evaluation import estimate_resources
from .method_specs import MethodSpec, MultiproductMethod, QSVTMethod, TrotterMethod
from .multiproduct import resolve_mpf_branch_count
from .trotter import resolve_trotter_structure


ResourceGridPreset: TypeAlias = Literal["sanity-low", "sanity-high", "full"]
ResourceGridStatus: TypeAlias = Literal["ok", "missing_empirical", "error"]
RESOURCE_GRID_SCHEMA_VERSION = "1.0"
RESOURCE_GRID_EXTRA_COLUMNS = (
    "grid_schema_version",
    "shard_id",
    "log10_target_error",
    "estimator_variant",
    "trotter_reps",
    "mpf_branch_count",
    "mpf_segments",
)
RESOURCE_GRID_COLUMNS = RESOURCE_GRID_EXTRA_COLUMNS + BENCHMARK_COLUMNS


def _normalize_models(values: Sequence[HamiltonianSpec]) -> list[HamiltonianSpec]:
    result = list(values)
    if not result:
        raise ValueError("models must not be empty")
    for model in result:
        if not isinstance(model, HamiltonianSpec):
            raise TypeError("models must contain HamiltonianSpec values")
        model.validate()
    names = [model.model for model in result]
    if len(names) != len(set(names)):
        raise ValueError("models must have unique model names")
    return result


def _normalize_sizes(values: Sequence[int]) -> list[int]:
    result = list(values)
    if not result:
        raise ValueError("system_sizes must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1
        for value in result
    ):
        raise ValueError("system_sizes must contain positive integers")
    normalized = [int(value) for value in result]
    if len(normalized) != len(set(normalized)):
        raise ValueError("system_sizes must not contain duplicates")
    return normalized


def _normalize_log_errors(values: Sequence[float]) -> list[float]:
    result = list(values)
    if not result:
        raise ValueError("log10_target_errors must not be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or float(value) >= 0
        for value in result
    ):
        raise ValueError("log10_target_errors must contain finite values below zero")
    normalized = [float(value) for value in result]
    if len(normalized) != len(set(normalized)):
        raise ValueError("log10_target_errors must not contain duplicates")
    return normalized


@dataclass
class ResourceGridConfig:
    """Concrete, notebook-friendly configuration for a Cartesian resource grid."""

    models: Sequence[HamiltonianSpec]
    system_sizes: Sequence[int]
    log10_target_errors: Sequence[float]
    methods: Sequence[MethodSpec]
    time: TimeScaling = field(default_factory=TimeScaling)
    synthesis_error_fraction: float = 0.1
    trotter_partition: Literal["auto", "individual", "commuting"] = "auto"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.models = _normalize_models(self.models)
        self.system_sizes = _normalize_sizes(self.system_sizes)
        self.log10_target_errors = _normalize_log_errors(self.log10_target_errors)
        if not isinstance(self.time, TimeScaling):
            raise TypeError("time must be a TimeScaling")
        self.time.validate()
        if (
            isinstance(self.synthesis_error_fraction, bool)
            or not isinstance(self.synthesis_error_fraction, Real)
            or not np.isfinite(self.synthesis_error_fraction)
            or not 0 < float(self.synthesis_error_fraction) < 1
        ):
            raise ValueError("synthesis_error_fraction must lie in (0, 1)")
        self.synthesis_error_fraction = float(self.synthesis_error_fraction)
        if self.trotter_partition not in {"auto", "individual", "commuting"}:
            raise ValueError("trotter_partition must be 'auto', 'individual', or 'commuting'")
        self.methods = list(self.methods)
        if not self.methods:
            raise ValueError("methods must not be empty")
        for method in self.methods:
            if not isinstance(method, (TrotterMethod, MultiproductMethod, QSVTMethod)):
                raise TypeError("methods must contain resource method specifications")
            method.validate()
        identifiers = [method.method_id for method in self.methods]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("methods must have unique method IDs")

    @property
    def target_errors(self) -> list[float]:
        return [10.0**value for value in self.log10_target_errors]

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "models": [model.as_dict() for model in self.models],
            "system_sizes": list(self.system_sizes),
            "log10_target_errors": list(self.log10_target_errors),
            "methods": [method.as_dict() for method in self.methods],
            "time": self.time.as_dict(),
            "synthesis_error_fraction": self.synthesis_error_fraction,
            "trotter_partition": self.trotter_partition,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResourceGridShard:
    model: HamiltonianSpec
    system_qubits: int

    def __post_init__(self) -> None:
        self.model.validate()
        if (
            isinstance(self.system_qubits, bool)
            or not isinstance(self.system_qubits, Integral)
            or self.system_qubits < 1
        ):
            raise ValueError("system_qubits must be a positive integer")

    @property
    def shard_id(self) -> str:
        return f"{self.model.model}:N{self.system_qubits:03d}"

    @property
    def relative_path(self) -> str:
        return f"{self.model.model}/N{self.system_qubits:03d}.csv"


@dataclass(frozen=True)
class ResourceGridRunSummary:
    output_directory: Path
    manifest_path: Path
    validation_path: Path
    merged_path: Path
    completed_shards: int
    skipped_shards: int
    expected_missing_rows: int
    failed_rows: int


class NonRigorousCommutatorResultError(RuntimeError):
    """A series declared as commutator-rigorous returned a proxy result."""


def _preset_models() -> list[HamiltonianSpec]:
    return [
        HamiltonianSpec(
            "transverse_field_ising",
            {"coupling": 1.0, "field": 3.0, "periodic": False},
        ),
        HamiltonianSpec(
            "heisenberg_chain",
            {"coupling": 1.0, "field_z": 0.3},
        ),
    ]


def _preset_methods() -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    for order in (2, 4, 6):
        methods.extend(
            (
                TrotterMethod(order),
                TrotterMethod(order, "empirical-operator-norm"),
            )
        )
    methods.extend(
        (
            QSVTMethod(),
            MultiproductMethod(
                None,
                schedule="new",
                error_method="empirical-operator-norm",
                branch_count_policy="mizuta2026-theorem6",
            ),
            MultiproductMethod(
                None,
                schedule="new",
                error_method="mizuta2026-commutator-ideal-rigorous",
                branch_count_policy="mizuta2026-theorem6",
            ),
        )
    )
    return methods


def resource_grid_preset(name: ResourceGridPreset) -> ResourceGridConfig:
    if name == "sanity-low":
        sizes = (3, 4, 6)
        log_errors = (-1.0, -2.0)
    elif name == "sanity-high":
        sizes = (100, 120)
        log_errors = (-3.0, -4.0)
    elif name == "full":
        sizes = tuple(range(3, 121))
        log_errors = tuple(round(-1.0 - 0.1 * index, 10) for index in range(31))
    else:
        raise ValueError(f"unknown resource-grid preset: {name!r}")
    return ResourceGridConfig(
        models=_preset_models(),
        system_sizes=sizes,
        log10_target_errors=log_errors,
        methods=_preset_methods(),
        time=TimeScaling("proportional", 1.0),
        synthesis_error_fraction=0.1,
        trotter_partition="auto",
    )


def expand_resource_grid(config: ResourceGridConfig) -> tuple[ResourceGridShard, ...]:
    if not isinstance(config, ResourceGridConfig):
        raise TypeError("config must be a ResourceGridConfig")
    config.validate()
    return tuple(
        ResourceGridShard(model, system_qubits)
        for model in config.models
        for system_qubits in config.system_sizes
    )


def _method_from_dict(raw: Mapping[str, Any]) -> MethodSpec:
    family = raw.get("family")
    if family == "trotter":
        unknown = set(raw) - {"family", "order", "error_policy"}
        if unknown or "order" not in raw:
            raise ValueError("Trotter methods require family, order, and optional error_policy")
        return TrotterMethod(raw["order"], raw.get("error_policy", "analytical"))
    if family == "multiproduct":
        unknown = set(raw) - {
            "family",
            "term_count",
            "schedule",
            "error_method",
            "branch_count_policy",
        }
        policy = raw.get("branch_count_policy", "fixed")
        if unknown or (policy == "fixed" and "term_count" not in raw):
            raise ValueError("fixed multiproduct methods require term_count")
        return MultiproductMethod(
            raw.get("term_count"),
            raw.get("schedule", "new"),
            raw.get("error_method", "low2019-l1-ideal-rigorous"),
            policy,
        )
    if family == "qsvt" and set(raw) == {"family"}:
        return QSVTMethod()
    raise ValueError(f"unknown resource-grid method: {family!r}")


def resource_grid_config_from_dict(raw: Mapping[str, Any]) -> ResourceGridConfig:
    allowed = {
        "models",
        "system_sizes",
        "log10_target_errors",
        "methods",
        "time",
        "synthesis_error_fraction",
        "trotter_partition",
    }
    unknown = set(raw) - allowed
    required = {"models", "system_sizes", "log10_target_errors", "methods"}
    if unknown or not required.issubset(raw):
        raise ValueError("resource-grid configuration fields are invalid or incomplete")
    models_raw = raw["models"]
    methods_raw = raw["methods"]
    time_raw = raw.get("time", {})
    if not isinstance(models_raw, Sequence) or isinstance(models_raw, (str, bytes)):
        raise TypeError("models must be an array")
    if not isinstance(methods_raw, Sequence) or isinstance(methods_raw, (str, bytes)):
        raise TypeError("methods must be an array")
    if not isinstance(time_raw, Mapping):
        raise TypeError("time must be an object")
    models: list[HamiltonianSpec] = []
    for item in models_raw:
        if not isinstance(item, Mapping):
            raise TypeError("each model must be an object")
        if set(item) - {"model", "parameters"} or "model" not in item:
            raise ValueError("model objects require model and optional parameters")
        models.append(HamiltonianSpec(str(item["model"]), item.get("parameters", {})))
    methods = []
    for item in methods_raw:
        if not isinstance(item, Mapping):
            raise TypeError("each method must be an object")
        methods.append(_method_from_dict(item))
    return ResourceGridConfig(
        models=models,
        system_sizes=raw["system_sizes"],
        log10_target_errors=raw["log10_target_errors"],
        methods=methods,
        time=TimeScaling(time_raw.get("mode", "proportional"), time_raw.get("coefficient", 1.0)),
        synthesis_error_fraction=raw.get("synthesis_error_fraction", 0.1),
        trotter_partition=raw.get("trotter_partition", "auto"),
    )


def load_resource_grid_config(path: str | Path) -> ResourceGridConfig:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TypeError("resource-grid configuration must be an object")
    if "resource_grid" in raw:
        if set(raw) != {"resource_grid"} or not isinstance(raw["resource_grid"], Mapping):
            raise ValueError("wrapped configuration must contain only a resource_grid object")
        raw = raw["resource_grid"]
    return resource_grid_config_from_dict(raw)


def _estimator_variant(method: MethodSpec) -> str:
    if isinstance(method, TrotterMethod):
        return "empirical" if method.error_policy == "empirical-operator-norm" else "commutator"
    if isinstance(method, MultiproductMethod):
        if method.error_method == "empirical-operator-norm":
            return "empirical"
        return "commutator" if "commutator" in method.error_method else "analytical"
    return "jacobi-anger"


def _base_record(
    config: ResourceGridConfig,
    shard: ResourceGridShard,
    run_metadata: Mapping[str, Any],
    log10_target_error: float,
    target_error: float,
    evolution_time: float,
    method: MethodSpec,
) -> dict[str, Any]:
    record = {column: None for column in RESOURCE_GRID_COLUMNS}
    record.update(
        grid_schema_version=RESOURCE_GRID_SCHEMA_VERSION,
        schema_version=SCHEMA_VERSION,
        shard_id=shard.shard_id,
        run_id=run_metadata["run_id"],
        generated_at_utc=run_metadata["generated_at_utc"],
        config_digest=config.digest,
        sweep="resource-grid",
        hamiltonian_model=shard.model.model,
        model_parameters_json=json.dumps(
            dict(shard.model.parameters), sort_keys=True, separators=(",", ":")
        ),
        system_qubits=shard.system_qubits,
        evolution_time=evolution_time,
        time_scaling_mode=config.time.mode,
        time_scaling_coefficient=config.time.coefficient,
        log10_target_error=log10_target_error,
        target_error=target_error,
        method_id=method.method_id,
        method_family=method.family,
        method_label=method.label,
        estimator_variant=_estimator_variant(method),
        error_policy=(
            method.error_policy
            if isinstance(method, TrotterMethod)
            else method.error_method if isinstance(method, MultiproductMethod) else "jacobi-anger-rigorous"
        ),
        trotter_order=method.order if isinstance(method, TrotterMethod) else None,
        mpf_term_count=method.term_count if isinstance(method, MultiproductMethod) else None,
        mpf_branch_count_policy=(
            method.branch_count_policy if isinstance(method, MultiproductMethod) else None
        ),
        mpf_branch_count_policy_target_error=(
            target_error * (1 - config.synthesis_error_fraction)
            if isinstance(method, MultiproductMethod)
            else None
        ),
        algorithm_error_budget=target_error * (1 - config.synthesis_error_fraction),
        **run_metadata["software"],
    )
    return record


def _pre_resolve_parameters(record: dict[str, Any], hamiltonian, method: MethodSpec) -> None:
    if isinstance(method, TrotterMethod):
        structure = resolve_trotter_structure(hamiltonian, method.order, record["trotter_partition"] or "auto")
        record["trotter_partition"] = structure.partition
        record["trotter_group_count"] = len(structure.group_term_indices)
    elif isinstance(method, MultiproductMethod):
        selection = resolve_mpf_branch_count(
            hamiltonian,
            record["evolution_time"],
            record["algorithm_error_budget"],
            policy=method.branch_count_policy,
            term_count=method.term_count,
            schedule=method.schedule,
        )
        record["mpf_term_count"] = selection.term_count
        record["mpf_branch_count"] = selection.term_count
        record["mpf_formal_order"] = selection.formal_order
        record["mpf_branch_count_policy_extensiveness_g"] = selection.extensiveness_g


def evaluate_resource_grid_shard(
    config: ResourceGridConfig,
    shard: ResourceGridShard,
    *,
    run_metadata: Mapping[str, Any] | None = None,
    commutator_progress: CommutatorProgressCallback | None = None,
) -> pd.DataFrame:
    """Evaluate one model/N shard, building its Hamiltonian and cache exactly once."""
    config.validate()
    if shard not in expand_resource_grid(config):
        raise ValueError("shard is not part of the resource-grid configuration")
    if run_metadata is None:
        run_metadata = {
            "run_id": str(uuid.uuid4()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "software": software_metadata(),
        }
    required_metadata = {"run_id", "generated_at_utc", "software"}
    if not required_metadata.issubset(run_metadata):
        raise ValueError("run metadata is incomplete")
    evolution_time = config.time.at(shard.system_qubits)
    try:
        hamiltonian_or_error = shard.model.build(shard.system_qubits)
    except Exception as exc:
        hamiltonian_or_error = exc
    records: list[dict[str, Any]] = []
    with CommutatorExecution(1, commutator_progress) as execution:
        for log_error, target_error in zip(
            config.log10_target_errors, config.target_errors, strict=True
        ):
            for method in config.methods:
                record = _base_record(
                    config,
                    shard,
                    run_metadata,
                    log_error,
                    target_error,
                    evolution_time,
                    method,
                )
                try:
                    if isinstance(hamiltonian_or_error, Exception):
                        raise hamiltonian_or_error
                    hamiltonian = hamiltonian_or_error
                    record.update(
                        hamiltonian_name=hamiltonian.name,
                        hamiltonian_alpha=hamiltonian.alpha,
                        hamiltonian_term_count=hamiltonian.term_count,
                    )
                    if isinstance(method, TrotterMethod):
                        structure = resolve_trotter_structure(
                            hamiltonian, method.order, config.trotter_partition
                        )
                        record["trotter_partition"] = structure.partition
                        record["trotter_group_count"] = len(structure.group_term_indices)
                    else:
                        _pre_resolve_parameters(record, hamiltonian, method)
                    report = estimate_resources(
                        hamiltonian,
                        method,
                        evolution_time,
                        target_error,
                        synthesis_error_fraction=config.synthesis_error_fraction,
                        trotter_partition=config.trotter_partition,
                        workers=1,
                        _execution=execution,
                    )
                    record.update(evaluation_report_metadata(report))
                    selected = report.selected_parameters
                    record["trotter_reps"] = selected.get("trotter_reps")
                    record["mpf_branch_count"] = selected.get("mpf_branch_count")
                    record["mpf_segments"] = selected.get("mpf_segments")
                    if record["estimator_variant"] == "commutator" and not bool(
                        record["bound_rigorous"]
                    ):
                        record.update(
                            status="error",
                            error_type=NonRigorousCommutatorResultError.__name__,
                            error_message=(
                                "commutator-labelled result used non-rigorous bound "
                                f"{record['bound_method']!r}"
                            ),
                        )
                except UnsupportedEmpiricalCalibrationError as exc:
                    status: ResourceGridStatus = (
                        "missing_empirical"
                        if record["estimator_variant"] == "empirical"
                        else "error"
                    )
                    record.update(
                        status=status,
                        error_type=type(exc).__name__,
                        error_message=str(exc) or repr(exc),
                    )
                except Exception as exc:
                    record.update(
                        status="error",
                        error_type=type(exc).__name__,
                        error_message=str(exc) or repr(exc),
                    )
                records.append(record)
    frame = pd.DataFrame.from_records(records, columns=RESOURCE_GRID_COLUMNS)
    validate_resource_grid_frame(frame, config, shards=(shard,), allow_unexpected_errors=True)
    return frame


def _log_key(value: Any) -> str:
    return format(float(value), ".12g")


def _truthy(value: Any) -> bool:
    return value is True or value == 1 or (isinstance(value, str) and value.lower() == "true")


def validate_resource_grid_frame(
    frame: pd.DataFrame,
    config: ResourceGridConfig,
    *,
    shards: Sequence[ResourceGridShard] | None = None,
    allow_unexpected_errors: bool = False,
) -> None:
    """Validate schema, completeness, statuses, resources, and resolved parameters."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing_columns = set(RESOURCE_GRID_COLUMNS) - set(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"resource-grid data is missing required columns: {names}")
    if frame.empty:
        raise ValueError("resource-grid data must not be empty")
    if set(frame["grid_schema_version"].astype(str)) != {RESOURCE_GRID_SCHEMA_VERSION}:
        raise ValueError("resource-grid schema version is incompatible")
    if set(frame["config_digest"].astype(str)) != {config.digest}:
        raise ValueError("resource-grid data does not match the configuration digest")
    selected_shards = tuple(shards) if shards is not None else expand_resource_grid(config)
    expected = {
        (shard.model.model, shard.system_qubits, _log_key(log_error), method.method_id)
        for shard in selected_shards
        for log_error in config.log10_target_errors
        for method in config.methods
    }
    actual_keys = [
        (str(row.hamiltonian_model), int(row.system_qubits), _log_key(row.log10_target_error), str(row.method_id))
        for row in frame.itertuples(index=False)
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("resource-grid rows contain duplicate point/method keys")
    if set(actual_keys) != expected:
        raise ValueError("resource-grid rows are incomplete or contain unexpected keys")
    target_errors = pd.to_numeric(frame["target_error"], errors="coerce").to_numpy()
    derived_errors = np.power(
        10.0,
        pd.to_numeric(frame["log10_target_error"], errors="coerce").to_numpy(),
    )
    if not np.isfinite(target_errors).all() or not np.allclose(
        target_errors, derived_errors, rtol=1e-13, atol=0.0
    ):
        raise ValueError("target_error does not match log10_target_error")
    statuses = set(frame["status"].astype(str))
    if not statuses.issubset({"ok", "missing_empirical", "error"}):
        raise ValueError("resource-grid status values are invalid")
    successful = frame[frame["status"] == "ok"]
    counts = successful[["t_count", "cnot_count"]].apply(pd.to_numeric, errors="coerce")
    if not successful.empty and (
        not np.isfinite(counts.to_numpy(dtype=float)).all() or (counts < 0).any().any()
    ):
        raise ValueError("successful resource counts must be finite and non-negative")
    trotter = successful[successful["method_family"] == "trotter"]
    if (
        pd.to_numeric(trotter["trotter_reps"], errors="coerce").isna().any()
        or (pd.to_numeric(trotter["trotter_reps"], errors="coerce") <= 0).any()
    ):
        raise ValueError("successful Trotter rows require positive trotter_reps")
    mpf = successful[successful["method_family"] == "multiproduct"]
    for column in ("mpf_branch_count", "mpf_segments"):
        values = pd.to_numeric(mpf[column], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise ValueError(f"successful MPF rows require positive {column}")
    qsvt = successful[successful["method_family"] == "qsvt"]
    degrees = pd.to_numeric(qsvt["qsvt_degree"], errors="coerce")
    if degrees.isna().any() or (degrees <= 0).any() or (degrees.astype(int) % 2 != 1).any():
        raise ValueError("successful QSVT rows require a positive odd qsvt_degree")
    commutator = successful[successful["estimator_variant"] == "commutator"]
    if not commutator["bound_rigorous"].map(_truthy).all():
        raise ValueError("commutator-labelled successful rows must be rigorous")
    missing_empirical = frame[frame["status"] == "missing_empirical"]
    if (
        not missing_empirical["estimator_variant"].eq("empirical").all()
        or not missing_empirical["error_type"].eq(
            UnsupportedEmpiricalCalibrationError.__name__
        ).all()
    ):
        raise ValueError("missing_empirical rows must use the empirical calibration exception")
    failed = frame[frame["status"].isin(("missing_empirical", "error"))]
    if failed["error_message"].isna().any() or failed["error_message"].eq("").any():
        raise ValueError("non-success rows require an error message")
    unexpected = frame[frame["status"] == "error"]
    if not allow_unexpected_errors and not unexpected.empty:
        raise ValueError(f"resource-grid data contains {len(unexpected)} unexpected failures")


def load_resource_grid(
    path: str | Path,
    config: ResourceGridConfig | None = None,
    *,
    allow_unexpected_errors: bool = True,
) -> pd.DataFrame:
    data_path = Path(path)
    if data_path.is_dir():
        data_path = data_path / "resource_grid.csv"
    frame = pd.read_csv(data_path, keep_default_na=True)
    if config is not None:
        validate_resource_grid_frame(
            frame,
            config,
            allow_unexpected_errors=allow_unexpected_errors,
        )
    elif set(RESOURCE_GRID_COLUMNS) - set(frame.columns):
        raise ValueError("resource-grid CSV is missing required columns")
    return frame

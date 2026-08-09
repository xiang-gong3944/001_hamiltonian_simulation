import json
import math

import numpy as np
import pandas as pd
import pytest

from hamiltonian_resources import (
    BENCHMARK_COLUMNS,
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    TimeScaling,
    TrotterMethod,
    run_benchmark,
    transverse_field_ising,
    validate_benchmark_frame,
)


def test_numpy_and_pandas_sequences_are_normalized_to_lists():
    config = BenchmarkConfig(
        system_sizes=np.arange(2, 9, 2),
        target_errors=pd.Index(np.logspace(-1, -3, 3)),
        methods=[QSVTMethod()],
    )

    assert config.system_sizes == [2, 4, 6, 8]
    assert config.target_errors == pytest.approx([0.1, 0.01, 0.001])
    assert isinstance(config.system_sizes, list)
    assert isinstance(config.target_errors, list)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("system_sizes", [True], "system_sizes"),
        ("system_sizes", [0], "system_sizes"),
        ("system_sizes", [2, 2], "duplicates"),
        ("target_errors", [np.inf], "target_errors"),
        ("target_errors", [0.0], "target_errors"),
        ("target_errors", [0.1, 0.1], "duplicates"),
    ],
)
def test_config_rejects_invalid_sweep_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        BenchmarkConfig(**{field: value})


def test_proportional_time_reaches_every_method_and_sweep():
    methods = [TrotterMethod(2), MultiproductMethod(3), QSVTMethod()]
    config = BenchmarkConfig(
        system_sizes=np.array([2, 4, 8]),
        target_errors=np.array([1e-2, 1e-3]),
        time=TimeScaling("proportional", 1.0),
        fixed_system_size=np.int64(4),
        methods=methods,
    )
    frame = run_benchmark(config)

    size_rows = frame[frame["sweep"] == "system-size"]
    error_rows = frame[frame["sweep"] == "target-error"]
    assert dict(
        size_rows[["system_qubits", "evolution_time"]]
        .drop_duplicates()
        .to_numpy()
    ) == {2: 2.0, 4: 4.0, 8: 8.0}
    assert set(error_rows["evolution_time"]) == {4.0}
    assert len(frame) == (3 + 2) * len(methods)
    assert set(frame["status"]) == {"ok"}


def test_fixed_time_and_custom_factory_are_supported():
    seen_sizes = []

    def factory(size):
        seen_sizes.append(size)
        return transverse_field_ising(size, field=0.7)

    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec("custom-tfim", factory=factory),
        system_sizes=(2, 3),
        target_errors=(1e-2,),
        time=TimeScaling("fixed", 0.25),
        methods=[QSVTMethod()],
    )
    frame = run_benchmark(config, sweeps="system-size")

    assert seen_sizes == [2, 3]
    assert set(frame["evolution_time"]) == {0.25}


def test_method_selection_and_progress_order_are_explicit():
    methods = [QSVTMethod(), TrotterMethod(4)]
    events = []
    config = BenchmarkConfig(system_sizes=[2], methods=methods)
    frame = run_benchmark(config, sweeps="system-size", progress=events.append)

    assert frame["method_id"].tolist() == ["qsvt", "trotter-p4"]
    assert [event.method_id for event in events] == ["qsvt", "trotter-p4"]
    assert events[-1].completed == events[-1].total == 2


def test_dynamic_mpf_rows_store_the_resolved_policy_order_and_inputs():
    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            parameters={"coupling": 1.0, "field": 3.0, "periodic": False}
        ),
        system_sizes=[4],
        time=TimeScaling("fixed", 4.0),
        fixed_target_error=1e-3,
        methods=[
            MultiproductMethod(
                None,
                branch_count_policy="mizuta2026-theorem6",
            )
        ],
    )

    row = run_benchmark(config, sweeps="system-size").iloc[0]

    assert row["status"] == "ok"
    assert row["mpf_branch_count_policy"] == "mizuta2026-theorem6"
    assert row["mpf_term_count"] == 6
    assert row["mpf_formal_order"] == 12
    assert row["mpf_branch_count_policy_extensiveness_g"] == pytest.approx(5.0)
    assert row["mpf_branch_count_policy_target_error"] == pytest.approx(9e-4)
    assert row["query_count"] == 3 * row["segment_count"] * sum(
        json.loads(row["mpf_exponents_json"])
    )


def test_unsupported_dynamic_order_is_an_explicit_benchmark_error_row():
    from hamiltonian_resources import PauliHamiltonian

    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            "unit-z",
            factory=lambda size: PauliHamiltonian.from_terms(size, [("Z", 1.0)]),
        ),
        system_sizes=[1],
        time=TimeScaling("fixed", 1.0),
        fixed_target_error=math.exp(-31.0) / 0.9,
        methods=[
            MultiproductMethod(
                None,
                branch_count_policy="mizuta2026-theorem6",
            )
        ],
    )

    row = run_benchmark(config, sweeps="system-size").iloc[0]

    assert row["status"] == "error"
    assert row["error_type"] == "ValueError"
    assert "unsupported J=16" in row["error_message"]
    assert "N=1" in row["error_message"]
    assert "2 <= J <= 15" in row["error_message"]


def test_config_is_mutable_and_revalidated_before_run():
    config = BenchmarkConfig(system_sizes=[2], methods=[QSVTMethod()])
    config.system_sizes = np.array([3, 5], dtype=np.int64)
    frame = run_benchmark(config, sweeps="system-size")
    assert sorted(frame["system_qubits"].unique()) == [3, 5]

    config.system_sizes = [0]
    with pytest.raises(ValueError, match="system_sizes"):
        run_benchmark(config, sweeps="system-size")


def test_schema_validation_allows_derived_columns_and_reordering():
    frame = run_benchmark(
        BenchmarkConfig(system_sizes=[2], methods=[QSVTMethod()]),
        sweeps="system-size",
    )
    modified = frame.assign(derived_ratio=frame["t_count"] / frame["cnot_count"])
    modified = modified[["derived_ratio", *reversed(BENCHMARK_COLUMNS)]]
    validate_benchmark_frame(modified)

    legacy = frame.copy()
    legacy["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="schema 1.x is not compatible"):
        validate_benchmark_frame(legacy)

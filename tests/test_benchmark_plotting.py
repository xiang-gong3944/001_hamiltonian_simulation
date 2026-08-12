import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from hamiltonian_resources import (
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    TimeScaling,
    TrotterMethod,
    compare_mpf_bounds,
    plot_benchmark,
    plot_mpf_crossover,
    run_benchmark,
    select_best_by_family,
)


@pytest.fixture
def benchmark_frame(small_benchmark_config):
    return run_benchmark(small_benchmark_config)


def test_full_plot_defaults_and_arbitrary_metric(benchmark_frame):
    size_figure = plot_benchmark(
        benchmark_frame, sweep="system-size", metric="total_qubits"
    )
    error_figure = plot_benchmark(
        benchmark_frame,
        sweep="target-error",
        metric="cnot_count",
        xscale="linear",
        yscale="linear",
    )
    size_axis = size_figure.axes[0]
    error_axis = error_figure.axes[0]

    assert isinstance(size_figure.canvas, FigureCanvasAgg)
    assert size_figure.number in plt.get_fignums()
    assert size_axis.get_xscale() == "log"
    assert size_axis.xaxis._scale.base == 10
    assert size_axis.get_yscale() == "log"
    assert len(size_axis.lines) == 3
    assert error_axis.get_xscale() == "linear"
    assert error_axis.get_yscale() == "linear"
    assert error_axis.get_xlim()[0] > error_axis.get_xlim()[1]


def test_proportional_time_is_described_in_size_title():
    config = BenchmarkConfig(system_sizes=[2, 4], methods=[QSVTMethod()])
    frame = run_benchmark(config, sweeps="system-size")
    figure = plot_benchmark(frame, sweep="system-size", metric="t_count")
    assert "t=1×n" in figure.axes[0].get_title()


def test_best_by_family_retains_selected_method_identity():
    config = BenchmarkConfig(
        system_sizes=[2, 3],
        time=TimeScaling("fixed", 0.1),
        methods=[TrotterMethod(1), TrotterMethod(2), QSVTMethod()],
    )
    frame = run_benchmark(config, sweeps="system-size")
    best = select_best_by_family(frame, "t_count", sweep="system-size")

    assert {"selected_method_id", "selected_method_label", "summary_label"} <= set(
        best.columns
    )
    assert set(best["selected_method_id"]) <= {"trotter-p1", "trotter-p2"}
    figure = plot_benchmark(
        frame, sweep="system-size", metric="t_count", summary=True
    )
    assert {line.get_label().split(" [")[0] for line in figure.axes[0].lines} == {
        "Trotter",
    }


def test_summary_can_select_by_t_count_for_cnot_plot_and_annotate_orders(
    benchmark_frame,
):
    frame = benchmark_frame[benchmark_frame["sweep"] == "system-size"].copy()
    trotter_p4 = frame[frame["method_family"] == "trotter"].copy()
    trotter_p4["method_id"] = "trotter-p4"
    trotter_p4["method_label"] = "Trotter-Suzuki p=4"
    trotter_p4["trotter_order"] = 4
    frame.loc[frame["method_family"] == "multiproduct", "method_id"] = "mpf-j3"
    mpf_j5 = frame[frame["method_family"] == "multiproduct"].copy()
    mpf_j5["method_id"] = "mpf-j5"
    mpf_j5["method_label"] = "MPF J=5"
    mpf_j5["mpf_term_count"] = 5
    mpf_j5["mpf_formal_order"] = 10
    frame = pd.concat([frame, trotter_p4, mpf_j5], ignore_index=True)

    values = {
        ("trotter-p2", 2): (10, 100),
        ("trotter-p2", 3): (30, 1),
        ("trotter-p4", 2): (20, 1),
        ("trotter-p4", 3): (5, 100),
        ("mpf-j3", 2): (12, 120),
        ("mpf-j3", 3): (40, 2),
        ("mpf-j5", 2): (24, 2),
        ("mpf-j5", 3): (6, 120),
    }
    for (method_id, size), (t_count, cnot_count) in values.items():
        row = (frame["method_id"] == method_id) & (frame["system_qubits"] == size)
        frame.loc[row, ["t_count", "cnot_count"]] = [t_count, cnot_count]

    figure = plot_benchmark(
        frame,
        sweep="system-size",
        metric="cnot_count",
        summary=True,
        selection_metric="t_count",
        certification_policy="declared-bound-scope",
    )
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.lines}

    assert lines["Trotter"].get_ydata().tolist() == [100, 100]
    assert lines["MPF"].get_ydata().tolist() == [120, 120]
    assert {text.get_text() for text in axis.texts} == {
        "p=2",
        "p=4",
        "J=3, 2J=6",
        "J=5, 2J=10",
    }
    assert "best by T count" in axis.get_title()

    default_figure = plot_benchmark(
        frame,
        sweep="system-size",
        metric="cnot_count",
        summary=True,
        certification_policy="declared-bound-scope",
    )
    default_lines = {line.get_label(): line for line in default_figure.axes[0].lines}
    assert default_lines["Trotter"].get_ydata().tolist() == [1, 1]
    assert default_lines["MPF"].get_ydata().tolist() == [2, 2]


def test_plot_can_group_mixed_models_without_unique_value_failure(benchmark_frame):
    second = benchmark_frame.copy()
    second["hamiltonian_model"] = "comparison-model"
    mixed = pd.concat([benchmark_frame, second], ignore_index=True)
    figure = plot_benchmark(
        mixed,
        sweep="system-size",
        metric="t_count",
        series_by=["hamiltonian_model", "method_label"],
    )
    assert len(figure.axes[0].lines) == 6


def test_heuristic_mpf_is_excluded_from_rigorous_best_and_styled():
    config = BenchmarkConfig(
        system_sizes=[2],
        time=TimeScaling("fixed", 0.2),
        methods=[
            MultiproductMethod(3),
            MultiproductMethod(3, error_method="legacy-w2-proxy"),
        ],
    )
    frame = run_benchmark(config, sweeps="system-size")
    rigorous = frame[frame["bound_rigorous"]].iloc[0]
    heuristic = frame[~frame["bound_rigorous"]].iloc[0]
    frame.loc[frame["method_id"] == heuristic["method_id"], "t_count"] = 1

    best = select_best_by_family(
        frame,
        "t_count",
        sweep="system-size",
        certification_policy="declared-bound-scope",
    )
    unconstrained = select_best_by_family(
        frame,
        "t_count",
        sweep="system-size",
        certification_policy="unconstrained",
    )
    figure = plot_benchmark(frame, sweep="system-size", metric="t_count")
    heuristic_lines = [
        line for line in figure.axes[0].lines if "heuristic" in line.get_label().lower()
    ]

    assert best.iloc[0]["method_id"] == rigorous["method_id"]
    assert unconstrained.iloc[0]["method_id"] == heuristic["method_id"]
    assert len(heuristic_lines) == 1
    assert heuristic_lines[0].get_linestyle() == ":"


def test_default_strict_summary_excludes_ideal_only_mpf_and_qsvt_rows():
    config = BenchmarkConfig(
        system_sizes=[2],
        time=TimeScaling("fixed", 0.2),
        methods=[TrotterMethod(2), MultiproductMethod(3), QSVTMethod()],
    )
    frame = run_benchmark(config, sweeps="system-size")

    strict = select_best_by_family(frame, "t_count", sweep="system-size")
    declared = select_best_by_family(
        frame,
        "t_count",
        sweep="system-size",
        certification_policy="declared-bound-scope",
    )
    strict_figure = plot_benchmark(
        frame,
        sweep="system-size",
        metric="t_count",
        summary=True,
    )
    declared_figure = plot_benchmark(
        frame,
        sweep="system-size",
        metric="t_count",
        summary=True,
        certification_policy="declared-bound-scope",
    )

    assert set(strict["method_family"]) == {"trotter"}
    assert set(declared["method_family"]) == {"trotter", "multiproduct", "qsvt"}
    assert "policy=implemented-circuit" in strict_figure.axes[0].get_title()
    assert "policy=declared-bound-scope" in declared_figure.axes[0].get_title()
    assert {line.get_label() for line in declared_figure.axes[0].lines} == {
        "Trotter",
        "MPF",
        "QSVT",
    }


@pytest.fixture
def paired_mpf_frame():
    return run_benchmark(
        BenchmarkConfig(
            hamiltonian=HamiltonianSpec(
                parameters={"coupling": 1.0, "field": 3.0, "periodic": False}
            ),
            system_sizes=[2, 3],
            target_errors=[1e-2],
            time=TimeScaling("fixed", 0.01),
            fixed_target_error=1e-2,
            methods=[
                MultiproductMethod(2, error_method="low2019-l1-ideal-rigorous"),
                MultiproductMethod(
                    2,
                    error_method="mizuta2026-commutator-ideal-rigorous",
                ),
            ],
        ),
        sweeps="system-size",
    )


def test_compare_mpf_bounds_pairs_only_identical_circuit_structures(paired_mpf_frame):
    paired = compare_mpf_bounds(paired_mpf_frame)

    assert paired["system_qubits"].tolist() == [2, 3]
    assert (paired["mizuta_to_low_ratio"] == 1).all()
    assert set(paired["tighter_bound_method"]) == {"tie"}
    assert set(paired["mizuta_active_constraints_json"]) == {'["error","time_2"]'}

    unmatched = paired_mpf_frame.copy()
    mizuta = unmatched["bound_method"] == "mizuta2026-commutator-ideal-rigorous"
    unmatched.loc[mizuta, "rotation_synthesis_error"] *= 2
    with pytest.raises(ValueError, match="no matched"):
        compare_mpf_bounds(unmatched)

    mixed_policy = paired_mpf_frame.copy()
    mizuta = mixed_policy["bound_method"] == (
        "mizuta2026-commutator-ideal-rigorous"
    )
    mixed_policy.loc[mizuta, "mpf_branch_count_policy"] = (
        "mizuta2026-theorem6"
    )
    with pytest.raises(ValueError, match="no matched"):
        compare_mpf_bounds(mixed_policy)


def test_plot_mpf_crossover_marks_constraints_and_fallback(paired_mpf_frame):
    marked = paired_mpf_frame.copy()
    mizuta_rows = marked.index[
        marked["bound_method"] == "mizuta2026-commutator-ideal-rigorous"
    ]
    marked.loc[mizuta_rows[0], "commutator_cap_fallback"] = True

    figure = plot_mpf_crossover(marked, sweep="system-size")
    axis = figure.axes[0]
    labels = [item.get_text() for item in axis.get_legend().get_texts()]

    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert len(axis.lines) == 2  # one J series plus the ratio-one crossover
    assert "active=multiple" in labels
    assert "active=multiple [fallback]" in labels

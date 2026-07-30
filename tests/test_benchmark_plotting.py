import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from hamiltonian_resources import (
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    TimeScaling,
    TrotterMethod,
    plot_benchmark,
    run_benchmark,
    select_best_by_family,
)


@pytest.fixture
def benchmark_frame():
    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            parameters={"coupling": 1.0, "field": 0.7, "periodic": False}
        ),
        system_sizes=[2, 3],
        target_errors=[1e-2, 1e-3],
        time=TimeScaling("fixed", 0.2),
        fixed_system_size=3,
        fixed_target_error=1e-2,
        methods=[TrotterMethod(2), MultiproductMethod(3), QSVTMethod()],
    )
    return run_benchmark(config)


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
    assert size_axis.get_xscale() == "log"
    assert size_axis.xaxis._scale.base == 2
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
    assert set(best["selected_method_id"]) <= {"trotter-p1", "trotter-p2", "qsvt"}
    figure = plot_benchmark(
        frame, sweep="system-size", metric="t_count", summary=True
    )
    assert {line.get_label().split(" [")[0] for line in figure.axes[0].lines} == {
        "Best evaluated Trotter",
        "QSVT",
    }


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

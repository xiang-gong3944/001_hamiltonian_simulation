import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from hamiltonian_resources import (
    BENCHMARK_COLUMNS,
    FAMILY_COLORS,
    METHOD_LABELS,
    METHOD_STYLES,
    MPF_TERM_COUNTS,
    SCHEMA_VERSION,
    TROTTER_ORDERS,
    ScalingBenchmarkConfig,
    create_benchmark_figure,
    generate_benchmark_sweep,
    load_benchmark_config,
    load_benchmark_data,
    plot_saved_benchmark,
    save_benchmark_data,
)
from hamiltonian_resources.benchmark_cli import main as benchmark_main


@pytest.fixture
def small_config(tmp_path):
    return ScalingBenchmarkConfig(
        model_parameters={"coupling": 1.0, "field": 3.0, "periodic": False},
        system_qubit_values=(2, 3),
        target_error_values=(1e-2, 1e-3),
        evolution_time=0.5,
        fixed_system_qubits_for_error_sweep=3,
        fixed_target_error_for_size_sweep=1e-3,
        output_directory=tmp_path,
        output_formats=("png", "svg"),
    )


@pytest.fixture
def size_frame(small_config):
    return generate_benchmark_sweep(small_config, "system-size")


@pytest.fixture
def error_frame(small_config):
    return generate_benchmark_sweep(small_config, "target-error")


def test_default_config_loads_and_resolves_output_directory():
    path = Path(__file__).resolve().parents[1] / "benchmark_config.json"
    config = load_benchmark_config(path)

    assert config.hamiltonian_model == "transverse_field_ising"
    assert config.model_parameters == {
        "coupling": 1.0,
        "field": 3.0,
        "periodic": False,
    }
    assert config.output_directory == path.parent / "benchmark_outputs"
    assert config.output_formats == ("png", "pdf")
    assert config.system_qubit_values == (2, 4, 8, 16, 32, 64, 128, 256, 500)
    assert config.target_error_values == (
        0.1,
        0.03,
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0001,
    )
    assert config.evolution_time_mode == "system-size"
    assert config.fixed_system_qubits_for_error_sweep == 100
    assert config.skip_expensive_higher_order_bounds
    assert SCHEMA_VERSION == "1.1"


def test_every_sweep_point_contains_the_eight_fixed_configurations(size_frame):
    assert tuple(size_frame.columns) == BENCHMARK_COLUMNS
    assert len(size_frame) == 2 * len(METHOD_LABELS)
    assert TROTTER_ORDERS == (1, 2, 4, 6)
    assert MPF_TERM_COUNTS == (3, 5, 7)
    for _, rows in size_frame.groupby("system_qubits", sort=False):
        assert tuple(rows["method_label"]) == METHOD_LABELS


def test_success_rows_have_resources_and_explicit_method_metadata(size_frame):
    assert set(size_frame["status"]) == {"ok"}
    assert size_frame[["t_count", "cnot_count"]].notna().all().all()
    assert (size_frame[["t_count", "cnot_count"]] > 0).all().all()

    trotter = size_frame[size_frame["method_family"] == "trotter"]
    multiproduct = size_frame[size_frame["method_family"] == "multiproduct"]
    qsvt = size_frame[size_frame["method_family"] == "qsvt"]
    assert trotter["trotter_order"].notna().all()
    assert trotter["mpf_term_count"].isna().all()
    assert multiproduct["mpf_term_count"].notna().all()
    assert multiproduct["mpf_exponents_json"].notna().all()
    assert multiproduct["mpf_coefficients_json"].notna().all()
    assert multiproduct["bound_rigorous"].eq(False).all()  # noqa: E712
    assert qsvt["qsvt_degree"].notna().all()
    assert qsvt["segment_count"].isna().all()


def test_csv_schema_and_metadata_sidecar_round_trip(size_frame, small_config):
    csv_path, metadata_path = save_benchmark_data(size_frame, small_config)
    loaded = load_benchmark_data(csv_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert tuple(loaded.columns) == BENCHMARK_COLUMNS
    assert loaded["method_label"].tolist() == size_frame["method_label"].tolist()
    assert metadata["row_count"] == len(size_frame)
    assert metadata["status_counts"] == {"ok": len(size_frame)}
    assert metadata["columns"] == list(BENCHMARK_COLUMNS)
    assert metadata["configuration"]["output_formats"] == ["png", "svg"]
    assert metadata["configuration"]["evolution_time_mode"] == "fixed"


def test_schema_1_0_data_remains_loadable(size_frame, tmp_path):
    legacy = size_frame.copy()
    legacy["schema_version"] = "1.0"
    path = tmp_path / "legacy.csv"
    legacy.to_csv(path, index=False)

    loaded = load_benchmark_data(path)

    assert set(loaded["schema_version"].astype(str)) == {"1.0"}


def test_default_tfim_resources_are_nondecreasing(size_frame, error_frame):
    for _, rows in size_frame.groupby("method_label"):
        rows = rows.sort_values("system_qubits")
        assert rows["t_count"].is_monotonic_increasing
        assert rows["cnot_count"].is_monotonic_increasing
    for _, rows in error_frame.groupby("method_label"):
        rows = rows.sort_values("target_error", ascending=False)
        assert rows["t_count"].is_monotonic_increasing
        assert rows["cnot_count"].is_monotonic_increasing


def test_system_size_time_mode_sets_time_from_each_sweep_point(small_config):
    config = replace(small_config, evolution_time_mode="system-size")

    size = generate_benchmark_sweep(config, "system-size")
    error = generate_benchmark_sweep(config, "target-error")

    size_points = size[["system_qubits", "evolution_time"]].drop_duplicates()
    assert size_points.to_records(index=False).tolist() == [(2, 2.0), (3, 3.0)]
    assert set(error["system_qubits"]) == {3}
    assert set(error["evolution_time"]) == {3.0}


def test_one_estimator_failure_does_not_omit_other_methods(monkeypatch, small_config):
    import hamiltonian_resources.benchmark_suite as suite

    original = suite.estimate_resources_analytically

    def fail_mpf_five(hamiltonian, config, algorithm):
        if algorithm == "multiproduct" and config.mpf_m == 5:
            raise RuntimeError("deliberate MPF m=5 failure")
        return original(hamiltonian, config, algorithm)

    monkeypatch.setattr(suite, "estimate_resources_analytically", fail_mpf_five)
    config = replace(small_config, system_qubit_values=(2,))
    frame = generate_benchmark_sweep(config, "system-size")

    assert len(frame) == 8
    assert (frame["status"] == "ok").sum() == 7
    failure = frame[frame["status"] == "error"].iloc[0]
    assert failure["method_label"] == "MPF m=5"
    assert failure["error_type"] == "RuntimeError"
    assert "deliberate" in failure["error_message"]
    assert pd.isna(failure["t_count"])
    assert pd.isna(failure["cnot_count"])


def test_expensive_higher_order_bound_is_skipped_without_omission(
    monkeypatch, small_config
):
    import hamiltonian_resources.benchmark_suite as suite

    monkeypatch.setattr(suite, "_HIGHER_ORDER_COMMUTATOR_WORK_LIMIT", 100)
    config = replace(small_config, system_qubit_values=(2,))
    frame = generate_benchmark_sweep(config, "system-size")

    assert len(frame) == len(METHOD_LABELS)
    assert (frame["status"] == "ok").sum() == 7
    skipped = frame[frame["status"] == "skipped"].iloc[0]
    assert skipped["method_label"] == "Trotter p=6"
    assert skipped["error_type"] == "HigherOrderBoundWorkLimit"
    assert "estimated work 384 exceeds limit 100" in skipped["error_message"]
    assert skipped["trotter_partition"] == "commuting"
    assert skipped["trotter_group_count"] == 2
    assert pd.isna(skipped["t_count"])
    assert pd.isna(skipped["cnot_count"])

    forced = generate_benchmark_sweep(
        replace(config, skip_expensive_higher_order_bounds=False),
        "system-size",
    )
    assert set(forced["status"]) == {"ok"}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"hamiltonian_model": "unknown"}, "hamiltonian_model"),
        ({"system_qubit_values": (0,)}, "system_qubit_values"),
        ({"target_error_values": (0.0,)}, "target_error_values"),
        ({"output_formats": ("eps",)}, "output formats"),
        ({"model_parameters": {"bad": 1}}, "unsupported parameters"),
        ({"evolution_time_mode": "bad"}, "evolution_time_mode"),
        ({"skip_expensive_higher_order_bounds": "yes"}, "must be a boolean"),
    ],
)
def test_scaling_config_rejects_invalid_values(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ScalingBenchmarkConfig(**changes)


def test_style_mapping_uses_family_colors_and_unique_variants():
    assert tuple(METHOD_STYLES) == METHOD_LABELS
    trotter_styles = [METHOD_STYLES[f"Trotter p={order}"] for order in TROTTER_ORDERS]
    mpf_styles = [METHOD_STYLES[f"MPF m={term_count}"] for term_count in MPF_TERM_COUNTS]
    assert {style["color"] for style in trotter_styles} == {FAMILY_COLORS["trotter"]}
    assert {style["color"] for style in mpf_styles} == {FAMILY_COLORS["multiproduct"]}
    assert len({(style["linestyle"], style["marker"]) for style in trotter_styles}) == 4
    assert len({(style["linestyle"], style["marker"]) for style in mpf_styles}) == 3
    assert METHOD_STYLES["QSVT"]["color"] == FAMILY_COLORS["qsvt"]


def test_full_figures_have_required_labels_scales_and_titles(size_frame, error_frame):
    size_figure = create_benchmark_figure(size_frame, "t_count")
    error_figure = create_benchmark_figure(error_frame, "cnot_count")
    size_axis = size_figure.axes[0]
    error_axis = error_figure.axes[0]

    assert isinstance(size_figure.canvas, FigureCanvasAgg)
    assert size_axis.get_yscale() == "log"
    assert [line.get_label() for line in size_axis.lines] == list(METHOD_LABELS)
    assert "target error" in size_axis.get_title()
    assert error_axis.get_xscale() == "log"
    assert error_axis.get_yscale() == "log"
    assert error_axis.get_xlim()[0] > error_axis.get_xlim()[1]
    assert "n=3 system qubits" in error_axis.get_title()
    plt.close(size_figure)
    plt.close(error_figure)


def test_system_size_time_mode_is_named_in_plot_title(small_config):
    frame = generate_benchmark_sweep(
        replace(small_config, evolution_time_mode="system-size"),
        "system-size",
    )

    figure = create_benchmark_figure(frame, "t_count")

    assert "t=n" in figure.axes[0].get_title()
    plt.close(figure)


def test_full_figure_preserves_a_gap_and_annotates_skipped_rows(size_frame):
    gapped = size_frame.copy()
    skipped_index = gapped[
        (gapped["method_label"] == "Trotter p=6")
        & (gapped["system_qubits"] == 3)
    ].index
    gapped.loc[skipped_index, "status"] = "skipped"
    gapped.loc[skipped_index, "error_type"] = "HigherOrderBoundWorkLimit"
    gapped.loc[skipped_index, "error_message"] = "test skip"
    gapped.loc[skipped_index, ["t_count", "cnot_count"]] = None

    with pytest.warns(RuntimeWarning, match="1 skipped rows"):
        figure = create_benchmark_figure(gapped, "t_count")

    lines = {line.get_label(): line for line in figure.axes[0].lines}
    values = lines["Trotter p=6"].get_ydata()
    assert len(values) == 2
    assert not pd.isna(values[0])
    assert pd.isna(values[1])
    assert "1 skipped rows" in figure.axes[0].texts[0].get_text()
    plt.close(figure)


def test_summary_is_pointwise_minimum_of_saved_configurations(size_frame):
    figure = create_benchmark_figure(size_frame, "t_count", summary=True)
    lines = {line.get_label(): line for line in figure.axes[0].lines}
    expected = (
        size_frame[
            (size_frame["status"] == "ok")
            & (size_frame["method_family"] == "trotter")
        ]
        .groupby("system_qubits")["t_count"]
        .min()
        .tolist()
    )

    assert set(lines) == {
        "Best Trotter (p=1,2,4,6)",
        "Best MPF (m=3,5,7)",
        "QSVT",
    }
    assert lines["Best Trotter (p=1,2,4,6)"].get_ydata().tolist() == expected
    plt.close(figure)


def test_plotting_reads_saved_data_and_writes_png_and_vector(
    monkeypatch, size_frame, small_config
):
    import hamiltonian_resources.benchmark_suite as suite

    csv_path, _ = save_benchmark_data(size_frame, small_config)

    def estimator_must_not_run(*args, **kwargs):
        raise AssertionError("plotting reran resource estimation")

    monkeypatch.setattr(suite, "estimate_resources_analytically", estimator_must_not_run)
    outputs = plot_saved_benchmark(
        csv_path,
        output_formats=("png", "svg"),
        summary=True,
    )

    assert len(outputs) == 8
    assert {path.suffix for path in outputs} == {".png", ".svg"}
    assert any(path.stem == "system_size_t_count" for path in outputs)
    assert any(path.stem == "system_size_cnot_count_summary" for path in outputs)
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_cli_reports_generation_failures_with_nonzero_exit(monkeypatch, tmp_path):
    import hamiltonian_resources.benchmark_suite as suite

    config = ScalingBenchmarkConfig(
        system_qubit_values=(2,),
        output_directory=tmp_path / "output",
        output_formats=("png",),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    original = suite.estimate_resources_analytically

    def fail_qsvt(hamiltonian, benchmark_config, algorithm):
        if algorithm == "qsvt":
            raise RuntimeError("QSVT unavailable")
        return original(hamiltonian, benchmark_config, algorithm)

    monkeypatch.setattr(suite, "estimate_resources_analytically", fail_qsvt)
    status = benchmark_main(
        ["generate", "--config", str(config_path), "--sweep", "system-size"]
    )

    assert status == 1
    frame = load_benchmark_data(config.output_directory / "system_size_scaling.csv")
    assert (frame["status"] == "error").sum() == 1


def test_cli_reports_skips_without_a_failure_exit(monkeypatch, tmp_path, capsys):
    import hamiltonian_resources.benchmark_suite as suite

    monkeypatch.setattr(suite, "_HIGHER_ORDER_COMMUTATOR_WORK_LIMIT", 100)
    config = ScalingBenchmarkConfig(
        system_qubit_values=(2,),
        output_directory=tmp_path / "output",
        output_formats=("png",),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")

    status = benchmark_main(
        ["generate", "--config", str(config_path), "--sweep", "system-size"]
    )

    assert status == 0
    assert "0 failures, 1 skipped" in capsys.readouterr().out
    frame = load_benchmark_data(config.output_directory / "system_size_scaling.csv")
    assert frame["status"].value_counts().to_dict() == {"ok": 7, "skipped": 1}
    metadata_path = config.output_directory / "system_size_scaling.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status_counts"] == {"ok": 7, "skipped": 1}


def test_heisenberg_model_registry_is_supported(tmp_path):
    config = ScalingBenchmarkConfig(
        hamiltonian_model="heisenberg_chain",
        model_parameters={"coupling": 1.0, "field_z": 0.3},
        system_qubit_values=(2,),
        output_directory=tmp_path,
    )
    frame = generate_benchmark_sweep(config, "system-size")

    assert len(frame) == 8
    assert set(frame["status"]) == {"ok"}
    assert set(frame["hamiltonian_model"]) == {"heisenberg_chain"}

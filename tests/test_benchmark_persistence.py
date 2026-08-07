import json
from pathlib import Path

import pytest

from hamiltonian_resources import (
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    TimeScaling,
    TrotterMethod,
    load_benchmark,
    load_benchmark_job,
    run_benchmark,
    save_benchmark,
    save_benchmark_plots,
)
from hamiltonian_resources.benchmark_cli import main as benchmark_main
from hamiltonian_resources.benchmark_cli import _worker_count


@pytest.fixture
def small_config():
    return BenchmarkConfig(
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


@pytest.fixture
def benchmark_frame(small_config):
    return run_benchmark(small_config)


def test_default_job_uses_proportional_time_and_resolves_output_root():
    path = Path(__file__).resolve().parents[1] / "benchmark_config.json"
    job = load_benchmark_job(path)

    assert job.benchmark.time.mode == "proportional"
    assert job.benchmark.time.coefficient == 1.0
    assert job.benchmark.system_sizes == [2, 4, 6, 8, 10, 12]
    assert job.output_root == path.parent / "benchmark_outputs"
    assert job.output_formats == ["png", "pdf"]


def test_cli_worker_selection_is_capped_and_validated(monkeypatch):
    monkeypatch.setattr("hamiltonian_resources.benchmark_cli.os.cpu_count", lambda: 20)

    assert _worker_count(0) == 4
    assert _worker_count(2) == 2
    with pytest.raises(ValueError, match="workers"):
        _worker_count(-1)


def test_python_run_is_in_memory_and_explicit_save_is_collision_free(
    tmp_path, small_config
):
    first = run_benchmark(small_config, sweeps="system-size")
    assert list(tmp_path.iterdir()) == []
    first_directory, first_csv, first_metadata = save_benchmark(
        first, small_config, output_root=tmp_path
    )
    second = run_benchmark(small_config, sweeps="system-size")
    second_directory, _, _ = save_benchmark(second, small_config, output_root=tmp_path)

    assert first_directory != second_directory
    assert first_csv.name == "benchmark.csv"
    assert first_metadata.name == "metadata.json"
    loaded = load_benchmark(first_csv)
    metadata = json.loads(first_metadata.read_text(encoding="utf-8"))
    mpf_config = next(
        method
        for method in metadata["configuration"]["methods"]
        if method["family"] == "multiproduct"
    )

    assert loaded["method_id"].tolist() == first["method_id"].tolist()
    assert {
        "bound_method",
        "bound_rigorous",
        "bound_scope",
        "circuit_bound_rigorous",
    } <= set(metadata["columns"])
    assert mpf_config["error_method"] == "low2019-l1-ideal-rigorous"


def test_load_benchmark_does_not_require_metadata(tmp_path, benchmark_frame):
    csv_path = tmp_path / "standalone.csv"
    benchmark_frame.to_csv(csv_path, index=False)
    loaded = load_benchmark(csv_path)
    assert len(loaded) == len(benchmark_frame)


def test_w2_triangle_provenance_round_trips_without_schema_relabeling(tmp_path):
    config = BenchmarkConfig(
        system_sizes=[2],
        time=TimeScaling("fixed", 0.2),
        methods=[
            MultiproductMethod(
                3,
                error_method="childs2021-w2-triangle-ideal-rigorous",
            ),
            MultiproductMethod(3, error_method="legacy-w2-proxy"),
        ],
    )
    frame = run_benchmark(config, sweeps="system-size")
    _, csv_path, _ = save_benchmark(frame, config, output_root=tmp_path)
    loaded = load_benchmark(csv_path)
    rigorous = loaded[
        loaded["bound_method"] == "childs2021-w2-triangle-ideal-rigorous"
    ].iloc[0]
    historical = loaded[loaded["bound_method"] == "legacy-w2-proxy"].iloc[0]
    components = json.loads(rigorous["bound_components_json"])

    assert set(components) == {
        "w2",
        "b2",
        "local_step_size",
        "local_step_error",
        "repeated_ideal_mpf_error",
    }
    assert bool(rigorous["bound_rigorous"])
    assert not bool(historical["bound_rigorous"])
    assert historical["method_id"] == "mpf-m3-legacy-w2-proxy"


def test_load_benchmark_upgrades_early_schema2_scope_columns(
    tmp_path, benchmark_frame
):
    extension_columns = [
        "bound_scope",
        "bound_target_satisfied",
        "circuit_bound_scope",
        "circuit_bound_rigorous",
        "circuit_target_satisfied",
    ]
    legacy = benchmark_frame.drop(columns=extension_columns)
    csv_path = tmp_path / "early-schema2.csv"
    legacy.to_csv(csv_path, index=False)

    loaded = load_benchmark(csv_path)
    mpf = loaded[loaded["method_family"] == "multiproduct"].iloc[0]

    assert set(extension_columns) <= set(loaded.columns)
    assert mpf["bound_scope"] == "ideal-mpf"
    assert mpf["circuit_bound_scope"] == "repeated-shared-ancilla-good-block"
    assert not bool(mpf["circuit_bound_rigorous"])


def test_load_benchmark_downgrades_withdrawn_qsvt_circuit_claim(
    tmp_path, benchmark_frame
):
    legacy = benchmark_frame.copy()
    qsvt = legacy["method_family"] == "qsvt"
    legacy.loc[qsvt, "bound_scope"] = "implemented-algorithm"
    legacy.loc[qsvt, "bound_rigorous"] = True
    legacy.loc[qsvt, "bound_target_satisfied"] = True
    legacy.loc[qsvt, "circuit_bound_scope"] = "implemented-algorithm"
    legacy.loc[qsvt, "circuit_bound_rigorous"] = True
    legacy.loc[qsvt, "circuit_target_satisfied"] = True
    csv_path = tmp_path / "legacy-qsvt-claim.csv"
    legacy.to_csv(csv_path, index=False)

    loaded = load_benchmark(csv_path)
    row = loaded[loaded["method_family"] == "qsvt"].iloc[0]

    assert row["bound_scope"] == "legacy-qsvt-unscoped"
    assert not bool(row["bound_rigorous"])
    assert not bool(row["bound_target_satisfied"])
    assert row["circuit_bound_scope"] == "implemented-qsvt-floating-phase-circuit"
    assert not bool(row["circuit_bound_rigorous"])
    assert not bool(row["circuit_target_satisfied"])


def test_save_standard_plots(tmp_path, benchmark_frame):
    outputs = save_benchmark_plots(
        benchmark_frame,
        output_directory=tmp_path,
        output_formats=["png", "svg"],
        summary=True,
    )
    assert len(outputs) == 16
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_cli_creates_a_new_run_directory_and_reports_failures(
    monkeypatch, tmp_path, small_config
):
    import hamiltonian_resources.benchmark_suite as suite

    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "benchmark": small_config.as_dict(),
                "output": {"root": "runs", "formats": ["png"]},
            }
        ),
        encoding="utf-8",
    )
    original = suite.estimate_resources

    def fail_qsvt(hamiltonian, method, time, target_error, **kwargs):
        if method.family == "qsvt":
            raise RuntimeError("QSVT unavailable")
        return original(hamiltonian, method, time, target_error, **kwargs)

    monkeypatch.setattr(suite, "estimate_resources", fail_qsvt)
    status = benchmark_main(
        ["generate", "--config", str(job_path), "--sweep", "system-size"]
    )

    assert status == 1
    csv_paths = list((tmp_path / "runs").glob("*/benchmark.csv"))
    assert len(csv_paths) == 1
    frame = load_benchmark(csv_paths[0])
    assert (frame["status"] == "error").sum() == 2

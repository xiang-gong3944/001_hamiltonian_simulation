import json

import pandas as pd
import pytest

from hamiltonian_resources import (
    HamiltonianSpec,
    QSVTMethod,
    ResourceGridConfig,
    TimeScaling,
    load_resource_grid,
    run_resource_grid,
)
from hamiltonian_resources.resource_grid_cli import main as resource_grid_main


def _config(*, models=None, log_errors=(-1.0,)):
    return ResourceGridConfig(
        models=models
        or [
            HamiltonianSpec(
                "transverse_field_ising",
                {"coupling": 1.0, "field": 3.0, "periodic": False},
            )
        ],
        system_sizes=[2],
        log10_target_errors=log_errors,
        methods=[QSVTMethod()],
        time=TimeScaling("fixed", 0.1),
    )


def test_runner_writes_manifest_validation_shards_and_deterministic_merge(tmp_path):
    config = _config(log_errors=(-1.0, -2.0))
    output = tmp_path / "grid"

    summary = run_resource_grid(config, output, workers=1)
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(summary.validation_path.read_text(encoding="utf-8"))
    frame = load_resource_grid(summary.merged_path, config)

    assert summary.completed_shards == 1
    assert summary.skipped_shards == 0
    assert summary.failed_rows == 0
    assert manifest["run"]["status"] == "complete"
    assert manifest["shards"]["transverse_field_ising:N002"]["state"] == "complete"
    assert validation["valid"]
    assert validation["row_count"] == 2
    assert frame["log10_target_error"].tolist() == [-2.0, -1.0]
    assert not list(output.rglob("*.tmp"))


def test_resume_skips_compatible_completed_shards(monkeypatch, tmp_path):
    import hamiltonian_resources.resource_grid as grid

    config = _config()
    output = tmp_path / "grid"
    run_resource_grid(config, output, workers=1)

    def should_not_run(*args, **kwargs):
        raise AssertionError("completed shard was recomputed")

    monkeypatch.setattr(grid, "evaluate_resource_grid_shard", should_not_run)
    summary = run_resource_grid(config, output, resume=True, workers=1)

    assert summary.skipped_shards == 1
    assert summary.failed_rows == 0


def test_resume_reconciles_atomic_shard_written_before_manifest_update(
    monkeypatch, tmp_path
):
    import hamiltonian_resources.resource_grid as grid

    config = _config()
    output = tmp_path / "grid"
    run_resource_grid(config, output, workers=1)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["shards"]["transverse_field_ising:N002"]
    entry.update(
        state="pending",
        checksum_sha256=None,
        row_count=None,
        status_counts={},
        completed_at_utc=None,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        grid,
        "evaluate_resource_grid_shard",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid orphan shard was recomputed")
        ),
    )
    summary = run_resource_grid(config, output, resume=True, workers=1)

    assert summary.skipped_shards == 1
    reconciled = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert reconciled["shards"]["transverse_field_ising:N002"]["state"] == "complete"


def test_resume_retries_failed_shards(monkeypatch, tmp_path):
    import hamiltonian_resources.resource_grid as grid

    config = _config()
    output = tmp_path / "grid"
    original = grid.estimate_resources

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic estimator failure")

    monkeypatch.setattr(grid, "estimate_resources", fail)
    failed = run_resource_grid(config, output, workers=1)
    assert failed.failed_rows == 1
    manifest = json.loads(failed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["shards"]["transverse_field_ising:N002"]["state"] == "failed"

    monkeypatch.setattr(grid, "estimate_resources", original)
    retried = run_resource_grid(config, output, resume=True, workers=1)
    assert retried.skipped_shards == 0
    assert retried.failed_rows == 0


def test_resume_rejects_configuration_and_checksum_mismatches(tmp_path):
    config = _config()
    output = tmp_path / "grid"
    run_resource_grid(config, output, workers=1)

    with pytest.raises(ValueError, match="configuration"):
        run_resource_grid(
            _config(log_errors=(-2.0,)), output, resume=True, workers=1
        )

    shard_path = output / "transverse_field_ising" / "N002.csv"
    shard_path.write_bytes(shard_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_resource_grid(config, output, resume=True, workers=1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("grid_schema_version", "0.9", "grid schema"),
        ("source_digest", "0" * 64, "estimator source"),
    ],
)
def test_resume_rejects_schema_and_source_mismatches(tmp_path, field, value, message):
    config = _config()
    output = tmp_path / field
    summary = run_resource_grid(config, output, workers=1)
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    summary.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_resource_grid(config, output, resume=True, workers=1)


def test_nonempty_output_requires_resume(tmp_path):
    output = tmp_path / "grid"
    output.mkdir()
    (output / "unrelated.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_resource_grid(_config(), output, workers=1)


def test_parallel_and_serial_runner_results_match(tmp_path):
    models = [
        HamiltonianSpec(
            "transverse_field_ising",
            {"coupling": 1.0, "field": 3.0, "periodic": False},
        ),
        HamiltonianSpec("heisenberg_chain", {"coupling": 1.0, "field_z": 0.3}),
    ]
    config = _config(models=models)
    serial = load_resource_grid(
        run_resource_grid(config, tmp_path / "serial", workers=1).merged_path,
        config,
    )
    parallel = load_resource_grid(
        run_resource_grid(config, tmp_path / "parallel", workers=2).merged_path,
        config,
    )
    columns = [
        "hamiltonian_model",
        "system_qubits",
        "target_error",
        "method_id",
        "qsvt_degree",
        "t_count",
        "cnot_count",
        "status",
    ]

    pd.testing.assert_frame_equal(serial[columns], parallel[columns])


def test_cli_runs_custom_grid_and_reports_compatibility_errors(tmp_path, capsys):
    config = _config()
    config_path = tmp_path / "config.json"
    output = tmp_path / "grid"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")

    status = resource_grid_main(
        [
            "run",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--workers",
            "1",
            "--no-progress",
        ]
    )
    repeated = resource_grid_main(
        [
            "run",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--workers",
            "1",
            "--no-progress",
        ]
    )

    assert status == 0
    assert repeated == 2
    assert (output / "resource_grid.csv").exists()
    assert "wrote" in capsys.readouterr().out

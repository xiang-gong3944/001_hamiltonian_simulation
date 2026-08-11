import json

import pytest

from hamiltonian_resources.calibration_pipeline import (
    assemble_calibration_artifacts,
    assemble_reproducibility_manifest,
    expand_calibration_tasks,
    load_calibration_config,
    reduce_calibration_shards,
    task_execution_digest,
)
from hamiltonian_resources.empirical import canonical_json_digest


def _config():
    return {
        "study_id": "synthetic-pipeline",
        "models": ["transverse_field_ising"],
        "model_parameters": {
            "transverse_field_ising": {
                "coupling": 1.0,
                "field": 3.0,
                "periodic": False,
            }
        },
        "formal_orders": [18],
        "sizes": [4],
        "segment_ratios": {"18": [3, 4, 5, 6]},
        "downstream_benchmark": {"system_sizes": [4, 20, 50, 100]},
        "reviewed_size_max": 100,
        "backend": "flint",
        "formula": "ordered-individual-pauli-strang-mpf-v1",
        "schedule": "new",
    }


def _shard(config, task):
    coefficient = 2e-7
    observations = []
    for point in task.points:
        error = coefficient * point.time**19 / point.segments**18
        observations.append(
            {
                "time": str(point.time),
                "segments": point.segments,
                "error": f"{error:.18e}",
                "coefficient_b_2j": f"{coefficient:.18e}",
                "precision_converged": True,
                "wall_seconds": 0.1,
            }
        )
    return {
        "schema_version": "task-1.0",
        "study_id": config["study_id"],
        "configuration_digest": canonical_json_digest(config),
        "task_execution_digest": task_execution_digest(config, task),
        "task": {
            "kind": task.kind,
            "model": task.model,
            "formal_order": task.formal_order,
            "system_size": task.system_size,
            "points": [
                {"time": point.time, "segments": point.segments}
                for point in task.points
            ],
        },
        "task_id": task.task_id,
        "observations": observations,
        "scientific_digest": "c" * 64,
    }


def test_reducer_is_deterministic_and_assembler_accepts_window(tmp_path):
    config = _config()
    task = expand_calibration_tasks(config)[0]
    shard_path = tmp_path / f"{task.task_id}.json"
    shard_path.write_text(json.dumps(_shard(config, task)), encoding="utf-8")

    first = reduce_calibration_shards(config, [shard_path])
    second = reduce_calibration_shards(config, [shard_path])
    assembled = assemble_calibration_artifacts(first)
    provenance = assemble_reproducibility_manifest(first, assembled)

    assert first == second
    assert first["expected_task_ids"] == [task.task_id]
    assert first["missing_task_ids"] == []
    assert len(assembled["accepted_windows"]) == 1
    assert assembled["accepted_windows"][0]["coefficient_b_2j"] == pytest.approx(
        2e-7
    )
    assert assembled["size_fits"][0]["selected_model"] is None
    assert provenance["completed_task_hashes"] == {task.task_id: "c" * 64}
    assert provenance["hash_definition"] == "SHA-256 of canonical parsed JSON"
    assert not provenance["raw_shards_committed"]


def test_reducer_rejects_missing_and_wrong_configuration_shards(tmp_path):
    config = _config()
    task = expand_calibration_tasks(config)[0]
    with pytest.raises(ValueError, match="missing 1"):
        reduce_calibration_shards(config, [])

    raw = _shard(config, task)
    raw["task_execution_digest"] = "0" * 64
    shard_path = tmp_path / "wrong.json"
    shard_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="different task execution"):
        reduce_calibration_shards(config, [shard_path])


def test_task_inventory_includes_fixed_segment_time_law_checks():
    config = {
        **_config(),
        "time_law_checks": [
            {
                "model": "transverse_field_ising",
                "formal_order": 18,
                "system_size": 4,
                "segments": 40,
                "time_factors": [0.8, 1.0, 1.2],
            }
        ],
    }
    tasks = expand_calibration_tasks(config)
    check = next(task for task in tasks if task.kind == "time-law")

    assert {point.segments for point in check.points} == {40}
    assert tuple(point.time for point in check.points) == pytest.approx((3.2, 4.0, 4.8))


def test_task_inventory_applies_model_order_and_size_ratio_overrides():
    config = {
        **_config(),
        "segment_ratio_overrides": {
            "transverse_field_ising": {"18": {"4": [8, 10, 12, 16]}}
        },
    }
    task = expand_calibration_tasks(config)[0]

    assert tuple(point.segments for point in task.points) == (32, 40, 48, 64)


def test_reviewed_domain_is_derived_from_downstream_benchmark(tmp_path):
    config = _config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_calibration_config(path)

    assert loaded["reviewed_size_max"] == max(
        loaded["downstream_benchmark"]["system_sizes"]
    )

    config["reviewed_size_max"] = 101
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="downstream benchmark maximum"):
        load_calibration_config(path)

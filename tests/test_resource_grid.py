import json

import numpy as np
import pandas as pd
import pytest

from hamiltonian_resources import (
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    ResourceGridConfig,
    ResourceGridShard,
    TimeScaling,
    TrotterMethod,
    evaluate_resource_grid_shard,
    expand_resource_grid,
    load_resource_grid_config,
    resource_grid_config_from_dict,
    resource_grid_preset,
    transverse_field_ising,
    validate_resource_grid_frame,
)


def _small_config(*methods, size=3, log_errors=(-1.0,)):
    return ResourceGridConfig(
        models=[
            HamiltonianSpec(
                "transverse_field_ising",
                {"coupling": 1.0, "field": 3.0, "periodic": False},
            )
        ],
        system_sizes=[size],
        log10_target_errors=log_errors,
        methods=list(methods) or [QSVTMethod()],
        time=TimeScaling("fixed", 0.2),
    )


def test_full_preset_is_the_exact_reviewed_grid():
    config = resource_grid_preset("full")
    shards = expand_resource_grid(config)

    assert [model.model for model in config.models] == [
        "transverse_field_ising",
        "heisenberg_chain",
    ]
    assert config.system_sizes == list(range(3, 121))
    assert config.log10_target_errors == pytest.approx(
        [-1.0 - 0.1 * index for index in range(31)]
    )
    assert len(config.methods) == 9
    assert len(shards) == 236
    assert 236 * 31 * 9 == 65_844
    assert config.digest == "c1ad72de628f71283ac69037e9bbe97c3e773952f5abde5eac91397ca322d1a6"


def test_sanity_presets_use_the_full_execution_method_set():
    low = resource_grid_preset("sanity-low")
    high = resource_grid_preset("sanity-high")

    assert low.system_sizes == [3, 4, 6]
    assert low.log10_target_errors == [-1.0, -2.0]
    assert high.system_sizes == [100, 120]
    assert high.log10_target_errors == [-3.0, -4.0]
    assert [method.method_id for method in low.methods] == [
        method.method_id for method in resource_grid_preset("full").methods
    ]


def test_shard_names_and_expansion_order_are_stable():
    config = resource_grid_preset("sanity-low")
    shards = expand_resource_grid(config)

    assert shards[0].shard_id == "transverse_field_ising:N003"
    assert shards[0].relative_path == "transverse_field_ising/N003.csv"
    assert shards[2].relative_path == "transverse_field_ising/N006.csv"
    assert shards[3].relative_path == "heisenberg_chain/N003.csv"


def test_configuration_json_round_trips(tmp_path):
    config = resource_grid_preset("sanity-low")
    raw = config.as_dict()
    path = tmp_path / "grid.json"
    path.write_text(json.dumps({"resource_grid": raw}), encoding="utf-8")

    parsed = resource_grid_config_from_dict(raw)
    loaded = load_resource_grid_config(path)

    assert parsed.as_dict() == raw
    assert loaded.as_dict() == raw
    assert parsed.digest == config.digest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("system_sizes", [3, 3], "duplicates"),
        ("log10_target_errors", [-1.0, 0.0], "below zero"),
        ("models", [], "must not be empty"),
    ],
)
def test_configuration_rejects_invalid_grid_values(field, value, message):
    kwargs = _small_config().as_dict()
    kwargs[field] = value
    if field == "models":
        kwargs["models"] = value
    with pytest.raises((ValueError, TypeError), match=message):
        resource_grid_config_from_dict(kwargs)


def test_shard_builds_the_hamiltonian_once_and_uses_one_execution():
    calls = []

    def factory(system_qubits):
        calls.append(system_qubits)
        return transverse_field_ising(system_qubits, field=0.7)

    model = HamiltonianSpec("synthetic-grid-model", factory=factory)
    config = ResourceGridConfig(
        models=[model],
        system_sizes=[2],
        log10_target_errors=[-1.0, -2.0],
        methods=[QSVTMethod()],
        time=TimeScaling("fixed", 0.1),
    )

    frame = evaluate_resource_grid_shard(config, ResourceGridShard(model, 2))

    assert calls == [2]
    assert len(frame) == 2
    assert frame["status"].eq("ok").all()
    assert frame["qsvt_degree"].gt(0).all()


def test_expected_empirical_gaps_preserve_rows_and_resolved_mpf_j():
    config = _small_config(
        TrotterMethod(2, "empirical-operator-norm"),
        MultiproductMethod(
            None,
            error_method="empirical-operator-norm",
            branch_count_policy="mizuta2026-theorem6",
        ),
    )
    frame = evaluate_resource_grid_shard(config, expand_resource_grid(config)[0])

    assert frame["status"].eq("missing_empirical").all()
    assert frame["error_type"].eq("UnsupportedEmpiricalCalibrationError").all()
    mpf = frame[frame["method_family"] == "multiproduct"].iloc[0]
    assert mpf["mpf_branch_count"] >= 2
    assert pd.isna(frame[frame["method_family"] == "trotter"].iloc[0]["trotter_reps"])


def test_successful_rows_expose_method_specific_selected_parameters():
    config = _small_config(
        TrotterMethod(2),
        MultiproductMethod(3),
        QSVTMethod(),
        size=4,
    )
    frame = evaluate_resource_grid_shard(config, expand_resource_grid(config)[0])

    trotter = frame[frame["method_family"] == "trotter"].iloc[0]
    mpf = frame[frame["method_family"] == "multiproduct"].iloc[0]
    qsvt = frame[frame["method_family"] == "qsvt"].iloc[0]
    assert trotter["trotter_reps"] > 0
    assert trotter["trotter_partition"] == "individual"
    assert mpf["mpf_branch_count"] == 3
    assert mpf["mpf_segments"] > 0
    assert qsvt["qsvt_degree"] > 0 and qsvt["qsvt_degree"] % 2 == 1


def test_validation_rejects_bad_counts_parameters_and_nonrigorous_labels():
    config = _small_config(TrotterMethod(2), size=4)
    shard = expand_resource_grid(config)[0]
    frame = evaluate_resource_grid_shard(config, shard)

    for column, value, message in (
        ("t_count", np.inf, "finite"),
        ("trotter_reps", 0, "positive"),
        ("bound_rigorous", False, "must be rigorous"),
    ):
        broken = frame.copy()
        if column == "t_count":
            broken[column] = broken[column].astype(float)
        broken.loc[0, column] = value
        with pytest.raises(ValueError, match=message):
            validate_resource_grid_frame(broken, config, shards=(shard,))


def test_validation_rejects_duplicates_incomplete_rows_and_unexpected_errors():
    config = _small_config(QSVTMethod())
    shard = expand_resource_grid(config)[0]
    frame = evaluate_resource_grid_shard(config, shard)

    with pytest.raises(ValueError, match="duplicate"):
        validate_resource_grid_frame(
            pd.concat([frame, frame], ignore_index=True), config, shards=(shard,)
        )
    with pytest.raises(ValueError, match="must not be empty"):
        validate_resource_grid_frame(frame.iloc[0:0], config, shards=(shard,))
    failed = frame.copy()
    failed.loc[0, ["status", "error_type", "error_message"]] = [
        "error",
        "RuntimeError",
        "unexpected",
    ]
    validate_resource_grid_frame(
        failed, config, shards=(shard,), allow_unexpected_errors=True
    )
    with pytest.raises(ValueError, match="unexpected failures"):
        validate_resource_grid_frame(failed, config, shards=(shard,))

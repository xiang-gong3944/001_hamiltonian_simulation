from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from hamiltonian_resources import (
    AffineSizeCoefficient,
    BenchmarkConfig,
    EmpiricalCalibrationKey,
    EmpiricalCalibrationRecord,
    EmpiricalCalibrationRegistry,
    HamiltonianModelMetadata,
    HamiltonianSpec,
    MultiproductMethod,
    PauliHamiltonian,
    PowerPlusOffsetSizeCoefficient,
    PowerSizeCoefficient,
    TrotterMethod,
    TimeScaling,
    UnsupportedEmpiricalCalibrationError,
    build_simulation_circuit,
    canonical_json_digest,
    default_empirical_calibrations,
    estimate_mpf_error,
    estimate_plan_resources,
    evaluate_empirical_error,
    heisenberg_chain,
    load_benchmark,
    mpf_lcu_structure,
    multiproduct_coefficients,
    plan_simulation,
    plot_benchmark,
    run_benchmark,
    save_benchmark,
    select_empirical_segments,
    transverse_field_ising,
)
from hamiltonian_resources.calibration_study import (
    dense_operator_norm_error,
    effective_power,
    fit_affine_size_coefficient,
    observed_error_coefficient,
    select_asymptotic_pair,
    sparse_operator_norm_error,
)
from hamiltonian_resources.multiproduct import mpf_exponent_cost


def _record(
    *,
    method="trotter",
    order=2,
    model="transverse_field_ising",
    parameters=(
        ("coupling", 1.0),
        ("field", 3.0),
        ("periodic", False),
    ),
    partition="individual",
    schedule=None,
    max_step_size=0.25,
):
    return EmpiricalCalibrationRecord(
        calibration_id=f"test-{method}-{order}",
        key=EmpiricalCalibrationKey(
            method=method,
            formal_order=order,
            model=model,
            parameters=parameters,
            geometry="1d-chain",
            boundary_condition="open",
            partition=partition,
            schedule=schedule,
            formula=(
                "repository-suzuki-v1"
                if method == "trotter"
                else "ordered-individual-pauli-strang-mpf-v1"
            ),
        ),
        coefficient=AffineSizeCoefficient(0.5, 1.0),
        size_range=(4, 20),
        time_range=(1.0, 20.0),
        max_step_size=max_step_size,
        sample_sizes=(4, 8, 12),
        sample_times=(1.0, 8.0, 20.0),
        error_metric="spectral operator 2-norm",
        source="synthetic test calibration",
        source_digest="a" * 64,
        reference="synthetic test reference",
        review_status="reviewed",
        fit_diagnostics=(("r_squared", 0.999),),
    )


def test_reviewed_package_calibrations_match_the_reviewed_artifacts():
    project_root = Path(__file__).resolve().parents[1]
    accepted_path = (
        project_root
        / "docs"
        / "calibration_data"
        / "empirical_1d_v1_accepted.json"
    )
    fits_path = (
        project_root / "docs" / "calibration_data" / "empirical_1d_v1_fits.json"
    )
    source_digest = hashlib.sha256(accepted_path.read_bytes()).hexdigest()
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    fits = json.loads(fits_path.read_text(encoding="utf-8"))
    registry = default_empirical_calibrations()

    assert len(registry.records) == 20
    assert len(accepted["observations"]) == 20
    assert len(fits["fits"]) == 20
    assert {row.source_digest for row in registry.records} == {source_digest}
    observations_by_id = {
        row["calibration_id"]: row for row in accepted["observations"]
    }
    fits_by_id = {row["calibration_id"]: row for row in fits["fits"]}
    assert {
        (row.key.model, row.key.method, row.key.formal_order)
        for row in registry.records
    } == {
        *{
            (model, "trotter", order)
            for model in ("transverse_field_ising", "heisenberg_chain")
            for order in (2, 4, 6)
        },
        *{
            (model, "multiproduct", 2 * branch_count)
            for model in ("transverse_field_ising", "heisenberg_chain")
            for branch_count in range(2, 9)
        },
    }
    for row in registry.records:
        fit = fits_by_id[row.calibration_id]
        observations = observations_by_id[row.calibration_id]
        residuals = [
            observed - row.coefficient.at(system_size)
            for system_size, observed in zip(
                observations["sizes"],
                observations["coefficients"],
                strict=True,
            )
        ]
        assert row.review_status == "reviewed"
        assert row.coefficient.slope == pytest.approx(fit["slope"])
        assert row.coefficient.intercept == pytest.approx(fit["intercept"])
        assert math.sqrt(sum(value**2 for value in residuals) / len(residuals)) == (
            pytest.approx(fit["rmse"], abs=1e-9)
        )
        assert row.coefficient.at(row.size_range[0]) > 0
        assert row.coefficient.at(row.size_range[1]) > 0


def test_empirical_formula_has_exact_fixed_powers_and_affine_size_scaling():
    record = _record(order=4)
    baseline = evaluate_empirical_error(record, 8, 2.0, 10)

    assert evaluate_empirical_error(record, 8, 2.0, 20) == pytest.approx(
        baseline / 2**4
    )
    assert evaluate_empirical_error(record, 8, 4.0, 10) == pytest.approx(
        baseline * 2**5
    )
    assert record.coefficient.at(12) - record.coefficient.at(8) == pytest.approx(2.0)


def test_tagged_size_coefficients_preserve_fixed_formal_powers():
    power_record = replace(
        _record(order=18),
        coefficient=PowerSizeCoefficient(2e-9, 1.5),
    )
    offset_record = replace(
        power_record,
        coefficient=PowerPlusOffsetSizeCoefficient(1e-9, 1.5, 2e-9),
    )

    for record in (power_record, offset_record):
        baseline = evaluate_empirical_error(record, 8, 4.0, 20)
        assert evaluate_empirical_error(record, 8, 4.0, 40) == pytest.approx(
            baseline / 2**18
        )
        assert record.coefficient.at(12) >= record.coefficient.at(8)


def test_v2_registry_loads_tagged_models_and_enforces_reviewed_domain():
    raw = {
        "schema_version": "2.0",
        "calibrations": [
            {
                "calibration_id": "v2-power-order-18",
                "key": {
                    "method": "multiproduct",
                    "formal_order": 18,
                    "model": "transverse_field_ising",
                    "parameters": {
                        "coupling": 1.0,
                        "field": 3.0,
                        "periodic": False,
                    },
                    "geometry": "1d-chain",
                    "boundary_condition": "open",
                    "partition": None,
                    "schedule": "new",
                    "formula": "ordered-individual-pauli-strang-mpf-v1",
                },
                "coefficient": {
                    "model": "power",
                    "parameters": {"amplitude": 2e-9, "exponent": 1.5},
                },
                "size_range": [4, 12],
                "reviewed_size_max": 100,
                "time_range": [4.0, 12.0],
                "max_step_size": 0.1,
                "sample_sizes": list(range(4, 13)),
                "sample_times": [4.0, 8.0, 12.0],
                "external_validation_sizes": [11, 12],
                "fit_diagnostics": {"holdout_max_relative_error": 0.08},
                "stability_diagnostics": {"spread_at_reviewed_max": 0.20},
                "precision_backend": "flint",
                "precision_digits": 128,
                "error_metric": "spectral operator 2-norm",
                "source": "synthetic v2 source",
                "source_digest": "b" * 64,
                "reference": "synthetic v2 reference",
                "review_status": "reviewed",
            }
        ],
    }
    registry = EmpiricalCalibrationRegistry.from_json_data(raw)
    record = registry.records[0]

    assert isinstance(record.coefficient, PowerSizeCoefficient)
    assert record.coefficient.model_name == "power"
    assert record.reviewed_size_max == 100
    assert select_empirical_segments(record, 100, 100.0, 0.1).size_extrapolated
    with pytest.raises(UnsupportedEmpiricalCalibrationError, match="reviewed only"):
        select_empirical_segments(record, 101, 100.0, 0.1)


def test_canonical_json_digest_is_whitespace_and_line_ending_independent():
    parsed = json.loads('{"b": 2, "a": [1, 3]}')
    same = json.loads('{\r\n  "a": [1, 3],\r\n  "b": 2\r\n}')

    assert canonical_json_digest(parsed) == canonical_json_digest(same)


def test_empirical_inversion_ceiling_and_asymptotic_guard_are_consistent():
    formula_record = _record(order=2, max_step_size=10.0)
    estimate = select_empirical_segments(formula_record, 4, 2.0, 0.2)

    assert estimate.segments == estimate.formula_segments
    assert estimate.error <= 0.2
    if estimate.segments > 1:
        assert (
            evaluate_empirical_error(
                formula_record,
                4,
                2.0,
                estimate.segments - 1,
            )
            > 0.2
        )
    guarded = select_empirical_segments(
        replace(formula_record, max_step_size=0.05),
        4,
        2.0,
        0.2,
    )
    assert guarded.segments == 40
    assert guarded.active_constraint == "asymptotic-domain"


def test_empirical_domain_flags_extrapolation_and_rejects_invalid_sizes():
    record = _record()

    assert select_empirical_segments(record, 21, 25.0, 0.1).size_extrapolated
    assert select_empirical_segments(record, 21, 25.0, 0.1).time_extrapolated
    with pytest.raises(UnsupportedEmpiricalCalibrationError, match="N >= 4"):
        select_empirical_segments(record, 3, 1.0, 0.1)
    with pytest.raises(UnsupportedEmpiricalCalibrationError, match="nonpositive"):
        AffineSizeCoefficient(1.0, -10.0).at(4)


def test_empirical_segment_count_is_monotone_in_size_time_and_accuracy():
    record = _record(order=4, max_step_size=1.0)
    baseline = select_empirical_segments(record, 8, 4.0, 1e-2).segments

    assert select_empirical_segments(record, 12, 4.0, 1e-2).segments >= baseline
    assert select_empirical_segments(record, 8, 8.0, 1e-2).segments >= baseline
    assert select_empirical_segments(record, 8, 4.0, 1e-3).segments >= baseline
    larger = replace(record, coefficient=AffineSizeCoefficient(1.0, 1.0))
    assert select_empirical_segments(larger, 8, 4.0, 1e-2).segments >= baseline


def test_registry_matches_full_model_and_algorithm_identity_exactly():
    record = _record()
    registry = EmpiricalCalibrationRegistry((record,))
    hamiltonian = transverse_field_ising(8, coupling=1.0, field=3.0, periodic=False)
    key = EmpiricalCalibrationKey.for_hamiltonian(
        hamiltonian,
        method="trotter",
        formal_order=2,
        partition="individual",
        formula="repository-suzuki-v1",
    )

    assert registry.lookup(key) is record
    for changed in (
        replace(
            key,
            parameters=(
                ("coupling", 1.0),
                ("field", 2.0),
                ("periodic", False),
            ),
        ),
        replace(key, boundary_condition="periodic"),
        replace(key, partition="commuting"),
        replace(key, formal_order=4),
    ):
        with pytest.raises(UnsupportedEmpiricalCalibrationError):
            registry.lookup(changed)

    custom = PauliHamiltonian.from_terms(1, [("Z", 1.0)])
    with pytest.raises(UnsupportedEmpiricalCalibrationError, match="metadata"):
        EmpiricalCalibrationKey.for_hamiltonian(
            custom,
            method="trotter",
            formal_order=2,
            partition="individual",
            formula="repository-suzuki-v1",
        )


def test_builtin_model_metadata_is_normalized_and_distinct_from_display_name():
    tfim = transverse_field_ising(4, coupling=1, field=3, periodic=False)
    heisenberg = heisenberg_chain(4, coupling=1, field_z=0.3)

    assert tfim.model_metadata == HamiltonianModelMetadata.from_mapping(
        "transverse_field_ising",
        {"coupling": 1, "field": 3, "periodic": False},
        geometry="1d-chain",
        boundary_condition="open",
    )
    assert heisenberg.model_metadata.as_dict()["parameters"] == {
        "coupling": 1.0,
        "field_z": 0.3,
    }


def test_empirical_trotter_plan_is_nonrigorous_and_has_no_error_claim(monkeypatch):
    import hamiltonian_resources.planning as planning

    registry = EmpiricalCalibrationRegistry((_record(),))
    monkeypatch.setattr(planning, "default_empirical_calibrations", lambda: registry)
    plan = plan_simulation(
        transverse_field_ising(8, coupling=1, field=3, periodic=False),
        TrotterMethod(2, "empirical-operator-norm"),
        8.0,
        1e-3,
    )

    assert plan.error_analysis.sizing_estimate.category == "empirical"
    assert plan.error_analysis.claims == ()
    assert plan.error_analysis.ideal_algorithm_target.outcome == "unavailable"
    assert plan.error_metadata["bound_rigorous"] is False


def test_synthetic_m16_calibration_reaches_aggregate_resource_compilation(monkeypatch):
    import hamiltonian_resources.planning as planning

    metadata = transverse_field_ising(4).model_metadata
    record = _record(
        method="multiproduct",
        order=32,
        model=metadata.model,
        parameters=metadata.parameters,
        partition=None,
        schedule="new",
        max_step_size=1.0,
    )
    registry = EmpiricalCalibrationRegistry((record,))
    monkeypatch.setattr(planning, "default_empirical_calibrations", lambda: registry)
    plan = plan_simulation(
        transverse_field_ising(4),
        MultiproductMethod(16, error_method="empirical-operator-norm"),
        1.0,
        1e-2,
    )
    report = estimate_plan_resources(plan)

    assert plan.schedule_cost == mpf_exponent_cost(16)
    assert plan.schedule_cost.exponent_sum == 297
    assert plan.exponents is None
    assert plan.coefficients is None
    assert plan.lcu_structure is None
    assert report.resources.cnot_count > 0
    assert report.resources.t_count > 0
    assert any(
        "does not imply an implementable" in assumption
        for assumption in report.resource_provenance.assumptions
    )
    with pytest.raises(ValueError, match="aggregate MPF resource cost only"):
        build_simulation_circuit(plan)


def test_empirical_benchmark_rows_round_trip_with_schema2_extensions(
    monkeypatch,
    tmp_path,
):
    import hamiltonian_resources.planning as planning

    registry = EmpiricalCalibrationRegistry((_record(),))
    monkeypatch.setattr(planning, "default_empirical_calibrations", lambda: registry)
    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            "transverse_field_ising",
            {"coupling": 1.0, "field": 3.0, "periodic": False},
        ),
        system_sizes=[8],
        target_errors=[1e-3],
        time=TimeScaling("fixed", 8.0),
        fixed_system_size=8,
        methods=[TrotterMethod(2, "empirical-operator-norm")],
    )
    frame = run_benchmark(config)
    row = frame.iloc[0]

    assert row["status"] == "ok"
    assert row["estimate_category"] == "empirical"
    assert row["error_policy"] == "empirical-operator-norm"
    assert row["empirical_calibration_id"] == "test-trotter-2"
    assert not row["bound_rigorous"]
    _, csv_path, _ = save_benchmark(frame, config, output_root=tmp_path)
    loaded = load_benchmark(csv_path)
    assert loaded.iloc[0]["empirical_calibration_id"] == "test-trotter-2"
    figure = plot_benchmark(
        loaded,
        sweep="system-size",
        metric="t_count",
        certification_policy="unconstrained",
    )
    assert len(figure.axes[0].lines) == 1


def test_unsupported_empirical_calibration_is_an_actionable_failure_row(
    monkeypatch,
):
    import hamiltonian_resources.planning as planning

    registry = EmpiricalCalibrationRegistry((_record(),))
    monkeypatch.setattr(planning, "default_empirical_calibrations", lambda: registry)
    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            "transverse_field_ising",
            {"coupling": 1.0, "field": 3.0, "periodic": False},
        ),
        system_sizes=[8],
        time=TimeScaling("fixed", 8.0),
        methods=[TrotterMethod(4, "empirical-operator-norm")],
    )

    row = run_benchmark(config, sweeps="system-size").iloc[0]
    assert row["status"] == "error"
    assert row["error_type"] == "UnsupportedEmpiricalCalibrationError"
    assert "no reviewed empirical operator-norm calibration" in row["error_message"]


@pytest.mark.parametrize("schedule", ("new", "legacy"))
@pytest.mark.parametrize("m", range(2, 16))
def test_registered_mpf_schedule_cost_is_the_exact_exponent_sum(schedule, m):
    cost = mpf_exponent_cost(m, schedule=schedule)

    assert cost.exponents is not None
    assert cost.exponent_sum == sum(cost.exponents)
    assert cost.source == "registered-exact"
    assert cost.explicit_schedule_available


def test_mpf_schedule_cost_extrapolates_only_new_aggregate_cost():
    cost = mpf_exponent_cost(16, schedule="new")

    assert cost.exponent_sum == 297
    assert cost.source == "extrapolated-0.418-m2-log-m"
    assert cost.exponents is None
    assert not cost.explicit_schedule_available
    with pytest.raises(ValueError, match="legacy"):
        mpf_exponent_cost(16, schedule="legacy")
    hamiltonian = transverse_field_ising(2)
    with pytest.raises(ValueError, match="between 2 and 15"):
        multiproduct_coefficients(16)
    with pytest.raises(ValueError, match="between 2 and 15"):
        mpf_lcu_structure(16)
    with pytest.raises(ValueError, match="between 2 and 15"):
        estimate_mpf_error(hamiltonian, 1.0, 1, 16)


def test_calibration_analysis_helpers_enforce_formal_order_and_fit_affine_law():
    pair = select_asymptotic_pair(
        ((10, 1.0), (20, 1 / 16), (40, 1 / 256)),
        4,
    )
    fit = fit_affine_size_coefficient((4, 6, 8, 10), (9.0, 13.0, 17.0, 21.0))

    assert pair.running_exponent == pytest.approx(4.0)
    assert effective_power(1.0, 10.0, 1 / 16, 20.0) == pytest.approx(4.0)
    assert observed_error_coefficient(0.25, 4, 2.0, 2) == pytest.approx(0.5)
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no consecutive"):
        select_asymptotic_pair(((10, 1.0), (20, 0.5)), 4)


def test_dense_and_sparse_operator_norm_kernels_agree_on_two_qubits():
    hamiltonian = transverse_field_ising(2, coupling=1, field=3, periodic=False)
    dense = dense_operator_norm_error(
        hamiltonian,
        0.5,
        3,
        algorithm="multiproduct",
        formal_order=4,
    )
    sparse = sparse_operator_norm_error(
        hamiltonian,
        0.5,
        3,
        algorithm="multiproduct",
        formal_order=4,
        tolerance=1e-11,
        max_iterations=100,
        restarts=3,
    )

    assert sparse.value == pytest.approx(dense.value, rel=1e-7, abs=1e-12)


def test_prestudy_m3_affine_equation_is_only_a_qualitative_sanity_check():
    prestudy = AffineSizeCoefficient(0.9287, -0.402)

    assert prestudy.at(8) == pytest.approx(7.0276)
    assert all("prestudy" not in row.calibration_id for row in default_empirical_calibrations().records)

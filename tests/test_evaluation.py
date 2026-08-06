import pytest

from hamiltonian_resources import (
    MultiproductMethod,
    QSVTMethod,
    TrotterMethod,
    estimate_plan_resources,
    estimate_resources,
    plan_simulation,
    transverse_field_ising,
)
from hamiltonian_resources.benchmark import (
    _EvaluationConfig,
    estimate_resources_analytically,
)


def test_single_point_evaluation_selects_parameters_once(monkeypatch):
    import hamiltonian_resources.evaluation as evaluation

    calls = 0
    original = evaluation.plan_simulation

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "plan_simulation", counted)
    report = evaluation.estimate_resources(
        transverse_field_ising(2, field=0.7),
        MultiproductMethod(3),
        0.1,
        1e-2,
    )

    assert calls == 1
    assert report.selected_parameters["mpf_segments"] == 1


def test_analytical_backend_consumes_existing_plan_without_selection(monkeypatch):
    import hamiltonian_resources.evaluation as evaluation

    plan = plan_simulation(
        transverse_field_ising(2, field=0.7),
        TrotterMethod(2),
        0.1,
        1e-2,
    )

    def fail(*args, **kwargs):
        raise AssertionError("parameter selection was repeated")

    monkeypatch.setattr(evaluation, "plan_simulation", fail)
    report = evaluation.estimate_plan_resources(plan)

    assert report.plan is plan
    assert report.resources.rotation_count == 5


def test_report_convenience_properties_are_derived_from_plan():
    report = estimate_resources(
        transverse_field_ising(2, field=0.7),
        QSVTMethod(),
        0.1,
        1e-2,
    )

    assert report.logical_counts == report.plan.logical_counts
    assert report.error_budget is report.plan.error_budget
    assert report.selected_parameters == report.plan.selected_parameters
    assert report.error_metadata == report.plan.error_metadata
    copied = report.selected_parameters
    copied["qsvt_degree"] = -1
    assert report.plan.degree == 3


@pytest.mark.parametrize(
    ("method", "algorithm", "config"),
    [
        (TrotterMethod(2), "trotter", _EvaluationConfig(time=0.1, target_error=1e-2)),
        (
            MultiproductMethod(3),
            "multiproduct",
            _EvaluationConfig(time=0.1, target_error=1e-2, mpf_m=3),
        ),
        (QSVTMethod(), "qsvt", _EvaluationConfig(time=0.1, target_error=1e-2)),
    ],
)
def test_new_report_preserves_legacy_analytical_resources(method, algorithm, config):
    hamiltonian = transverse_field_ising(2, field=0.7)
    report = estimate_resources(
        hamiltonian,
        method,
        config.time,
        config.target_error,
        synthesis_error_fraction=config.synthesis_error_fraction,
        trotter_partition=config.trotter_partition,
    )
    legacy = estimate_resources_analytically(hamiltonian, config, algorithm)

    assert report.resources == legacy


def test_resource_provenance_is_backend_specific_not_part_of_plan():
    plan = plan_simulation(
        transverse_field_ising(2, field=0.7),
        QSVTMethod(),
        0.1,
        1e-2,
    )
    report = estimate_plan_resources(plan)

    assert report.resource_provenance.backend == "analytical"
    assert report.resource_provenance.model == "structured-analytical-v1"
    assert any("multiplex" in assumption for assumption in report.resource_provenance.assumptions)
    assert not hasattr(plan, "resource_provenance")

import numpy as np
import pytest

from hamiltonian_resources import (
    CalibrationMetadata,
    InitialStateRecord,
    MetricObservation,
    MultiproductMethod,
    QSVTMethod,
    StateObservationContext,
    compare_with_exact,
    estimate_resources,
    synthesize_hamsim_phases,
    transverse_field_ising,
)
from hamiltonian_resources.error_models import (
    good_subspace_leakage_bound,
    oaa_good_block_error_bound,
    repeated_block_encoding_error_bound,
)


def test_proxy_sizing_succeeds_without_certifying_either_target():
    report = estimate_resources(
        transverse_field_ising(2, field=0.7),
        MultiproductMethod(3, error_method="legacy-w2-proxy"),
        0.2,
        1e-3,
    )

    assert report.parameter_selection_succeeded
    assert report.error_analysis.sizing_estimate.category == "proxy"
    assert report.error_analysis.sizing_estimate.certification == "nonrigorous"
    assert not report.ideal_algorithm_target_certified
    assert not report.implemented_circuit_target_certified
    assert report.error_analysis.claims == ()


def test_report_keeps_ideal_and_implemented_target_assessments_separate():
    report = estimate_resources(
        transverse_field_ising(2, field=0.7),
        MultiproductMethod(2),
        0.5,
        1e-2,
    )

    assert report.parameter_selection_succeeded
    assert report.ideal_algorithm_target_certified
    assert not report.implemented_circuit_target_certified
    assert report.error_analysis.ideal_algorithm_target.outcome == "certified"
    assert report.error_analysis.implemented_circuit_target.outcome == "not_met"
    assert report.error_analysis.claim_for_scope("ideal-mpf") is not None
    assert (
        report.error_analysis.claim_for_scope(
            "one-segment-amplified-good-block"
        )
        is not None
    )
    assert (
        report.error_analysis.claim_for_scope(
            "repeated-shared-ancilla-good-block"
        )
        is not None
    )


def test_qsvt_ideal_polynomial_is_certified_but_floating_circuit_is_not():
    report = estimate_resources(
        transverse_field_ising(2, field=0.7),
        QSVTMethod(),
        0.1,
        1e-2,
    )

    assert report.parameter_selection_succeeded
    assert report.ideal_algorithm_target_certified
    assert not report.implemented_circuit_target_certified
    assert report.error_analysis.implemented_circuit_target.outcome == "unavailable"
    assert report.error_metadata["bound_scope"] == "ideal-qsvt-oaa-good-block"
    assert not report.error_metadata["circuit_bound_rigorous"]


def test_state_metrics_retain_state_and_postselection_context():
    initial_state = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    comparison = compare_with_exact(
        transverse_field_ising(1),
        0.05,
        method="multiproduct",
        initial_state=initial_state,
        mpf_m=2,
    )

    assert comparison["fidelity"] > 0.999
    assert len(comparison.observations) == 3
    assert {observation.metric for observation in comparison.observations} == {
        "phase-aligned-state-2-norm",
        "pure-state-fidelity",
        "good-subspace-probability",
    }
    for observation in comparison.observations:
        assert isinstance(observation, MetricObservation)
        assert isinstance(observation.context, StateObservationContext)
        assert isinstance(observation.context.initial_state, InitialStateRecord)
        assert observation.context.initial_state.source == "user-supplied"
        assert observation.context.initial_state.digest


def test_evaluation_report_can_attach_observations_without_changing_claims():
    hamiltonian = transverse_field_ising(1)
    report = estimate_resources(
        hamiltonian,
        MultiproductMethod(2),
        0.05,
        1e-2,
    )
    comparison = compare_with_exact(
        hamiltonian,
        0.05,
        method="multiproduct",
        mpf_m=2,
    )
    combined = report.with_observations(comparison.observations)

    assert report.observations == ()
    assert combined.error_analysis.observations == comparison.observations
    assert combined.error_analysis.claims == report.error_analysis.claims


def test_phase_residuals_are_grid_observations_not_error_claims():
    phases = synthesize_hamsim_phases(0.2, 1e-2)

    assert len(phases.observations) == 2
    for observation in phases.observations:
        assert isinstance(observation.context, CalibrationMetadata)
        assert observation.context.domain == (-1.0, 1.0)
        assert observation.context.sample_count == 2049
        assert observation.scope == "floating-pyqsp-phase-reconstruction"


@pytest.mark.parametrize("delta", [0.0, 1e-6, 0.1, 1.0])
def test_oaa_and_leakage_envelopes_are_consistent(delta):
    eta = oaa_good_block_error_bound(delta)
    leakage = good_subspace_leakage_bound(eta)

    assert eta >= delta
    if eta <= 1:
        assert leakage == pytest.approx(np.sqrt(2 * eta - eta**2))
    else:
        assert leakage is None


def test_repeated_block_encoding_bound_has_an_explicit_validity_gate():
    assert repeated_block_encoding_error_bound(0.01, 3) == pytest.approx(0.36)
    assert repeated_block_encoding_error_bound(1.01, 3) is None
    assert repeated_block_encoding_error_bound(1.01, 1) == pytest.approx(1.01)

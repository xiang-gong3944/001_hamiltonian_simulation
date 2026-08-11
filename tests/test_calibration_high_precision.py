import pytest

from hamiltonian_resources import transverse_field_ising
from hamiltonian_resources.calibration_high_precision import (
    adaptive_mpf_operator_norm_error,
    recommended_initial_digits,
)
from hamiltonian_resources.calibration_study import dense_operator_norm_error


pytest.importorskip("mpmath")


def test_high_precision_mpf_reproduces_dense_low_order_error():
    hamiltonian = transverse_field_ising(2, coupling=1.0, field=3.0)
    expected = dense_operator_norm_error(
        hamiltonian,
        2.0,
        4,
        algorithm="multiproduct",
        formal_order=4,
    )
    actual = adaptive_mpf_operator_norm_error(
        hamiltonian,
        2.0,
        4,
        2,
        initial_digits=48,
        digit_increment=24,
        max_digits=72,
        relative_tolerance=1e-12,
    )

    assert actual.converged
    assert actual.formal_order == 4
    assert actual.attempted_digits == (48, 72)
    assert actual.value == pytest.approx(expected.value, rel=1e-11)
    assert actual.eigensystem_residual < 1e-40


def test_recommended_precision_grows_with_order_and_step_refinement():
    low_order = recommended_initial_digits(2, 4.0, 20)
    high_order = recommended_initial_digits(15, 4.0, 20)
    finer_high_order = recommended_initial_digits(15, 4.0, 32)

    assert low_order == 64
    assert high_order > low_order
    assert finer_high_order > high_order

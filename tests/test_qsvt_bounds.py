import numpy as np
import pytest
from scipy.special import jv

from hamiltonian_resources import estimate_qsvt_degree, qsvt_polynomial_error_bound


def _unscaled_jacobi_anger_polynomials(alpha_time, degree, grid):
    truncation_order = (degree - 1) // 2
    cosine_coefficients = np.zeros(degree)
    cosine_coefficients[0] = jv(0, alpha_time)
    for k in range(1, truncation_order + 1):
        cosine_coefficients[2 * k] = 2 * (-1) ** k * jv(2 * k, alpha_time)
    sine_coefficients = np.zeros(degree + 1)
    for k in range(truncation_order + 1):
        sine_coefficients[2 * k + 1] = 2 * (-1) ** k * jv(
            2 * k + 1,
            alpha_time,
        )
    return (
        np.polynomial.chebyshev.chebval(grid, cosine_coefficients),
        np.polynomial.chebyshev.chebval(grid, sine_coefficients),
    )


@pytest.mark.parametrize(
    ("alpha_time", "epsilon"),
    [(0.2, 1e-2), (1.7, 1e-3), (5.0, 1e-4)],
)
def test_qsvt_bound_dominates_direct_polynomial_and_oaa_errors(alpha_time, epsilon):
    degree = estimate_qsvt_degree(alpha_time, epsilon)
    estimate = qsvt_polynomial_error_bound(alpha_time, epsilon, degree)
    grid = np.cos(np.pi * np.arange(8193) / 8192)
    cosine, sine = _unscaled_jacobi_anger_polynomials(alpha_time, degree, grid)
    exact_cosine = np.cos(alpha_time * grid)
    exact_sine = np.sin(alpha_time * grid)
    exact_evolution = np.exp(-1j * alpha_time * grid)
    polynomial = estimate.scale * (cosine - 1j * sine)
    amplified = 1.5 * polynomial - 0.5 * polynomial * np.abs(polynomial) ** 2

    assert (
        np.max(np.abs(cosine - exact_cosine))
        <= estimate.cosine_tail_bound + 1e-14
    )
    assert np.max(np.abs(sine - exact_sine)) <= estimate.sine_tail_bound + 1e-14
    assert (
        np.max(np.abs(polynomial - exact_evolution))
        <= estimate.polynomial_error + 1e-14
    )
    assert (
        np.max(np.abs(amplified - exact_evolution))
        <= estimate.amplified_good_block_error + 1e-14
    )


def test_qsvt_degree_convention_matches_first_omitted_parity_terms():
    degree = estimate_qsvt_degree(2.0, 1e-3)
    estimate = qsvt_polynomial_error_bound(2.0, 1e-3, degree)

    assert degree % 2 == 1
    assert estimate.truncation_order == (degree - 1) // 2
    assert 2 * estimate.truncation_order == degree - 1
    assert 2 * estimate.truncation_order + 1 == degree
    assert degree + 1 == 2 * estimate.truncation_order + 2
    assert degree + 2 == 2 * estimate.truncation_order + 3


def test_qsvt_polynomial_bound_does_not_include_phase_residuals():
    estimate = qsvt_polynomial_error_bound(0.2, 1e-2, 3)
    source_budget = 1e-2 / 18
    expected = (1 - estimate.scale) + estimate.scale * (
        estimate.cosine_tail_bound + estimate.sine_tail_bound
    )

    assert estimate.scale == pytest.approx(1 - source_budget)
    assert estimate.polynomial_error == pytest.approx(expected)

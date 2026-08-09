import math

import numpy as np
import pytest

from hamiltonian_resources import heisenberg_chain, transverse_field_ising
from hamiltonian_resources._mizuta_bch import (
    certify_lemma10_tail,
    lemma9_direct_log_coefficient,
    lemma9_log_coefficients,
    lemma10_majorant,
    refined_mizuta_remainder,
    mizuta_schedule_weights,
)
from hamiltonian_resources.trotter import pauli_nested_commutator_bounds


@pytest.mark.parametrize(
    "hamiltonian",
    [
        transverse_field_ising(4, coupling=1.0, field=3.0),
        heisenberg_chain(4, coupling=1.0, field_z=0.0),
    ],
)
def test_strang_schedule_half_steps_have_total_absolute_weight_one(hamiltonian):
    weights = mizuta_schedule_weights(hamiltonian, suzuki_order=2)
    locality = pauli_nested_commutator_bounds(hamiltonian, 2)

    assert np.allclose(weights.group_weights, 1.0, rtol=0.0, atol=2e-15)
    assert weights.maximum_group_weight == pytest.approx(1.0, abs=2e-15)
    assert weights.locality_k == locality.locality_k
    assert weights.weighted_extensiveness == pytest.approx(
        locality.extensiveness_g,
        rel=2e-15,
    )


@pytest.mark.parametrize("locality_k, g_alpha", [(1, 0.25), (2, 3.0), (4, 1.125)])
def test_lemma9_recurrence_matches_order_resolved_finite_sum(locality_k, g_alpha):
    recurrent = lemma9_log_coefficients(locality_k, g_alpha, 16)

    for order in range(1, 17):
        direct = lemma9_direct_log_coefficient(locality_k, g_alpha, order)
        assert math.exp(recurrent[order]) == pytest.approx(
            math.exp(direct),
            rel=2e-13,
        )


def test_lemma9_zero_extensiveness_has_no_positive_order_coefficients():
    coefficients = lemma9_log_coefficients(2, 0.0, 8)

    assert coefficients[0] == 0.0
    assert all(value == -math.inf for value in coefficients[1:])


def _brute_force_lemma10(locality_k, g_alpha, p0, maximum_order):
    majorant = lemma10_majorant(locality_k, g_alpha, p0, maximum_order)
    g_values = np.array(
        [0.0 if value == -math.inf else math.exp(value) for value in majorant.log_g]
    )
    c_rows = [g_values]
    for _ in range(maximum_order):
        following = np.zeros(maximum_order + 1)
        for order in range(1, maximum_order + 1):
            for phi_order in range(1, min(p0, order - 1) + 1):
                following[order] += (
                    2
                    * locality_k
                    * (order - phi_order)
                    * g_values[phi_order]
                    * c_rows[-1][order - phi_order]
                )
        if not np.any(following):
            break
        c_rows.append(following)
    d_values = sum(row / math.factorial(index) for index, row in enumerate(c_rows))
    b_values = np.zeros(maximum_order + 1)
    b_values[0] = 1.0
    for order in range(1, maximum_order + 1):
        b_values[order] = sum(
            generator_order
            * d_values[generator_order]
            * b_values[order - generator_order]
            for generator_order in range(1, order + 1)
        ) / order
    return c_rows, d_values, b_values


def test_lemma10_recurrences_match_brute_force_low_order_compositions():
    majorant = lemma10_majorant(2, 0.7, 7, 12)
    c_rows, d_values, b_values = _brute_force_lemma10(2, 0.7, 7, 12)

    for row_index, row in enumerate(c_rows):
        actual = np.array(
            [0.0 if value == -math.inf else math.exp(value) for value in majorant.log_c_by_adjoint[row_index]]
        )
        assert actual == pytest.approx(row, rel=2e-12, abs=1e-15)
    actual_d = [0.0 if value == -math.inf else math.exp(value) for value in majorant.log_d]
    actual_b = [0.0 if value == -math.inf else math.exp(value) for value in majorant.log_b]
    assert actual_d == pytest.approx(d_values, rel=2e-12, abs=1e-15)
    assert actual_b == pytest.approx(b_values, rel=2e-12, abs=1e-15)


def test_lemma10_preserves_strang_parity_and_subsystem_insertion_bound():
    majorant = lemma10_majorant(2, 3.0, 11, 24)

    assert all(majorant.log_g[order] == -math.inf for order in range(2, 12, 2))
    # The local-insertion/subsystem-difference majorant is the same G_s; it has
    # no extra schedule multiplicity and thus inherits the exact parity zeros.
    assert majorant.log_g[1] == pytest.approx(math.log(3.0), rel=1e-15)


def test_scalar_flow_certificate_satisfies_radius_and_integral_conditions():
    certificate = certify_lemma10_tail(2, 0.4, 7, 0.005, maximum_order=46)

    assert certificate is not None
    assert 0.005 < certificate.rho < certificate.flow_radius
    assert certificate.integral_lower_bound >= 1.0
    assert certificate.log_b_at_rho_upper > 0.0


@pytest.mark.parametrize("radius_fraction", [0.2, 0.75, 0.92])
def test_certified_tail_dominates_direct_generation_through_four_q(radius_fraction):
    num_qubits = 5
    locality_k = 2
    g_alpha = 0.4
    p0 = 7
    probe = refined_mizuta_remainder(num_qubits, locality_k, g_alpha, p0, 0.001)
    assert probe is not None and probe.certificate is not None
    x = radius_fraction * probe.certificate.rho
    certified = refined_mizuta_remainder(num_qubits, locality_k, g_alpha, p0, x)
    assert certified is not None

    q = certified.explicit_order
    direct_a = lemma9_log_coefficients(locality_k, g_alpha, 4 * q)
    direct_b = lemma10_majorant(locality_k, g_alpha, p0, 4 * q).log_b
    direct = 0.0
    for order in range(p0 + 1, 4 * q + 1):
        coefficient = sum(
            0.0 if values[order] == -math.inf else math.exp(values[order])
            for values in (direct_a, direct_b)
        )
        direct += coefficient * x**order
    direct *= num_qubits

    assert certified.total >= direct


def test_refined_remainder_is_monotone_in_time_argument():
    values = [
        refined_mizuta_remainder(5, 2, 0.4, 7, x)
        for x in (0.001, 0.002, 0.004)
    ]
    assert all(value is not None for value in values)
    totals = [value.total for value in values if value is not None]
    assert totals == sorted(totals)

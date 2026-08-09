import math

import numpy as np
import pytest

from hamiltonian_resources import heisenberg_chain, transverse_field_ising
from hamiltonian_resources._mizuta_bch import (
    lemma9_direct_log_coefficient,
    lemma9_log_coefficients,
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

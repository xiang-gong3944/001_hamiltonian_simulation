import math

import pytest

from hamiltonian_resources import transverse_field_ising
from hamiltonian_resources.multiproduct import (
    legacy_w2_proxy_error,
    legacy_w2_proxy_segments,
)
from hamiltonian_resources.trotter import suzuki_commutator_bounds


@pytest.mark.parametrize("m", [2, 3, 5, 7])
def test_legacy_w2_proxy_exactly_reproduces_historical_rule(m):
    hamiltonian = transverse_field_ising(4, field=3.0)
    time = 1.7
    target_error = 9e-5
    _, w2 = suzuki_commutator_bounds(hamiltonian)
    alpha_effective = min(hamiltonian.alpha, w2 ** (1 / 3))
    formal_order = 2 * m
    historical_segments = max(
        1,
        math.ceil(
            (
                (alpha_effective * time) ** (formal_order + 1)
                / target_error
            )
            ** (1 / formal_order)
        ),
    )

    segments = legacy_w2_proxy_segments(hamiltonian, time, target_error, m)
    error, prefactor = legacy_w2_proxy_error(hamiltonian, time, m, segments)

    assert segments == historical_segments
    assert prefactor == pytest.approx(alpha_effective ** (formal_order + 1))
    assert error == pytest.approx(
        (alpha_effective * time) ** (formal_order + 1)
        / segments**formal_order
    )


def test_legacy_w2_proxy_is_explicitly_noncertifying_metadata():
    from hamiltonian_resources import BenchmarkConfig, MultiproductMethod, run_benchmark

    frame = run_benchmark(
        BenchmarkConfig(system_sizes=[2], methods=[MultiproductMethod(3)]),
        sweeps="system-size",
    )
    row = frame.iloc[0]

    assert row["bound_method"] == "legacy-w2-proxy"
    assert not bool(row["bound_rigorous"])

import numpy as np

from hamiltonian_resources import PauliHamiltonian, transverse_field_ising


def test_tfim_term_count_and_norm():
    hamiltonian = transverse_field_ising(4, coupling=1.0, field=0.5)
    assert hamiltonian.term_count == 7
    assert np.isclose(hamiltonian.alpha, 5.0)
    assert hamiltonian.matrix().shape == (16, 16)


def test_duplicate_terms_are_combined():
    hamiltonian = PauliHamiltonian.from_terms(1, [("X", 1), ("X", -0.25)])
    assert hamiltonian.terms == (("X", 0.75),)


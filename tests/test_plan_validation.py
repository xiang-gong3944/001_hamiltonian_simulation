import pytest

from hamiltonian_resources import (
    MultiproductMethod,
    QSVTMethod,
    TrotterMethod,
    build_simulation_circuit,
    compare_plan_with_exact,
    plan_simulation,
    transverse_field_ising,
)


@pytest.mark.parametrize(
    "method",
    [TrotterMethod(2), MultiproductMethod(2), QSVTMethod()],
)
def test_reference_circuit_metadata_is_a_derived_plan_view(method):
    plan = plan_simulation(
        transverse_field_ising(1, field=0.7),
        method,
        0.05,
        2e-2,
    )
    circuit = build_simulation_circuit(plan)
    metadata = circuit.metadata

    assert metadata["method_id"] == plan.method.method_id
    assert metadata["selected_parameters"] == plan.selected_parameters
    assert metadata["logical_operation_counts"] == plan.logical_counts.as_dict()
    assert metadata["error_metadata"] == plan.error_metadata
    assert metadata["resource_provenance"]["backend"] == "qiskit-reference"
    assert metadata["resource_provenance"]["model"] != "structured-analytical-v1"

    metadata["selected_parameters"][next(iter(metadata["selected_parameters"]))] = -1
    assert -1 not in plan.selected_parameters.values()


@pytest.mark.parametrize(
    ("method", "minimum_fidelity"),
    [
        (TrotterMethod(2), 0.99),
        (MultiproductMethod(2), 0.99),
        (QSVTMethod(), 0.999),
    ],
)
def test_plan_validation_reuses_selected_structure(method, minimum_fidelity):
    plan = plan_simulation(
        transverse_field_ising(1, field=0.7),
        method,
        0.05,
        2e-2,
    )
    result = compare_plan_with_exact(plan)

    assert result["method"] == method.family
    assert result["fidelity"] > minimum_fidelity
    assert result["success_probability"] > 0.99


def test_plan_validation_rejects_nonplans():
    with pytest.raises(TypeError, match="plan"):
        compare_plan_with_exact(object())

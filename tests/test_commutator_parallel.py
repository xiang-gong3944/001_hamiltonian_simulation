import pytest

from hamiltonian_resources import (
    BenchmarkConfig,
    QSVTMethod,
    TrotterMethod,
    estimate_suzuki_error,
    pauli_nested_commutator_bounds,
    run_benchmark,
    select_mpf_segments,
    transverse_field_ising,
)


def test_parallel_trotter_bound_matches_serial(monkeypatch):
    import hamiltonian_resources.trotter as trotter

    monkeypatch.setattr(trotter, "_TROTTER_PARALLEL_WORK_THRESHOLD", 1)
    hamiltonian = transverse_field_ising(4, field=0.7)

    serial = estimate_suzuki_error(hamiltonian, 0.2, order=6)
    parallel = estimate_suzuki_error(hamiltonian, 0.2, order=6, workers=2)

    assert parallel.error == pytest.approx(serial.error, rel=1e-12, abs=1e-14)
    assert parallel.prefactor == pytest.approx(serial.prefactor, rel=1e-12, abs=1e-14)
    assert (
        parallel.order,
        parallel.partition,
        parallel.group_count,
        parallel.method,
        parallel.rigorous,
    ) == (
        serial.order,
        serial.partition,
        serial.group_count,
        serial.method,
        serial.rigorous,
    )


def test_parallel_pauli_recurrence_matches_serial(monkeypatch):
    import hamiltonian_resources._commutator_execution as execution

    monkeypatch.setattr(execution, "_PARALLEL_WORK_THRESHOLD", 1)
    hamiltonian = transverse_field_ising(6, field=0.7)
    pauli_nested_commutator_bounds.cache_clear()
    serial = pauli_nested_commutator_bounds(hamiltonian, 8)
    pauli_nested_commutator_bounds.cache_clear()
    parallel = pauli_nested_commutator_bounds(hamiltonian, 8, workers=2)

    assert parallel.values == pytest.approx(serial.values, rel=1e-12, abs=1e-14)
    assert parallel.state_counts == serial.state_counts
    assert parallel.max_exact_order == serial.max_exact_order
    assert parallel.fallback_reason == serial.fallback_reason
    assert parallel.used_locality_fallback == serial.used_locality_fallback


@pytest.mark.parametrize("workers", [0, -1])
def test_public_parallel_apis_reject_nonpositive_workers(workers):
    hamiltonian = transverse_field_ising(2)
    with pytest.raises(ValueError, match="workers"):
        estimate_suzuki_error(hamiltonian, 0.1, order=2, workers=workers)
    with pytest.raises(ValueError, match="workers"):
        pauli_nested_commutator_bounds(hamiltonian, 3, workers=workers)


def test_parallel_benchmark_preserves_row_and_callback_order():
    events = []
    config = BenchmarkConfig(
        system_sizes=[2],
        methods=[QSVTMethod(), TrotterMethod(4)],
    )

    frame = run_benchmark(
        config,
        sweeps="system-size",
        progress=events.append,
        workers=2,
    )

    assert frame["method_id"].tolist() == ["qsvt", "trotter-p4"]
    assert [event.method_id for event in events] == ["qsvt", "trotter-p4"]


def test_trotter_progress_reports_known_chunk_totals(monkeypatch):
    import hamiltonian_resources.trotter as trotter

    monkeypatch.setattr(trotter, "_TROTTER_PARALLEL_WORK_THRESHOLD", 1)
    events = []

    estimate_suzuki_error(
        transverse_field_ising(4, field=0.7),
        0.2,
        order=6,
        progress=events.append,
    )

    assert events
    assert all(event.family == "trotter" for event in events)
    assert all(event.total is not None for event in events)
    for key in {(event.phase, event.commutator_order) for event in events}:
        completed = [
            event.completed for event in events if (event.phase, event.commutator_order) == key
        ]
        assert completed == sorted(completed)


def test_adaptive_mpf_progress_never_invents_a_total():
    events = []
    target_error = 1e-3

    select_mpf_segments(
        transverse_field_ising(3, field=0.7),
        0.01,
        target_error,
        2,
        method="mizuta2026-commutator-ideal-rigorous",
        progress=events.append,
    )

    assert events
    assert all(event.family == "multiproduct" for event in events)
    assert all(event.total is None for event in events)
    assert all(event.system_qubits == 3 for event in events)
    assert all(event.target_error == target_error for event in events)
    candidates = [event.segment_candidate for event in events if event.phase == "segment-candidate"]
    assert candidates
    assert all(candidate is not None and candidate > 0 for candidate in candidates)


def test_tqdm_progress_is_rendered_to_stderr_only(capsys):
    run_benchmark(
        BenchmarkConfig(system_sizes=[2], methods=[QSVTMethod()]),
        sweeps="system-size",
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "benchmark" in captured.err

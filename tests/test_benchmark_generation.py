from dataclasses import replace

import pandas as pd

from hamiltonian_resources import (
    BENCHMARK_COLUMNS,
    METHOD_LABELS,
    ScalingBenchmarkConfig,
    generate_benchmark_sweep,
)


def test_generation_includes_all_fixed_methods_with_stable_schema(tmp_path):
    config = ScalingBenchmarkConfig(
        system_qubit_values=(2,),
        output_directory=tmp_path,
    )
    frame = generate_benchmark_sweep(config, "system-size")

    assert tuple(frame.columns) == BENCHMARK_COLUMNS
    assert tuple(frame["method_label"]) == METHOD_LABELS
    assert set(frame["status"]) == {"ok"}
    assert frame[["t_count", "cnot_count"]].notna().all().all()


def test_generation_records_one_method_failure_without_omission(monkeypatch, tmp_path):
    import hamiltonian_resources.benchmark_suite as suite

    original = suite.estimate_resources_analytically

    def fail_mpf_five(hamiltonian, config, algorithm):
        if algorithm == "multiproduct" and config.mpf_m == 5:
            raise RuntimeError("deliberate failure")
        return original(hamiltonian, config, algorithm)

    monkeypatch.setattr(suite, "estimate_resources_analytically", fail_mpf_five)
    config = replace(
        ScalingBenchmarkConfig(),
        system_qubit_values=(2,),
        output_directory=tmp_path,
    )
    frame = generate_benchmark_sweep(config, "system-size")

    assert len(frame) == 8
    assert (frame["status"] == "ok").sum() == 7
    failure = frame[frame["status"] == "error"].iloc[0]
    assert failure["method_label"] == "MPF m=5"
    assert pd.isna(failure["t_count"])
    assert "deliberate failure" in failure["error_message"]

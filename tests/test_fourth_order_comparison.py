import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from hamiltonian_resources import (
    FOURTH_ORDER_COMPARISON_COLUMNS,
    FourthOrderComparisonConfig,
    create_fourth_order_comparison_figure,
    generate_fourth_order_comparison,
    plot_fourth_order_comparison,
    save_fourth_order_comparison,
)


@pytest.fixture(scope="module")
def comparison_config(tmp_path_factory):
    return FourthOrderComparisonConfig(
        system_qubit_values=(3,),
        target_error_values=(1e-2, 1e-3),
        fixed_system_qubits_for_error_sweep=3,
        output_directory=tmp_path_factory.mktemp("fourth-order-comparison"),
        output_formats=("png",),
    )


@pytest.fixture(scope="module")
def size_frame(comparison_config):
    return generate_fourth_order_comparison(comparison_config, "system-size")


def test_comparison_retains_all_centers_and_required_machine_schema(size_frame):
    assert tuple(size_frame.columns) == FOURTH_ORDER_COMPARISON_COLUMNS
    assert len(size_frame) == 38
    assert set(size_frame["status"]) == {"ok"}
    assert size_frame[
        [
            "hamiltonian_decomposition_json",
            "ordered_exponentials_json",
            "specific_theorem_or_equation",
            "commutator_contributions_json",
            "diagnostic_message",
        ]
    ].notna().all().all()
    centers = size_frame[size_frame["bound_variant"] == "theorem-1-expanded-base"]
    assert centers.groupby("decomposition_case")["center_index_s"].nunique().to_dict() == {
        "multi-term": 21,
        "two-term": 11,
    }
    assert centers.groupby("decomposition_case")["is_centered_s"].sum().to_dict() == {
        "multi-term": 1,
        "two-term": 1,
    }
    assert centers.groupby("decomposition_case")["is_minimizing_s"].sum().to_dict() == {
        "multi-term": 1,
        "two-term": 1,
    }


def test_expected_appendix_m_ratios_and_center_effects_are_visible(size_frame):
    two_term = size_frame[size_frame["decomposition_case"] == "two-term"]
    two_appendix = two_term[two_term["bound_variant"] == "appendix-m-expanded"].iloc[0]
    two_s6 = two_term[
        (two_term["bound_variant"] == "theorem-1-expanded-base")
        & (two_term["center_index_s"] == 6)
    ].iloc[0]
    assert two_appendix["ratio_to_reference"] == pytest.approx(1.0)
    assert two_s6["ratio_to_reference"] == pytest.approx(1.0)
    assert two_s6["symbolic_prefactor_ratio_to_reference"] == pytest.approx(1.0)
    assert two_s6["one_step_coefficient_c5"] == pytest.approx(
        two_appendix["one_step_coefficient_c5"]
    )

    multi = size_frame[size_frame["decomposition_case"] == "multi-term"]
    appendix = multi[multi["bound_variant"] == "appendix-m-expanded"].iloc[0]
    s10 = multi[
        (multi["bound_variant"] == "theorem-1-expanded-base")
        & (multi["center_index_s"] == 10)
    ].iloc[0]
    centered = multi[
        (multi["bound_variant"] == "theorem-1-expanded-base")
        & (multi["is_centered_s"])
    ].iloc[0]
    minimum = multi[
        (multi["bound_variant"] == "theorem-1-expanded-base")
        & (multi["is_minimizing_s"])
    ].iloc[0]
    assert s10["ratio_to_reference"] == pytest.approx(1.0)
    assert s10["symbolic_prefactor_ratio_to_reference"] == pytest.approx(1.0)
    assert s10["one_step_coefficient_c5"] == pytest.approx(
        appendix["one_step_coefficient_c5"]
    )
    assert centered["center_index_s"] == 11
    assert centered["ratio_to_reference"] != pytest.approx(1.0)
    assert minimum["one_step_coefficient_c5"] <= centered["one_step_coefficient_c5"]


def test_general_curve_is_labeled_as_proof_relaxation(size_frame):
    general = size_frame[size_frame["bound_variant"] == "general-proof-relaxation"]

    assert len(general) == 2
    assert general["specific_theorem_or_equation"].str.contains(
        "proof relaxation"
    ).all()
    assert general["additional_relaxations_json"].str.contains(
        "unspecified big-O constant"
    ).all()
    assert (general["ratio_to_reference"] > 1).all()


def test_target_error_sweep_has_fourth_order_segment_scaling(comparison_config):
    frame = generate_fourth_order_comparison(comparison_config, "target-error")
    selected = frame[
        (frame["decomposition_case"] == "two-term")
        & (frame["bound_variant"] == "appendix-m-expanded")
    ].sort_values("target_error", ascending=False)

    assert len(selected) == 2
    assert selected["one_step_coefficient_c5"].nunique() == 1
    assert selected["required_segment_count"].iloc[1] >= selected[
        "required_segment_count"
    ].iloc[0]
    assert (selected["accumulated_error_bound"] > 0).all()


def test_save_and_plot_outputs_remain_separate_from_resource_figures(
    size_frame,
    comparison_config,
):
    csv_path, metadata_path = save_fourth_order_comparison(
        size_frame,
        comparison_config,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = plot_fourth_order_comparison(
        csv_path,
        output_formats=("png",),
    )

    assert csv_path.exists()
    assert metadata["row_count"] == len(size_frame)
    assert "Appendix M" in metadata["ratio_denominator_policy"]
    assert len(outputs) == 6
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    assert all("_t_count." not in path.name and "_cnot_count." not in path.name for path in outputs)


def test_figures_use_required_log_axes_and_primary_curves(size_frame):
    coefficient = create_fourth_order_comparison_figure(
        size_frame,
        "one_step_coefficient",
        "two-term",
    )
    ratio = create_fourth_order_comparison_figure(
        size_frame,
        "ratio_to_reference",
        "multi-term",
    )

    assert coefficient.axes[0].get_yscale() == "log"
    assert ratio.axes[0].get_yscale() == "log"
    assert {line.get_label() for line in coefficient.axes[0].lines} == {
        "Childs general proof relaxation",
        "Childs Appendix M",
        "Schubert--Mendl centered",
        "Schubert--Mendl minimizing",
    }
    plt.close(coefficient)
    plt.close(ratio)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"system_qubit_values": (2,)}, "at least three"),
        ({"target_error_values": (0.0,)}, "positive"),
        ({"segment_count": 0}, "positive integer"),
        ({"norm_method": "bad"}, "norm_method"),
        ({"output_formats": ("eps",)}, "output_formats"),
    ],
)
def test_comparison_config_rejects_invalid_values(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        FourthOrderComparisonConfig(**changes)


def test_saved_csv_round_trip_preserves_required_columns(size_frame, comparison_config):
    csv_path, _ = save_fourth_order_comparison(size_frame, comparison_config)
    loaded = pd.read_csv(csv_path)

    assert tuple(loaded.columns) == FOURTH_ORDER_COMPARISON_COLUMNS

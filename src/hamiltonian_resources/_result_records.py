"""Shared serialization helpers for analytical resource result rows."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import EvaluationReport
from .planning import MPFPlan, QSVTPlan, TrotterPlan


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def git_metadata() -> tuple[str, bool | None]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def software_metadata() -> dict[str, Any]:
    git_commit, git_dirty = git_metadata()
    return {
        "package_version": package_version("hamiltonian-resources"),
        "python_version": platform.python_version(),
        "qiskit_version": package_version("qiskit"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def evaluation_report_metadata(report: EvaluationReport) -> dict[str, Any]:
    """Flatten one evaluation report while preserving all estimator diagnostics."""
    plan = report.plan
    result = dict(report.error_metadata)
    if isinstance(plan, TrotterPlan):
        result.update(
            segment_count=plan.repetitions,
            trotter_partition=plan.resolved_partition,
            trotter_group_count=len(plan.group_term_indices),
            lcu_normalization=1.0,
            amplitude_amplification="none",
            amplitude_amplification_rounds=0,
            good_subspace="system register",
            nominal_success_probability=1.0,
        )
    elif isinstance(plan, MPFPlan):
        error = plan.error_estimate
        diagnostics = getattr(error, "segment_diagnostics", None)
        structure = plan.lcu_structure
        per_segment = plan.logical_counts.as_dict()["per_segment"]
        physical_branch_count = (
            structure.physical_branch_count if structure is not None else plan.term_count
        )
        negative_coefficient_count = (
            structure.negative_coefficient_count
            if structure is not None
            else plan.term_count // 2
        )
        padding_branch_count = structure.padding_branch_count if structure is not None else 2
        active_branch_count = physical_branch_count + padding_branch_count
        branch_bits = max(1, int(np.ceil(np.log2(active_branch_count))))
        sign_branch_count = (
            structure.sign_branch_count
            if structure is not None
            else negative_coefficient_count + 1
        )
        if (
            diagnostics is None
            or diagnostics.local_commutator_error is None
            or diagnostics.total_branchwise_bch_remainder is None
        ):
            local_error_dominance = None
        elif diagnostics.local_commutator_error > diagnostics.total_branchwise_bch_remainder:
            local_error_dominance = "commutator"
        elif diagnostics.local_commutator_error < diagnostics.total_branchwise_bch_remainder:
            local_error_dominance = "bch"
        else:
            local_error_dominance = "tie"
        result.update(
            segment_count=plan.segments,
            mpf_term_count=plan.term_count,
            mpf_formal_order=plan.formal_order,
            mpf_branch_count_policy=plan.branch_count_selection.policy,
            mpf_branch_count_policy_extensiveness_g=(
                plan.branch_count_selection.extensiveness_g
            ),
            mpf_branch_count_policy_target_error=(
                plan.branch_count_selection.target_error
            ),
            mpf_r_error=(diagnostics.r_error if diagnostics is not None else None),
            mpf_r_time_1=(diagnostics.r_time_1 if diagnostics is not None else None),
            mpf_r_time_2=(diagnostics.r_time_2 if diagnostics is not None else None),
            mpf_active_constraints_json=json.dumps(
                diagnostics.active_constraints if diagnostics is not None else (),
                separators=(",", ":"),
            ),
            mpf_mu_upper=(diagnostics.mu_upper if diagnostics is not None else None),
            mpf_truncation_order_p0=(
                diagnostics.truncation_order_p0 if diagnostics is not None else None
            ),
            mpf_auxiliary_error=(
                diagnostics.auxiliary_error if diagnostics is not None else None
            ),
            mpf_auxiliary_allocation_fraction=(
                diagnostics.auxiliary_allocation_fraction
                if diagnostics is not None
                else None
            ),
            mpf_local_commutator_error=(
                diagnostics.local_commutator_error if diagnostics is not None else None
            ),
            mpf_local_truncated_bch_error=(
                diagnostics.local_truncated_bch_error if diagnostics is not None else None
            ),
            mpf_refined_lemma9_remainder=(
                diagnostics.refined_lemma9_remainder if diagnostics is not None else None
            ),
            mpf_refined_lemma10_remainder=(
                diagnostics.refined_lemma10_remainder if diagnostics is not None else None
            ),
            mpf_total_branchwise_bch_remainder=(
                diagnostics.total_branchwise_bch_remainder
                if diagnostics is not None
                else None
            ),
            mpf_local_step_error=(
                diagnostics.local_step_error if diagnostics is not None else None
            ),
            mpf_repeated_global_error=(
                diagnostics.repeated_global_error if diagnostics is not None else None
            ),
            mpf_legacy_first_time_limit=(
                diagnostics.legacy_first_time_limit if diagnostics is not None else None
            ),
            mpf_legacy_first_condition_passed=(
                diagnostics.legacy_first_condition_passed if diagnostics is not None else None
            ),
            mpf_second_time_limit=(
                diagnostics.second_time_limit if diagnostics is not None else None
            ),
            mpf_schedule_weights_json=json.dumps(
                diagnostics.schedule_weights if diagnostics is not None else (),
                separators=(",", ":"),
            ),
            mpf_schedule_weighted_extensiveness=(
                diagnostics.schedule_weighted_extensiveness
                if diagnostics is not None
                else None
            ),
            mpf_exact_commutator_cutoff=(
                diagnostics.max_exact_nested_commutator_order
                if diagnostics is not None
                else None
            ),
            mpf_locality_fallback=(
                diagnostics.used_locality_fallback if diagnostics is not None else None
            ),
            mpf_locality_fallback_reason=(
                diagnostics.locality_fallback_reason if diagnostics is not None else None
            ),
            mpf_refined_tail_fallback_status=(
                diagnostics.refined_tail_fallback_status if diagnostics is not None else None
            ),
            mpf_local_error_dominance=local_error_dominance,
            mpf_bound_policy=(
                getattr(error, "requested_method", None) or plan.method.error_method
            ),
            mpf_bound_candidates_json=json.dumps(
                [candidate.as_dict() for candidate in getattr(error, "bound_candidates", ())],
                sort_keys=True,
                separators=(",", ":"),
            ),
            query_count=plan.logical_counts.as_dict()["totals"]["controlled_s2"],
            bound_components_json=json.dumps(
                dict(result.get("bound_components", ())),
                sort_keys=True,
                separators=(",", ":"),
            ),
            bound_assumptions_json=json.dumps(
                result.get("bound_assumptions", ()), separators=(",", ":")
            ),
            commutator_cap_fallback=(getattr(error, "fallback_reason", None) is not None),
            commutator_bounds_json=json.dumps(
                dict(getattr(error, "commutator_bounds", ())),
                sort_keys=True,
                separators=(",", ":"),
            ),
            mpf_schedule=plan.method.schedule,
            mpf_exponents_json=(
                json.dumps(plan.exponents, separators=(",", ":"))
                if plan.exponents is not None
                else None
            ),
            mpf_coefficients_json=(
                json.dumps(plan.coefficients, separators=(",", ":"))
                if plan.coefficients is not None
                else None
            ),
            mpf_coefficient_l1_norm=plan.schedule_cost.coefficient_l1_norm,
            mpf_padding_weight=(structure.padding_weight if structure is not None else None),
            mpf_exponent_sum=plan.schedule_cost.exponent_sum,
            mpf_exponent_sum_source=plan.schedule_cost.source,
            mpf_explicit_schedule_available=(plan.schedule_cost.explicit_schedule_available),
            mpf_physical_branch_count=physical_branch_count,
            mpf_negative_coefficient_count=negative_coefficient_count,
            mpf_padding_branch_count=padding_branch_count,
            mpf_sign_branch_count=sign_branch_count,
            mpf_active_branch_count=active_branch_count,
            mpf_unused_branch_state_count=2**branch_bits - active_branch_count,
            mpf_prepare_calls_per_segment=per_segment["prepare"],
            mpf_select_calls_per_segment=per_segment["select"],
            mpf_good_reflections_per_segment=per_segment["good_reflection"],
            mpf_base_lcu_uses_per_segment=per_segment["select"],
            lcu_normalization=2.0,
            amplitude_amplification="one robust OAA round per segment",
            amplitude_amplification_rounds=plan.segments,
            good_subspace="branch register all-zero",
            nominal_success_probability=None,
        )
    elif isinstance(plan, QSVTPlan):
        result.update(
            query_count=plan.logical_counts.as_dict()["totals"]["block_encoding_query_slot"],
            qsvt_degree=plan.degree,
            lcu_normalization=2.0,
            amplitude_amplification="one robust OAA round",
            amplitude_amplification_rounds=plan.oaa_rounds,
            good_subspace="component, quadrature, and index registers all-zero",
            nominal_success_probability=None,
            bound_components_json=json.dumps(
                dict(result["bound_components"]),
                sort_keys=True,
                separators=(",", ":"),
            ),
            bound_assumptions_json=json.dumps(
                result["bound_assumptions"],
                separators=(",", ":"),
            ),
        )
    if "bound_components" in result and "bound_components_json" not in result:
        result["bound_components_json"] = json.dumps(
            dict(result["bound_components"]),
            sort_keys=True,
            separators=(",", ":"),
        )
    if "bound_assumptions" in result and "bound_assumptions_json" not in result:
        result["bound_assumptions_json"] = json.dumps(
            result["bound_assumptions"],
            separators=(",", ":"),
        )
    resource = report.resources
    result.update(
        total_qubits=resource.num_qubits,
        rotation_count=resource.rotation_count,
        toffoli_count=resource.toffoli_count,
        depth=resource.depth,
        t_count=resource.t_count,
        cnot_count=resource.cnot_count,
        counting_mode=resource.counting_mode,
        rotation_synthesis_error=resource.rotation_synthesis_error,
        status="ok",
    )
    return result

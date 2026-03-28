from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .disk import atomic_write_csv, atomic_write_json, atomic_write_npz
from .mode1 import (
    Mode1ModelLike,
    Mode1ObjectiveFn,
    Mode1ParamTree,
    SMPLMode1OptimizationResult,
    SMPLMode1StepDiagnostics,
    _coerce_params,
    default_mode1_objective,
    optimize_mode1,
)
from .mode3 import (
    SMPLMode3GroupSpec,
    SMPLMode3GroupSummary,
    SMPLMode3OptimizationResult,
    _binary_soft_weights,
    _coerce_group_specs,
    _group_activation_values,
    _group_summaries,
    _mode1_compatible_result,
    optimize_mode3,
)
from .mode4 import SMPLMode4OptimizationResult, optimize_mode4


class SMPLMode5PhaseSummary(NamedTuple):
    cycle_index: int
    surrogate_variant: str
    steps: int
    active_groups: tuple[str, ...]
    initial_objective: float
    final_objective: float
    initial_grad_norm: float
    final_grad_norm: float
    status: str


class SMPLMode5ControllerDecision(NamedTuple):
    cycle_index: int
    surrogate_variant: str
    target_active_groups: int
    selected_groups: tuple[str, ...]
    max_activation: float
    min_activation: float
    surrogate_initial_objective: float
    surrogate_final_objective: float
    refinement_initial_objective: float
    refinement_final_objective: float
    reason: str


class SMPLMode5TransferSummary(NamedTuple):
    cycle_index: int
    transferred_keys: tuple[str, ...]
    selected_groups: tuple[str, ...]
    runtime_policy: str
    logits_retained: bool


class SMPLMode5OptimizationResult(NamedTuple):
    implementation_status: str
    summary: str
    metadata: dict[str, Any]
    params: dict[str, jax.Array]
    output: Any
    faces: jax.Array
    parents: jax.Array
    logits: jax.Array
    weights: jax.Array
    objective_history: jax.Array
    grad_norm_history: jax.Array
    phase_summaries: tuple[SMPLMode5PhaseSummary, ...] = ()
    controller_history: tuple[SMPLMode5ControllerDecision, ...] = ()
    transfer_history: tuple[SMPLMode5TransferSummary, ...] = ()
    group_summaries: tuple[SMPLMode3GroupSummary, ...] = ()
    step_diagnostics: tuple[SMPLMode1StepDiagnostics, ...] = ()
    raw_params: dict[str, jax.Array] | None = None


def _freeze_to_selected_groups(
    stage_start: Mapping[str, jax.Array],
    selected_groups: tuple[SMPLMode3GroupSpec, ...],
):
    selected_keys = {key for spec in selected_groups for key in spec.keys}

    def _project(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
        updated = dict(params)
        for key, value in stage_start.items():
            if key not in selected_keys:
                updated[key] = value
        return updated

    return _project


def _surrogate_activation_scores(result: SMPLMode3OptimizationResult | SMPLMode4OptimizationResult) -> np.ndarray:
    if isinstance(result, SMPLMode4OptimizationResult):
        return np.asarray(result.soft_weights)[:, 0]
    return np.asarray(result.weights)[:, 0]


def _select_groups(
    group_specs: Sequence[SMPLMode3GroupSpec],
    activation_scores: np.ndarray,
    *,
    min_count: int,
    threshold: float,
) -> tuple[SMPLMode3GroupSpec, ...]:
    ordered = sorted(range(len(group_specs)), key=lambda idx: (-float(activation_scores[idx]), idx))
    selected = [idx for idx in ordered if float(activation_scores[idx]) >= threshold]
    if len(selected) < min_count:
        selected = ordered[:min_count]
    selected = sorted(set(selected))
    return tuple(group_specs[idx] for idx in selected)


def _phase_status(result: SMPLMode1OptimizationResult) -> str:
    return "converged" if float(result.grad_norm_history[-1]) <= 1.0e-6 else "improving"


def optimize_mode5(
    model: Mode1ModelLike,
    params: Mode1ParamTree,
    *,
    group_specs: Sequence[SMPLMode3GroupSpec | Mapping[str, Any]] | None = None,
    logits: jax.Array | None = None,
    objective_fn: Mode1ObjectiveFn | None = None,
    cycles: int = 3,
    surrogate_variant: str = "soft",
    surrogate_steps: int = 6,
    surrogate_step_size: float = 2.0e-2,
    refinement_steps: int = 8,
    refinement_step_size: float = 1.0e-2,
    logit_step_size: float | None = None,
    temperature: float = 0.35,
    activation_threshold: float = 0.55,
    min_active_groups: int = 1,
    activation_weight: float = 1.0e-3,
    entropy_weight: float = 1.0e-3,
    diagnostics_every: int = 10,
) -> SMPLMode5OptimizationResult:
    if cycles <= 0:
        raise ValueError("Mode 5 cycles must be positive")
    if surrogate_steps <= 0 or refinement_steps <= 0:
        raise ValueError("Mode 5 surrogate and refinement steps must be positive")
    if surrogate_variant not in {"soft", "straight-through"}:
        raise ValueError("surrogate_variant must be 'soft' or 'straight-through'")

    objective = default_mode1_objective if objective_fn is None else objective_fn
    current = _coerce_params(params)
    groups = _coerce_group_specs(group_specs, current)
    current_logits = jnp.zeros((len(groups),), dtype=jnp.float32) if logits is None else jnp.asarray(logits)
    if current_logits.shape != (len(groups),):
        raise ValueError(f"logits must have shape ({len(groups)},), got {current_logits.shape}")
    current_logits = current_logits.astype(jnp.asarray(next(iter(current.values()))).dtype)

    objective_parts: list[jax.Array] = []
    grad_parts: list[jax.Array] = []
    diagnostics: list[SMPLMode1StepDiagnostics] = []
    phase_summaries: list[SMPLMode5PhaseSummary] = []
    controller_history: list[SMPLMode5ControllerDecision] = []
    transfer_history: list[SMPLMode5TransferSummary] = []
    step_offset = 0
    final_result: SMPLMode1OptimizationResult | None = None

    for cycle in range(cycles):
        if surrogate_variant == "soft":
            surrogate = optimize_mode3(
                model,
                current,
                group_specs=groups,
                logits=current_logits,
                objective_fn=objective,
                steps=surrogate_steps,
                step_size=surrogate_step_size,
                logit_step_size=logit_step_size,
                temperature=temperature,
                activation_weight=activation_weight,
                entropy_weight=entropy_weight,
                diagnostics_every=max(diagnostics_every, surrogate_steps),
            )
        else:
            surrogate = optimize_mode4(
                model,
                current,
                group_specs=groups,
                logits=current_logits,
                objective_fn=objective,
                steps=surrogate_steps,
                step_size=surrogate_step_size,
                logit_step_size=logit_step_size,
                temperature=temperature,
                activation_weight=activation_weight,
                diagnostics_every=max(diagnostics_every, surrogate_steps),
            )

        activation_scores = _surrogate_activation_scores(surrogate)
        target_count = min(len(groups), max(min_active_groups, min_active_groups + cycle))
        active_groups = _select_groups(
            groups,
            activation_scores,
            min_count=target_count,
            threshold=float(activation_threshold),
        )
        stage_start = dict(surrogate.raw_params or current)
        refine = optimize_mode1(
            model,
            stage_start,
            objective_fn=objective,
            steps=refinement_steps,
            step_size=refinement_step_size,
            diagnostics_every=diagnostics_every,
            project_fn=_freeze_to_selected_groups(stage_start, active_groups),
        )
        current = refine.params
        current_logits = surrogate.logits
        objective_parts.append(refine.objective_history)
        grad_parts.append(refine.grad_norm_history)
        diagnostics.extend(
            SMPLMode1StepDiagnostics(
                step=diag.step + step_offset,
                objective=diag.objective,
                grad_norm=diag.grad_norm,
                vertex_rms=diag.vertex_rms,
                joint_rms=diag.joint_rms,
                transl_norm=diag.transl_norm,
            )
            for diag in refine.step_diagnostics
        )
        status = _phase_status(refine)
        selected_names = tuple(spec.name for spec in active_groups)
        phase_summaries.append(
            SMPLMode5PhaseSummary(
                cycle_index=cycle,
                surrogate_variant=surrogate_variant,
                steps=refinement_steps,
                active_groups=selected_names,
                initial_objective=float(refine.objective_history[0]),
                final_objective=float(refine.objective_history[-1]),
                initial_grad_norm=float(refine.grad_norm_history[0]),
                final_grad_norm=float(refine.grad_norm_history[-1]),
                status=status,
            )
        )
        reason = "initial-controller-selection" if cycle == 0 else ("expanded-active-set" if target_count > min_active_groups else "refined-active-set")
        controller_history.append(
            SMPLMode5ControllerDecision(
                cycle_index=cycle,
                surrogate_variant=surrogate_variant,
                target_active_groups=target_count,
                selected_groups=selected_names,
                max_activation=float(np.max(activation_scores)) if activation_scores.size else 0.0,
                min_activation=float(np.min(activation_scores)) if activation_scores.size else 0.0,
                surrogate_initial_objective=float(surrogate.objective_history[0]),
                surrogate_final_objective=float(surrogate.objective_history[-1]),
                refinement_initial_objective=float(refine.objective_history[0]),
                refinement_final_objective=float(refine.objective_history[-1]),
                reason=reason,
            )
        )
        transfer_history.append(
            SMPLMode5TransferSummary(
                cycle_index=cycle,
                transferred_keys=tuple(sorted(current)),
                selected_groups=selected_names,
                runtime_policy="carry-forward-params-and-logits",
                logits_retained=True,
            )
        )
        step_offset += refinement_steps
        final_result = refine

    assert final_result is not None
    objective_history = jnp.concatenate(objective_parts) if objective_parts else jnp.zeros((0,), dtype=jnp.float32)
    grad_norm_history = jnp.concatenate(grad_parts) if grad_parts else jnp.zeros((0,), dtype=jnp.float32)
    final_weights = _binary_soft_weights(current_logits, temperature=temperature)
    status = "converged" if float(grad_norm_history[-1]) <= 1.0e-6 else "improving"
    metadata = {
        "mode": 5,
        "cycles": cycles,
        "n_groups": len(groups),
        "group_names": [spec.name for spec in groups],
        "surrogate_variant": surrogate_variant,
        "temperature": float(temperature),
        "status": status,
    }
    summary = (
        f"Completed {cycles} dynamic controller cycles with {surrogate_variant} surrogate routing; "
        f"final objective={float(objective_history[-1]):.6g}, final grad_norm={float(grad_norm_history[-1]):.6g}"
    )
    return SMPLMode5OptimizationResult(
        implementation_status="implemented-dynamic-controller",
        summary=summary,
        metadata=metadata,
        params=final_result.params,
        output=final_result.output,
        faces=final_result.faces,
        parents=final_result.parents,
        logits=current_logits,
        weights=final_weights,
        objective_history=objective_history,
        grad_norm_history=grad_norm_history,
        phase_summaries=tuple(phase_summaries),
        controller_history=tuple(controller_history),
        transfer_history=tuple(transfer_history),
        group_summaries=_group_summaries(groups, _group_activation_values(final_weights)),
        step_diagnostics=tuple(diagnostics),
        raw_params=current,
    )


def mode5_history_payload(result: SMPLMode5OptimizationResult) -> dict[str, Any]:
    steps = np.arange(1, int(result.objective_history.shape[0]) + 1, dtype=np.int32)
    return {
        "schema_name": "smpljax.mode5.history",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "step": steps,
        "objective_history": np.asarray(result.objective_history),
        "grad_norm_history": np.asarray(result.grad_norm_history),
    }


def mode5_metrics_payload(result: SMPLMode5OptimizationResult) -> dict[str, Any]:
    vertices = np.asarray(result.output.vertices)
    joints = np.asarray(result.output.joints)
    activations = np.asarray(result.weights)[:, 0]
    return {
        "schema_name": "smpljax.mode5.metrics",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "summary": result.summary,
        "metadata": result.metadata,
        "cycles": len(result.phase_summaries),
        "n_groups": len(result.group_summaries),
        "n_steps": int(result.objective_history.shape[0]),
        "initial_objective": float(result.objective_history[0]),
        "final_objective": float(result.objective_history[-1]),
        "objective_drop": float(result.objective_history[0] - result.objective_history[-1]),
        "initial_grad_norm": float(result.grad_norm_history[0]),
        "final_grad_norm": float(result.grad_norm_history[-1]),
        "n_vertices": int(vertices.shape[-2]),
        "n_joints": int(joints.shape[-2]),
        "batch_size": int(vertices.shape[0]),
        "vertex_rms": float(np.sqrt(np.mean(np.square(vertices)))),
        "joint_rms": float(np.sqrt(np.mean(np.square(joints)))),
        "mean_activation": float(np.mean(activations)) if activations.size else 0.0,
        "status": str(result.metadata.get("status", "improving")),
    }


def build_mode5_visualization_payload(
    result: SMPLMode5OptimizationResult,
    *,
    title: str = "SMPL Mode 5 Result",
) -> dict[str, Any]:
    return {
        "schema_name": "smpljax.mode5.visualization",
        "schema_version": "1.0",
        "title": title,
        "implementation_status": result.implementation_status,
        "summary": result.summary,
        "metadata": result.metadata,
        "vertices": np.asarray(result.output.vertices[0]).tolist(),
        "joints": np.asarray(result.output.joints[0]).tolist(),
        "params": {key: np.asarray(value).tolist() for key, value in result.params.items()},
        "groups": [
            {
                "group_index": group.group_index,
                "group_name": group.group_name,
                "keys": list(group.keys),
                "activation": group.activation,
            }
            for group in result.group_summaries
        ],
        "phases": [
            {
                "cycle_index": phase.cycle_index,
                "surrogate_variant": phase.surrogate_variant,
                "steps": phase.steps,
                "active_groups": list(phase.active_groups),
                "initial_objective": phase.initial_objective,
                "final_objective": phase.final_objective,
                "initial_grad_norm": phase.initial_grad_norm,
                "final_grad_norm": phase.final_grad_norm,
                "status": phase.status,
            }
            for phase in result.phase_summaries
        ],
    }


def export_mode5_artifacts(
    output_dir: str | Path,
    result: SMPLMode5OptimizationResult,
    *,
    prefix: str = "smpl_mode5",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    history_npz_path = out_dir / f"{prefix}_history.npz"
    history_json_path = out_dir / f"{prefix}_history.json"
    history_csv_path = out_dir / f"{prefix}_history.csv"
    phases_path = out_dir / f"{prefix}_phases.json"
    controller_path = out_dir / f"{prefix}_controller.json"
    transfer_path = out_dir / f"{prefix}_transfer.json"

    metrics = mode5_metrics_payload(result)
    history = mode5_history_payload(result)
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(viewer_path, build_mode5_visualization_payload(result))
    atomic_write_npz(
        history_npz_path,
        step=np.asarray(history["step"]),
        objective_history=np.asarray(history["objective_history"]),
        grad_norm_history=np.asarray(history["grad_norm_history"]),
    )
    rows = [
        {"step": int(step), "objective": float(obj), "grad_norm": float(grad)}
        for step, obj, grad in zip(history["step"], history["objective_history"], history["grad_norm_history"])
    ]
    atomic_write_json(history_json_path, rows)
    atomic_write_csv(history_csv_path, rows)
    atomic_write_json(phases_path, [phase._asdict() for phase in result.phase_summaries])
    atomic_write_json(controller_path, [entry._asdict() for entry in result.controller_history])
    atomic_write_json(transfer_path, [entry._asdict() for entry in result.transfer_history])
    return {
        "metrics": metrics_path,
        "viewer_payload": viewer_path,
        "history": history_npz_path,
        "history_json": history_json_path,
        "history_csv": history_csv_path,
        "phases": phases_path,
        "controller": controller_path,
        "transfer": transfer_path,
    }


def visualize_mode5_result(result: SMPLMode5OptimizationResult, *, backend: str = "matplotlib", **kwargs):
    from .visualization.mode1 import visualize_mode1_result as _visualize_mode1_result

    return _visualize_mode1_result(_mode1_compatible_result(result), backend=backend, **kwargs)

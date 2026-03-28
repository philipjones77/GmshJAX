from __future__ import annotations

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
    _forward,
    _tree_l2_norm,
    default_mode1_objective,
)
from .mode3 import (
    SMPLMode3GroupSpec,
    SMPLMode3GroupSummary,
    _apply_group_gates,
    _binary_soft_weights,
    _coerce_group_specs,
    _group_activation_values,
    _group_summaries,
    _mode1_compatible_result,
    _result_status,
)


class SMPLMode4OptimizationResult(NamedTuple):
    implementation_status: str
    summary: str
    metadata: dict[str, Any]
    params: dict[str, jax.Array]
    output: Any
    faces: jax.Array
    parents: jax.Array
    logits: jax.Array
    hard_weights: jax.Array
    soft_weights: jax.Array
    objective_history: jax.Array
    grad_norm_history: jax.Array
    group_summaries: tuple[SMPLMode3GroupSummary, ...] = ()
    step_diagnostics: tuple[SMPLMode1StepDiagnostics, ...] = ()
    raw_params: dict[str, jax.Array] | None = None


def _binary_straight_through_weights(logits: jax.Array, temperature: float = 0.25) -> jax.Array:
    soft = _binary_soft_weights(logits, temperature=temperature)
    hard_index = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(hard_index, 2, dtype=soft.dtype)
    return hard + soft - jax.lax.stop_gradient(soft)


def optimize_mode4(
    model: Mode1ModelLike,
    params: Mode1ParamTree,
    *,
    group_specs: tuple[SMPLMode3GroupSpec, ...] | None = None,
    logits: jax.Array | None = None,
    objective_fn: Mode1ObjectiveFn | None = None,
    steps: int = 24,
    step_size: float = 1.0e-2,
    logit_step_size: float | None = None,
    temperature: float = 0.25,
    activation_weight: float = 1.0e-3,
    diagnostics_every: int = 10,
) -> SMPLMode4OptimizationResult:
    if steps <= 0:
        raise ValueError("Mode 4 steps must be positive")

    objective = default_mode1_objective if objective_fn is None else objective_fn
    current = _coerce_params(params)
    groups = _coerce_group_specs(group_specs, current)
    theta = jnp.zeros((len(groups),), dtype=jnp.float32) if logits is None else jnp.asarray(logits)
    if theta.shape != (len(groups),):
        raise ValueError(f"logits must have shape ({len(groups)},), got {theta.shape}")
    theta = theta.astype(jnp.asarray(next(iter(current.values()))).dtype)
    logit_lr = float(step_size if logit_step_size is None else logit_step_size)

    objective_history: list[jax.Array] = []
    grad_norm_history: list[jax.Array] = []
    diagnostics: list[SMPLMode1StepDiagnostics] = []

    def _loss_fn(local_params, local_logits):
        st_weights = _binary_straight_through_weights(local_logits, temperature=temperature)
        gated_params = _apply_group_gates(local_params, groups, _group_activation_values(st_weights))
        output = _forward(model, gated_params)
        return objective(model, gated_params, output) + jnp.asarray(activation_weight, dtype=st_weights.dtype) * jnp.mean(st_weights[:, 0])

    value_and_grad = jax.value_and_grad(_loss_fn, argnums=(0, 1))

    for step in range(1, steps + 1):
        loss, (param_grads, logit_grads) = value_and_grad(current, theta)
        grad_norm = jnp.sqrt(jnp.square(_tree_l2_norm(param_grads)) + jnp.sum(jnp.square(logit_grads)))
        objective_history.append(loss)
        grad_norm_history.append(grad_norm)
        current = jax.tree_util.tree_map(
            lambda p, g: p - jnp.asarray(step_size, dtype=p.dtype) * g,
            current,
            param_grads,
        )
        theta = theta - jnp.asarray(logit_lr, dtype=theta.dtype) * logit_grads
        if diagnostics_every > 0 and (step == 1 or step == steps or step % diagnostics_every == 0):
            st_weights = _binary_straight_through_weights(theta, temperature=temperature)
            gated = _apply_group_gates(current, groups, _group_activation_values(st_weights))
            out = _forward(model, gated)
            transl = gated.get("transl", jnp.zeros((1, 3), dtype=out.vertices.dtype))
            diagnostics.append(
                SMPLMode1StepDiagnostics(
                    step=step,
                    objective=float(loss),
                    grad_norm=float(grad_norm),
                    vertex_rms=float(jnp.sqrt(jnp.mean(jnp.square(out.vertices)))),
                    joint_rms=float(jnp.sqrt(jnp.mean(jnp.square(out.joints)))),
                    transl_norm=float(jnp.sqrt(jnp.mean(jnp.square(transl)))),
                )
            )

    final_hard_weights = _binary_straight_through_weights(theta, temperature=temperature)
    final_soft_weights = _binary_soft_weights(theta, temperature=temperature)
    final_params = _apply_group_gates(current, groups, _group_activation_values(final_hard_weights))
    final_output = _forward(model, final_params)
    objective_hist = jnp.asarray(objective_history)
    grad_hist = jnp.asarray(grad_norm_history)
    status = _result_status(grad_hist)
    metadata = {
        "mode": 4,
        "n_groups": len(groups),
        "group_names": [spec.name for spec in groups],
        "temperature": float(temperature),
        "candidate_kind": "smpl-parameter-group-straight-through",
        "status": status,
    }
    summary = (
        f"Completed straight-through parameter-group routing over {len(groups)} groups; "
        f"final objective={float(objective_hist[-1]):.6g}, final grad_norm={float(grad_hist[-1]):.6g}"
    )
    return SMPLMode4OptimizationResult(
        implementation_status="implemented-straight-through-routing",
        summary=summary,
        metadata=metadata,
        params=final_params,
        output=final_output,
        faces=jnp.asarray(model.data.faces_tensor, dtype=jnp.int32),
        parents=jnp.asarray(model.data.parents, dtype=jnp.int32),
        logits=theta,
        hard_weights=final_hard_weights,
        soft_weights=final_soft_weights,
        objective_history=objective_hist,
        grad_norm_history=grad_hist,
        group_summaries=_group_summaries(groups, _group_activation_values(final_soft_weights)),
        step_diagnostics=tuple(diagnostics),
        raw_params=current,
    )


def mode4_history_payload(result: SMPLMode4OptimizationResult) -> dict[str, Any]:
    steps = np.arange(1, int(result.objective_history.shape[0]) + 1, dtype=np.int32)
    return {
        "schema_name": "smpljax.mode4.history",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "step": steps,
        "objective_history": np.asarray(result.objective_history),
        "grad_norm_history": np.asarray(result.grad_norm_history),
    }


def mode4_metrics_payload(result: SMPLMode4OptimizationResult) -> dict[str, Any]:
    vertices = np.asarray(result.output.vertices)
    joints = np.asarray(result.output.joints)
    hard_active = np.asarray(result.hard_weights)[:, 0]
    return {
        "schema_name": "smpljax.mode4.metrics",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "summary": result.summary,
        "metadata": result.metadata,
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
        "n_active_groups": int(np.sum(hard_active >= 0.5)),
        "status": str(result.metadata.get("status", "improving")),
    }


def build_mode4_visualization_payload(
    result: SMPLMode4OptimizationResult,
    *,
    title: str = "SMPL Mode 4 Result",
) -> dict[str, Any]:
    return {
        "schema_name": "smpljax.mode4.visualization",
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
        "hard_weights": np.asarray(result.hard_weights).tolist(),
        "soft_weights": np.asarray(result.soft_weights).tolist(),
    }


def export_mode4_artifacts(
    output_dir: str | Path,
    result: SMPLMode4OptimizationResult,
    *,
    prefix: str = "smpl_mode4",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    history_npz_path = out_dir / f"{prefix}_history.npz"
    history_json_path = out_dir / f"{prefix}_history.json"
    history_csv_path = out_dir / f"{prefix}_history.csv"
    groups_path = out_dir / f"{prefix}_groups.json"

    metrics = mode4_metrics_payload(result)
    history = mode4_history_payload(result)
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(viewer_path, build_mode4_visualization_payload(result))
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
    atomic_write_json(
        groups_path,
        [
            {
                "group_index": group.group_index,
                "group_name": group.group_name,
                "keys": list(group.keys),
                "activation": group.activation,
            }
            for group in result.group_summaries
        ],
    )
    return {
        "metrics": metrics_path,
        "viewer_payload": viewer_path,
        "history": history_npz_path,
        "history_json": history_json_path,
        "history_csv": history_csv_path,
        "groups": groups_path,
    }


def visualize_mode4_result(result: SMPLMode4OptimizationResult, *, backend: str = "matplotlib", **kwargs):
    from .visualization.mode1 import visualize_mode1_result as _visualize_mode1_result

    return _visualize_mode1_result(_mode1_compatible_result(result), backend=backend, **kwargs)

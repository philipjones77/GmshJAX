from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .body_models import ModelOutput
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


class SMPLMode3GroupSpec(NamedTuple):
    name: str
    keys: tuple[str, ...]


class SMPLMode3GroupSummary(NamedTuple):
    group_index: int
    group_name: str
    keys: tuple[str, ...]
    activation: float


class SMPLMode3OptimizationResult(NamedTuple):
    implementation_status: str
    summary: str
    metadata: dict[str, Any]
    params: dict[str, jax.Array]
    output: ModelOutput
    faces: jax.Array
    parents: jax.Array
    logits: jax.Array
    weights: jax.Array
    objective_history: jax.Array
    grad_norm_history: jax.Array
    group_summaries: tuple[SMPLMode3GroupSummary, ...] = ()
    step_diagnostics: tuple[SMPLMode1StepDiagnostics, ...] = ()
    raw_params: dict[str, jax.Array] | None = None


def _default_group_specs(params: Mapping[str, jax.Array]) -> tuple[SMPLMode3GroupSpec, ...]:
    specs = (
        SMPLMode3GroupSpec("translation", ("transl",)),
        SMPLMode3GroupSpec("global_orient", ("global_orient",)),
        SMPLMode3GroupSpec("body_pose", ("body_pose",)),
        SMPLMode3GroupSpec("betas", ("betas",)),
        SMPLMode3GroupSpec("expression", ("expression",)),
        SMPLMode3GroupSpec("face_pose", ("jaw_pose", "leye_pose", "reye_pose")),
        SMPLMode3GroupSpec("left_hand_pose", ("left_hand_pose",)),
        SMPLMode3GroupSpec("right_hand_pose", ("right_hand_pose",)),
    )
    available = set(params)
    return tuple(
        SMPLMode3GroupSpec(spec.name, tuple(key for key in spec.keys if key in available))
        for spec in specs
        if any(key in available for key in spec.keys)
    )


def _coerce_group_specs(
    group_specs: Sequence[SMPLMode3GroupSpec | Mapping[str, Any]] | None,
    params: Mapping[str, jax.Array],
) -> tuple[SMPLMode3GroupSpec, ...]:
    if group_specs is None:
        specs = _default_group_specs(params)
    else:
        specs = tuple(
            spec
            if isinstance(spec, SMPLMode3GroupSpec)
            else SMPLMode3GroupSpec(
                name=str(spec["name"]),
                keys=tuple(str(key) for key in spec["keys"]),
            )
            for spec in group_specs
        )
        specs = tuple(
            SMPLMode3GroupSpec(spec.name, tuple(key for key in spec.keys if key in params))
            for spec in specs
            if any(key in params for key in spec.keys)
        )
    if not specs:
        raise ValueError("Mode 3 requires at least one parameter group")
    return specs


def _binary_soft_weights(logits: jax.Array, temperature: float = 0.5) -> jax.Array:
    pair_logits = jnp.stack([jnp.asarray(logits), jnp.zeros_like(logits)], axis=-1)
    return jax.nn.softmax(pair_logits / jnp.maximum(jnp.asarray(temperature, dtype=pair_logits.dtype), 1.0e-6), axis=-1)


def _group_activation_values(weights: jax.Array) -> jax.Array:
    return jnp.asarray(weights)[:, 0]


def _apply_group_gates(
    params: Mapping[str, jax.Array],
    group_specs: Sequence[SMPLMode3GroupSpec],
    gate_values: jax.Array,
) -> dict[str, jax.Array]:
    gated = {key: jnp.asarray(value) for key, value in params.items()}
    for index, spec in enumerate(group_specs):
        gate = jnp.asarray(gate_values[index])
        for key in spec.keys:
            gated[key] = jnp.asarray(params[key]) * gate
    return gated


def _group_summaries(group_specs: Sequence[SMPLMode3GroupSpec], gate_values: jax.Array) -> tuple[SMPLMode3GroupSummary, ...]:
    return tuple(
        SMPLMode3GroupSummary(
            group_index=index,
            group_name=spec.name,
            keys=spec.keys,
            activation=float(gate_values[index]),
        )
        for index, spec in enumerate(group_specs)
    )


def _result_status(grad_norm_history: jax.Array) -> str:
    if int(grad_norm_history.shape[0]) == 0:
        return "unknown"
    return "converged" if float(grad_norm_history[-1]) <= 1.0e-6 else "improving"


def _mode1_compatible_result(result: SMPLMode3OptimizationResult) -> SMPLMode1OptimizationResult:
    return SMPLMode1OptimizationResult(
        params=result.params,
        output=result.output,
        faces=result.faces,
        parents=result.parents,
        objective_history=result.objective_history,
        grad_norm_history=result.grad_norm_history,
        step_diagnostics=result.step_diagnostics,
    )


def optimize_mode3(
    model: Mode1ModelLike,
    params: Mode1ParamTree,
    *,
    group_specs: Sequence[SMPLMode3GroupSpec | Mapping[str, Any]] | None = None,
    logits: jax.Array | None = None,
    objective_fn: Mode1ObjectiveFn | None = None,
    steps: int = 24,
    step_size: float = 1.0e-2,
    logit_step_size: float | None = None,
    temperature: float = 0.5,
    activation_weight: float = 1.0e-3,
    entropy_weight: float = 1.0e-3,
    diagnostics_every: int = 10,
) -> SMPLMode3OptimizationResult:
    if steps <= 0:
        raise ValueError("Mode 3 steps must be positive")

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

    def _loss_fn(local_params: Mapping[str, jax.Array], local_logits: jax.Array) -> jax.Array:
        weights = _binary_soft_weights(local_logits, temperature=temperature)
        gated_params = _apply_group_gates(local_params, groups, _group_activation_values(weights))
        output = _forward(model, gated_params)
        entropy = -jnp.sum(weights * jnp.log(jnp.maximum(weights, 1.0e-12)), axis=-1)
        return (
            objective(model, gated_params, output)
            + jnp.asarray(activation_weight, dtype=weights.dtype) * jnp.mean(weights[:, 0])
            + jnp.asarray(entropy_weight, dtype=weights.dtype) * jnp.mean(entropy)
        )

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
            weights = _binary_soft_weights(theta, temperature=temperature)
            gated = _apply_group_gates(current, groups, _group_activation_values(weights))
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

    final_weights = _binary_soft_weights(theta, temperature=temperature)
    final_params = _apply_group_gates(current, groups, _group_activation_values(final_weights))
    final_output = _forward(model, final_params)
    objective_hist = jnp.asarray(objective_history)
    grad_hist = jnp.asarray(grad_norm_history)
    status = _result_status(grad_hist)
    metadata = {
        "mode": 3,
        "n_groups": len(groups),
        "group_names": [spec.name for spec in groups],
        "temperature": float(temperature),
        "candidate_kind": "smpl-parameter-group-soft-routing",
        "status": status,
    }
    summary = (
        f"Completed soft parameter-group routing over {len(groups)} groups; "
        f"final objective={float(objective_hist[-1]):.6g}, final grad_norm={float(grad_hist[-1]):.6g}"
    )
    return SMPLMode3OptimizationResult(
        implementation_status="implemented-soft-routing",
        summary=summary,
        metadata=metadata,
        params=final_params,
        output=final_output,
        faces=jnp.asarray(model.data.faces_tensor, dtype=jnp.int32),
        parents=jnp.asarray(model.data.parents, dtype=jnp.int32),
        logits=theta,
        weights=final_weights,
        objective_history=objective_hist,
        grad_norm_history=grad_hist,
        group_summaries=_group_summaries(groups, _group_activation_values(final_weights)),
        step_diagnostics=tuple(diagnostics),
        raw_params=current,
    )


def mode3_history_payload(result: SMPLMode3OptimizationResult) -> dict[str, Any]:
    steps = np.arange(1, int(result.objective_history.shape[0]) + 1, dtype=np.int32)
    return {
        "schema_name": "smpljax.mode3.history",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "step": steps,
        "objective_history": np.asarray(result.objective_history),
        "grad_norm_history": np.asarray(result.grad_norm_history),
    }


def mode3_metrics_payload(result: SMPLMode3OptimizationResult) -> dict[str, Any]:
    vertices = np.asarray(result.output.vertices)
    joints = np.asarray(result.output.joints)
    activations = np.asarray(result.weights)[:, 0]
    return {
        "schema_name": "smpljax.mode3.metrics",
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
        "mean_activation": float(np.mean(activations)) if activations.size else 0.0,
        "max_activation": float(np.max(activations)) if activations.size else 0.0,
        "status": str(result.metadata.get("status", "improving")),
    }


def build_mode3_visualization_payload(
    result: SMPLMode3OptimizationResult,
    *,
    title: str = "SMPL Mode 3 Result",
) -> dict[str, Any]:
    return {
        "schema_name": "smpljax.mode3.visualization",
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
        "weights": np.asarray(result.weights).tolist(),
    }


def export_mode3_artifacts(
    output_dir: str | Path,
    result: SMPLMode3OptimizationResult,
    *,
    prefix: str = "smpl_mode3",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    history_npz_path = out_dir / f"{prefix}_history.npz"
    history_json_path = out_dir / f"{prefix}_history.json"
    history_csv_path = out_dir / f"{prefix}_history.csv"
    groups_path = out_dir / f"{prefix}_groups.json"

    metrics = mode3_metrics_payload(result)
    history = mode3_history_payload(result)
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(viewer_path, build_mode3_visualization_payload(result))
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


def visualize_mode3_result(result: SMPLMode3OptimizationResult, *, backend: str = "matplotlib", **kwargs):
    from .visualization.mode1 import visualize_mode1_result as _visualize_mode1_result

    return _visualize_mode1_result(_mode1_compatible_result(result), backend=backend, **kwargs)

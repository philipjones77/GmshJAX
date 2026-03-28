from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, NamedTuple

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
    optimize_mode1,
)


class SMPLMode2StageConfig(NamedTuple):
    name: str
    steps: int = 20
    step_size: float = 1.0e-2
    trainable_keys: tuple[str, ...] | None = None
    objective_fn: Mode1ObjectiveFn | None = None
    project_fn: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None = None


class SMPLMode2PhaseSummary(NamedTuple):
    stage_index: int
    stage_name: str
    steps: int
    trainable_keys: tuple[str, ...]
    initial_objective: float
    final_objective: float
    initial_grad_norm: float
    final_grad_norm: float
    status: str


class SMPLMode2OptimizationResult(NamedTuple):
    implementation_status: str
    summary: str
    metadata: dict[str, Any]
    params: dict[str, jax.Array] | None = None
    output: ModelOutput | None = None
    faces: jax.Array | None = None
    parents: jax.Array | None = None
    objective_history: jax.Array | None = None
    grad_norm_history: jax.Array | None = None
    phase_summaries: tuple[SMPLMode2PhaseSummary, ...] = ()
    step_diagnostics: tuple[SMPLMode1StepDiagnostics, ...] = ()


def _default_stage_configs() -> tuple[SMPLMode2StageConfig, ...]:
    return (
        SMPLMode2StageConfig(name="translation_warmup", steps=4, step_size=0.05, trainable_keys=("transl",)),
        SMPLMode2StageConfig(name="full_refine", steps=8, step_size=0.01, trainable_keys=None),
    )


def _coerce_stage_config(stage: SMPLMode2StageConfig | Mapping[str, Any]) -> SMPLMode2StageConfig:
    if isinstance(stage, SMPLMode2StageConfig):
        return stage
    return SMPLMode2StageConfig(
        name=str(stage["name"]),
        steps=int(stage.get("steps", 20)),
        step_size=float(stage.get("step_size", 1.0e-2)),
        trainable_keys=None if stage.get("trainable_keys") is None else tuple(str(key) for key in stage["trainable_keys"]),
        objective_fn=stage.get("objective_fn"),
        project_fn=stage.get("project_fn"),
    )


def _compose_project_fn(
    stage_start: Mapping[str, jax.Array],
    *,
    trainable_keys: tuple[str, ...] | None,
    project_fn: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None,
) -> Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None:
    if trainable_keys is None and project_fn is None:
        return project_fn

    trainable = None if trainable_keys is None else set(trainable_keys)

    def _project(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
        updated = dict(params)
        if trainable is not None:
            for key, value in stage_start.items():
                if key not in trainable:
                    updated[key] = value
        if project_fn is not None:
            updated = project_fn(updated)
        return updated

    return _project


def _result_status(result: SMPLMode1OptimizationResult) -> str:
    final_grad = float(result.grad_norm_history[-1]) if result.grad_norm_history is not None and result.grad_norm_history.size else 0.0
    return "converged" if final_grad <= 1.0e-6 else "improving"


def optimize_mode2(
    model: Mode1ModelLike,
    params: Mode1ParamTree,
    *,
    stages: Sequence[SMPLMode2StageConfig | Mapping[str, Any]] | None = None,
    diagnostics_every: int = 10,
) -> SMPLMode2OptimizationResult:
    stage_configs = tuple(_coerce_stage_config(stage) for stage in (stages or _default_stage_configs()))
    if not stage_configs:
        raise ValueError("Mode 2 requires at least one stage")

    current = _coerce_params(params)
    objective_parts: list[jax.Array] = []
    grad_parts: list[jax.Array] = []
    diagnostics: list[SMPLMode1StepDiagnostics] = []
    phase_summaries: list[SMPLMode2PhaseSummary] = []
    step_offset = 0
    final_result: SMPLMode1OptimizationResult | None = None

    for stage_index, stage in enumerate(stage_configs):
        if stage.steps <= 0:
            raise ValueError("Mode 2 stage steps must be positive")
        stage_start = dict(current)
        stage_result = optimize_mode1(
            model,
            current,
            objective_fn=stage.objective_fn,
            steps=stage.steps,
            step_size=stage.step_size,
            diagnostics_every=diagnostics_every,
            project_fn=_compose_project_fn(stage_start, trainable_keys=stage.trainable_keys, project_fn=stage.project_fn),
        )
        current = stage_result.params
        objective_parts.append(stage_result.objective_history)
        grad_parts.append(stage_result.grad_norm_history)
        diagnostics.extend(
            SMPLMode1StepDiagnostics(
                step=diag.step + step_offset,
                objective=diag.objective,
                grad_norm=diag.grad_norm,
                vertex_rms=diag.vertex_rms,
                joint_rms=diag.joint_rms,
                transl_norm=diag.transl_norm,
            )
            for diag in stage_result.step_diagnostics
        )
        trainable_keys = tuple(sorted(stage_result.params)) if stage.trainable_keys is None else tuple(stage.trainable_keys)
        phase_summaries.append(
            SMPLMode2PhaseSummary(
                stage_index=stage_index,
                stage_name=stage.name,
                steps=stage.steps,
                trainable_keys=trainable_keys,
                initial_objective=float(stage_result.objective_history[0]),
                final_objective=float(stage_result.objective_history[-1]),
                initial_grad_norm=float(stage_result.grad_norm_history[0]),
                final_grad_norm=float(stage_result.grad_norm_history[-1]),
                status=_result_status(stage_result),
            )
        )
        step_offset += stage.steps
        final_result = stage_result

    assert final_result is not None
    objective_history = jnp.concatenate(objective_parts) if objective_parts else jnp.zeros((0,), dtype=jnp.float32)
    grad_norm_history = jnp.concatenate(grad_parts) if grad_parts else jnp.zeros((0,), dtype=jnp.float32)
    summary = (
        f"Completed {len(stage_configs)} staged phases; "
        f"final objective={float(objective_history[-1]):.6g}, final grad_norm={float(grad_norm_history[-1]):.6g}"
    )
    metadata = {
        "mode": 2,
        "n_stages": len(stage_configs),
        "stage_names": [stage.name for stage in stage_configs],
        "total_steps": int(objective_history.shape[0]),
        "status": _result_status(final_result),
    }
    return SMPLMode2OptimizationResult(
        implementation_status="implemented-staged-workflow",
        summary=summary,
        metadata=metadata,
        params=final_result.params,
        output=final_result.output,
        faces=final_result.faces,
        parents=final_result.parents,
        objective_history=objective_history,
        grad_norm_history=grad_norm_history,
        phase_summaries=tuple(phase_summaries),
        step_diagnostics=tuple(diagnostics),
    )


def mode2_history_payload(result: SMPLMode2OptimizationResult) -> dict[str, Any]:
    steps = (
        np.arange(1, int(result.objective_history.shape[0]) + 1, dtype=np.int32)
        if result.objective_history is not None
        else np.zeros((0,), dtype=np.int32)
    )
    return {
        "schema_name": "smpljax.mode2.history",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "step": steps,
        "objective_history": np.asarray(result.objective_history) if result.objective_history is not None else np.zeros((0,), dtype=np.float32),
        "grad_norm_history": np.asarray(result.grad_norm_history) if result.grad_norm_history is not None else np.zeros((0,), dtype=np.float32),
    }


def mode2_metrics_payload(result: SMPLMode2OptimizationResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_name": "smpljax.mode2.metrics",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "summary": result.summary,
        "metadata": result.metadata,
        "n_stages": len(result.phase_summaries),
    }
    if result.output is not None and result.objective_history is not None and result.grad_norm_history is not None:
        vertices = np.asarray(result.output.vertices)
        joints = np.asarray(result.output.joints)
        payload.update(
            {
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
                "status": str(result.metadata.get("status", "improving")),
            }
        )
    return payload


def build_mode2_visualization_payload(result: SMPLMode2OptimizationResult, *, title: str = "SMPL Mode 2 Result") -> dict[str, Any]:
    payload = {
        "schema_name": "smpljax.mode2.visualization",
        "schema_version": "1.0",
        "title": title,
        "implementation_status": result.implementation_status,
        "summary": result.summary,
        "metadata": result.metadata,
        "phases": [
            {
                "stage_index": phase.stage_index,
                "stage_name": phase.stage_name,
                "steps": phase.steps,
                "trainable_keys": list(phase.trainable_keys),
                "initial_objective": phase.initial_objective,
                "final_objective": phase.final_objective,
                "initial_grad_norm": phase.initial_grad_norm,
                "final_grad_norm": phase.final_grad_norm,
                "status": phase.status,
            }
            for phase in result.phase_summaries
        ],
    }
    if result.output is not None and result.params is not None:
        payload["vertices"] = np.asarray(result.output.vertices[0]).tolist()
        payload["joints"] = np.asarray(result.output.joints[0]).tolist()
        payload["params"] = {key: np.asarray(value).tolist() for key, value in result.params.items()}
    return payload


def export_mode2_artifacts(output_dir: str | Path, result: SMPLMode2OptimizationResult, *, prefix: str = "smpl_mode2") -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    history_npz_path = out_dir / f"{prefix}_history.npz"
    history_json_path = out_dir / f"{prefix}_history.json"
    history_csv_path = out_dir / f"{prefix}_history.csv"
    phases_path = out_dir / f"{prefix}_phases.json"

    metrics = mode2_metrics_payload(result)
    history = mode2_history_payload(result)
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(viewer_path, build_mode2_visualization_payload(result))
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
        phases_path,
        [
            {
                "stage_index": phase.stage_index,
                "stage_name": phase.stage_name,
                "steps": phase.steps,
                "trainable_keys": list(phase.trainable_keys),
                "initial_objective": phase.initial_objective,
                "final_objective": phase.final_objective,
                "initial_grad_norm": phase.initial_grad_norm,
                "final_grad_norm": phase.final_grad_norm,
                "status": phase.status,
            }
            for phase in result.phase_summaries
        ],
    )
    return {
        "metrics": metrics_path,
        "viewer_payload": viewer_path,
        "history": history_npz_path,
        "history_json": history_json_path,
        "history_csv": history_csv_path,
        "phases": phases_path,
    }


def _mode1_compatible_result(result: SMPLMode2OptimizationResult) -> SMPLMode1OptimizationResult:
    if result.output is None or result.params is None or result.faces is None or result.parents is None:
        raise NotImplementedError("SMPL Mode 2 visualization requires geometry payloads.")
    if result.objective_history is None or result.grad_norm_history is None:
        raise NotImplementedError("SMPL Mode 2 visualization requires objective and gradient history.")
    return SMPLMode1OptimizationResult(
        params=result.params,
        output=result.output,
        faces=result.faces,
        parents=result.parents,
        objective_history=result.objective_history,
        grad_norm_history=result.grad_norm_history,
        step_diagnostics=result.step_diagnostics,
    )


def visualize_mode2_result(result: SMPLMode2OptimizationResult, *, backend: str = "matplotlib", **kwargs):
    from .visualization.mode1 import visualize_mode1_result as _visualize_mode1_result

    return _visualize_mode1_result(_mode1_compatible_result(result), backend=backend, **kwargs)

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .api import RuntimeMode, create_runtime
from .body_models import ModelOutput, SMPLJAXModel
from .disk import atomic_write_csv, atomic_write_json, atomic_write_npz
from .mode_snapshot import export_mode1_snapshot
from .optimized import CachePolicy, ForwardInputs, OptimizedSMPLJAX


Mode1ModelLike = SMPLJAXModel | OptimizedSMPLJAX
Mode1ParamTree = Mapping[str, Any] | ForwardInputs
Mode1ObjectiveFn = Callable[[Mode1ModelLike, Mapping[str, jax.Array], ModelOutput], jax.Array]


class SMPLMode1StepDiagnostics(NamedTuple):
    step: int
    objective: float
    grad_norm: float
    vertex_rms: float
    joint_rms: float
    transl_norm: float


class SMPLMode1OptimizationResult(NamedTuple):
    params: dict[str, jax.Array]
    output: ModelOutput
    faces: jax.Array
    parents: jax.Array
    objective_history: jax.Array
    grad_norm_history: jax.Array
    step_diagnostics: tuple[SMPLMode1StepDiagnostics, ...]


class SMPLMode1Provision(NamedTuple):
    model: Mode1ModelLike
    params: dict[str, jax.Array]
    runtime_mode: str
    model_path: str | None


class SMPLMode1WorkflowRun(NamedTuple):
    provision: SMPLMode1Provision
    result: SMPLMode1OptimizationResult
    artifacts: dict[str, Path]
    viewer: Any | None = None


def _coerce_params(params: Mode1ParamTree) -> dict[str, jax.Array]:
    if isinstance(params, ForwardInputs):
        return {
            "betas": params.betas,
            "body_pose": params.body_pose,
            "global_orient": params.global_orient,
            "transl": params.transl,
            "expression": params.expression,
            "jaw_pose": params.jaw_pose,
            "leye_pose": params.leye_pose,
            "reye_pose": params.reye_pose,
            "left_hand_pose": params.left_hand_pose,
            "right_hand_pose": params.right_hand_pose,
        }
    return {str(key): jnp.asarray(value) for key, value in params.items()}


def _emit_progress(message: str, *, progress: bool) -> None:
    if progress:
        print(f"[smpljax] {message}")


def _model_dtype(model: Mode1ModelLike) -> jnp.dtype:
    return jnp.asarray(model.data.v_template).dtype


def _num_body_joints(model: Mode1ModelLike) -> int:
    data = model.data
    if data.num_body_joints is not None:
        return int(data.num_body_joints)
    total_joints = int(np.asarray(data.parents).shape[0])
    return max(total_joints - 1 - int(data.num_hand_joints or 0) * 2 - int(data.num_face_joints or 0), 0)


def default_mode1_params(
    model: Mode1ModelLike,
    *,
    batch_size: int = 1,
    transl: jax.Array | None = None,
) -> dict[str, jax.Array]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    data = model.data
    dtype = _model_dtype(model)
    num_betas = int(data.num_betas or np.asarray(data.shapedirs).shape[-1])
    num_expr = int(data.num_expression_coeffs or 0)
    num_body = _num_body_joints(model)
    num_hand = int(data.num_hand_joints or 0)
    num_face = int(data.num_face_joints or 0)

    params: dict[str, jax.Array] = {
        "betas": jnp.zeros((batch_size, num_betas), dtype=dtype),
        "body_pose": jnp.zeros((batch_size, num_body, 3), dtype=dtype),
        "global_orient": jnp.zeros((batch_size, 1, 3), dtype=dtype),
        "transl": jnp.zeros((batch_size, 3), dtype=dtype) if transl is None else jnp.asarray(transl, dtype=dtype),
    }
    if num_expr > 0:
        params["expression"] = jnp.zeros((batch_size, num_expr), dtype=dtype)
    if num_face > 0:
        params["jaw_pose"] = jnp.zeros((batch_size, 1, 3), dtype=dtype)
        params["leye_pose"] = jnp.zeros((batch_size, 1, 3), dtype=dtype)
        params["reye_pose"] = jnp.zeros((batch_size, 1, 3), dtype=dtype)
    if num_hand > 0:
        params["left_hand_pose"] = jnp.zeros((batch_size, num_hand, 3), dtype=dtype)
        params["right_hand_pose"] = jnp.zeros((batch_size, num_hand, 3), dtype=dtype)
    return params


def initialize_mode1_model(
    *,
    model_path: str | Path | None = None,
    model: Mode1ModelLike | None = None,
    runtime_mode: RuntimeMode = "optimized",
    io_cache_entries: int = 2,
    dtype: str | None = None,
    cache_policy: CachePolicy | None = None,
    batch_size: int = 1,
    params: Mode1ParamTree | None = None,
    transl: jax.Array | None = None,
    progress: bool = True,
) -> SMPLMode1Provision:
    if model is None and model_path is None:
        raise ValueError("Either model_path or model must be provided")
    if model is not None and model_path is not None:
        raise ValueError("Provide either model_path or model, not both")

    runtime_label = runtime_mode if model is None else type(model).__name__
    _emit_progress(f"initializing smpl mode1 runtime: {runtime_label}", progress=progress)
    runtime = model if model is not None else create_runtime(
        model_path,
        mode=runtime_mode,
        io_cache_entries=io_cache_entries,
        dtype=dtype,
        cache_policy=cache_policy,
    )
    base_params = default_mode1_params(runtime, batch_size=batch_size, transl=transl)
    if params is not None:
        base_params.update(_coerce_params(params))
    _emit_progress(
        f"smpl mode1 runtime ready: batch={int(next(iter(base_params.values())).shape[0])}",
        progress=progress,
    )
    return SMPLMode1Provision(
        model=runtime,
        params=base_params,
        runtime_mode=runtime_mode if model is None else type(model).__name__,
        model_path=None if model_path is None else str(model_path),
    )


def _tree_l2_norm(tree: Mapping[str, jax.Array]) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(jnp.asarray(leaf))) for leaf in leaves))


def _forward(model: Mode1ModelLike, params: Mapping[str, jax.Array]) -> ModelOutput:
    clean = {key: value for key, value in params.items() if value is not None}
    if isinstance(model, OptimizedSMPLJAX):
        prepared = model.prepare_inputs(
            batch_size=int(next(iter(clean.values())).shape[0]),
            betas=clean.get("betas"),
            body_pose=clean.get("body_pose"),
            global_orient=clean.get("global_orient"),
            transl=clean.get("transl"),
            expression=clean.get("expression"),
            jaw_pose=clean.get("jaw_pose"),
            leye_pose=clean.get("leye_pose"),
            reye_pose=clean.get("reye_pose"),
            left_hand_pose=clean.get("left_hand_pose"),
            right_hand_pose=clean.get("right_hand_pose"),
            pose2rot=bool(clean.get("pose2rot", True)) if "pose2rot" in clean else True,
        )
        return model.forward(prepared, pose2rot=True, return_full_pose=False)
    kwargs = {key: value for key, value in clean.items() if key != "pose2rot"}
    return model(**kwargs, pose2rot=bool(clean.get("pose2rot", True)) if "pose2rot" in clean else True)


def default_mode1_objective(model: Mode1ModelLike, params: Mapping[str, jax.Array], output: ModelOutput) -> jax.Array:
    del model
    transl = params.get("transl")
    transl_penalty = jnp.asarray(0.0, dtype=output.vertices.dtype) if transl is None else jnp.mean(jnp.square(transl))
    return jnp.mean(jnp.square(output.vertices)) + 1.0e-2 * transl_penalty


def mode1_history_payload(result: SMPLMode1OptimizationResult) -> dict[str, np.ndarray]:
    steps = np.arange(1, int(result.objective_history.shape[0]) + 1, dtype=np.int32)
    return {
        "step": steps,
        "objective_history": np.asarray(result.objective_history),
        "grad_norm_history": np.asarray(result.grad_norm_history),
    }


def mode1_metrics_payload(result: SMPLMode1OptimizationResult) -> dict[str, Any]:
    vertices = np.asarray(result.output.vertices)
    joints = np.asarray(result.output.joints)
    initial_objective = float(result.objective_history[0])
    final_objective = float(result.objective_history[-1])
    initial_grad = float(result.grad_norm_history[0])
    final_grad = float(result.grad_norm_history[-1])
    return {
        "schema_name": "smpljax.mode1.metrics",
        "schema_version": "1.0",
        "n_steps": int(result.objective_history.shape[0]),
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "objective_drop": initial_objective - final_objective,
        "initial_grad_norm": initial_grad,
        "final_grad_norm": final_grad,
        "grad_norm_drop": initial_grad - final_grad,
        "n_vertices": int(vertices.shape[-2]),
        "n_joints": int(joints.shape[-2]),
        "batch_size": int(vertices.shape[0]),
        "vertex_rms": float(np.sqrt(np.mean(np.square(vertices)))),
        "joint_rms": float(np.sqrt(np.mean(np.square(joints)))),
        "status": "converged" if final_grad <= 1.0e-6 else "improving",
    }


def summarize_mode1_result(result: SMPLMode1OptimizationResult) -> dict[str, Any]:
    metrics = mode1_metrics_payload(result)
    return {
        "n_steps": metrics["n_steps"],
        "n_vertices": metrics["n_vertices"],
        "n_joints": metrics["n_joints"],
        "final_objective": metrics["final_objective"],
        "final_grad_norm": metrics["final_grad_norm"],
        "status": metrics["status"],
    }


def build_mode1_visualization_payload(
    result: SMPLMode1OptimizationResult,
    *,
    title: str = "SMPL Mode 1 Result",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mesh = result.output.vertices[0]
    joints = result.output.joints[0]
    params = {key: np.asarray(value).tolist() for key, value in result.params.items()}
    return {
        "schema_name": "smpljax.mode1.visualization",
        "schema_version": "1.0",
        "title": title,
        "vertices": np.asarray(mesh).tolist(),
        "joints": np.asarray(joints).tolist(),
        "params": params,
        "metrics": {
            str(key): (value.item() if isinstance(value, np.generic) else value)
            for key, value in (metrics or mode1_metrics_payload(result)).items()
        },
    }


def optimize_mode1(
    model: Mode1ModelLike,
    params: Mode1ParamTree,
    *,
    objective_fn: Mode1ObjectiveFn | None = None,
    steps: int = 40,
    step_size: float = 1.0e-2,
    diagnostics_every: int = 10,
    project_fn: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None = None,
) -> SMPLMode1OptimizationResult:
    objective = default_mode1_objective if objective_fn is None else objective_fn
    current = _coerce_params(params)
    objective_history: list[jax.Array] = []
    grad_norm_history: list[jax.Array] = []
    diagnostics: list[SMPLMode1StepDiagnostics] = []

    def _loss_fn(p: Mapping[str, jax.Array]) -> jax.Array:
        out = _forward(model, p)
        return objective(model, p, out)

    loss_and_grad = jax.value_and_grad(_loss_fn)

    for step in range(1, steps + 1):
        loss, grads = loss_and_grad(current)
        grad_norm = _tree_l2_norm(grads)
        objective_history.append(loss)
        grad_norm_history.append(grad_norm)
        current = jax.tree_util.tree_map(lambda p, g: p - jnp.asarray(step_size, dtype=p.dtype) * g, current, grads)
        if project_fn is not None:
            current = project_fn(current)
        if diagnostics_every > 0 and (step == 1 or step == steps or step % diagnostics_every == 0):
            out = _forward(model, current)
            transl = current.get("transl", jnp.zeros((1, 3), dtype=out.vertices.dtype))
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

    final_output = _forward(model, current)
    return SMPLMode1OptimizationResult(
        params=current,
        output=final_output,
        faces=jnp.asarray(model.data.faces_tensor, dtype=jnp.int32),
        parents=jnp.asarray(model.data.parents, dtype=jnp.int32),
        objective_history=jnp.asarray(objective_history),
        grad_norm_history=jnp.asarray(grad_norm_history),
        step_diagnostics=tuple(diagnostics),
    )


def export_mode1_artifacts(
    output_dir: str | Path,
    result: SMPLMode1OptimizationResult,
    *,
    prefix: str = "smpl_mode1",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = mode1_metrics_payload(result)
    history = mode1_history_payload(result)
    viewer_payload = build_mode1_visualization_payload(result, metrics=metrics)

    snapshot_path = out_dir / f"{prefix}_snapshot.npz"
    metrics_path = out_dir / f"{prefix}_metrics.json"
    history_npz_path = out_dir / f"{prefix}_history.npz"
    history_json_path = out_dir / f"{prefix}_history.json"
    history_csv_path = out_dir / f"{prefix}_history.csv"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    topo_snapshot_path = out_dir / f"{prefix}_snapshot.mode1.npz"

    atomic_write_npz(
        snapshot_path,
        vertices=np.asarray(result.output.vertices),
        joints=np.asarray(result.output.joints),
        faces=np.asarray(result.faces),
        parents=np.asarray(result.parents),
        objective_history=np.asarray(result.objective_history),
        grad_norm_history=np.asarray(result.grad_norm_history),
        **{f"param_{key}": np.asarray(value) for key, value in result.params.items()},
    )
    atomic_write_json(metrics_path, metrics)
    atomic_write_npz(history_npz_path, **history)
    atomic_write_json(
        history_json_path,
        [
            {"step": int(step), "objective": float(obj), "grad_norm": float(grad)}
            for step, obj, grad in zip(history["step"], history["objective_history"], history["grad_norm_history"])
        ],
    )
    atomic_write_csv(
        history_csv_path,
        [
            {"step": int(step), "objective": float(obj), "grad_norm": float(grad)}
            for step, obj, grad in zip(history["step"], history["objective_history"], history["grad_norm_history"])
        ],
    )
    atomic_write_json(viewer_path, viewer_payload)
    export_mode1_snapshot(
        topo_snapshot_path,
        params={key: np.asarray(value) for key, value in result.params.items()},
        vertices=np.asarray(result.output.vertices),
        joints=np.asarray(result.output.joints),
        faces=np.asarray(result.faces),
        parents=np.asarray(result.parents),
        objective_history=np.asarray(result.objective_history),
        grad_norm_history=np.asarray(result.grad_norm_history),
        metrics=metrics,
        visualization_payload=viewer_payload,
    )
    return {
        "snapshot": snapshot_path,
        "mode1_snapshot": topo_snapshot_path,
        "metrics": metrics_path,
        "history": history_npz_path,
        "history_json": history_json_path,
        "history_csv": history_csv_path,
        "viewer_payload": viewer_path,
    }


def run_mode1_workflow(
    provision: SMPLMode1Provision,
    *,
    output_dir: str | Path,
    prefix: str = "smpl_mode1",
    params: Mode1ParamTree | None = None,
    objective_fn: Mode1ObjectiveFn | None = None,
    steps: int = 40,
    step_size: float = 1.0e-2,
    diagnostics_every: int = 10,
    project_fn: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None = None,
    visualize_backend: str | None = None,
    title: str = "SMPL Mode 1 Result",
    show: bool = False,
    host: str = "127.0.0.1",
    port: int = 8090,
    progress: bool = True,
) -> SMPLMode1WorkflowRun:
    active_params = dict(provision.params)
    if params is not None:
        active_params.update(_coerce_params(params))
    batch_size = int(next(iter(active_params.values())).shape[0]) if active_params else 0
    _emit_progress(f"mode1 optimize start: batch={batch_size} steps={steps}", progress=progress)
    result = optimize_mode1(
        provision.model,
        active_params,
        objective_fn=objective_fn,
        steps=steps,
        step_size=step_size,
        diagnostics_every=diagnostics_every,
        project_fn=project_fn,
    )
    _emit_progress(f"mode1 export start: {output_dir}", progress=progress)
    artifacts = export_mode1_artifacts(output_dir, result, prefix=prefix)
    viewer = None
    if visualize_backend is not None:
        _emit_progress(f"mode1 visualize start: backend={visualize_backend}", progress=progress)
        from .visualization.mode1 import visualize_mode1_result as _visualize_mode1_result

        viewer = _visualize_mode1_result(result, backend=visualize_backend, title=title, show=show, host=host, port=port)
    _emit_progress(f"mode1 complete: snapshot={artifacts['mode1_snapshot']}", progress=progress)
    return SMPLMode1WorkflowRun(
        provision=provision,
        result=result,
        artifacts=artifacts,
        viewer=viewer,
    )

"""Straight-through estimators for discrete connectivity choices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
import json

import jax
import jax.numpy as jnp
import numpy as np

from topojax.ad.surrogate import (
    _binary_soft_weights,
    quad_split_candidate_qualities,
    soft_tet_connectivity_energy,
    soft_triangle_connectivity_energy,
    tet_split_candidate_qualities,
    triangle_flip_candidate_patches,
    triangle_flip_candidate_qualities,
)
from topojax.io.exports import GmshElementBlock, export_gmsh_msh, export_metrics_json


def _binary_straight_through_weights(logits: jnp.ndarray, temperature: float = 0.25) -> jnp.ndarray:
    soft = _binary_soft_weights(logits, temperature=temperature)
    hard_index = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(hard_index, 2, dtype=soft.dtype)
    return hard + soft - jax.lax.stop_gradient(soft)


def straight_through_quad_diagonal_weights(logits: jnp.ndarray, temperature: float = 0.25) -> jnp.ndarray:
    """Hard forward quad split choice with soft backward gradients."""
    return _binary_straight_through_weights(logits, temperature=temperature)


def straight_through_triangle_flip_weights(logits: jnp.ndarray, temperature: float = 0.25) -> jnp.ndarray:
    """Hard forward triangle flip choice with soft backward gradients."""
    return _binary_straight_through_weights(logits, temperature=temperature)


def straight_through_tet_split_weights(logits: jnp.ndarray, temperature: float = 0.25) -> jnp.ndarray:
    """Hard forward tetra split choice with soft backward gradients."""
    return _binary_straight_through_weights(logits, temperature=temperature)


def _straight_through_energy_from_qualities(
    current_quality: jnp.ndarray,
    candidate_quality: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float,
    quality_target: float,
) -> jnp.ndarray:
    if current_quality.shape[0] == 0:
        return jnp.asarray(0.0, dtype=logits.dtype if logits.shape[0] else jnp.float32)
    weights = _binary_straight_through_weights(logits, temperature=temperature)
    mixed_quality = weights[:, 0] * candidate_quality + weights[:, 1] * current_quality
    return jnp.mean(jax.nn.softplus((quality_target - mixed_quality) * 10.0))


def straight_through_quad_connectivity_energy(
    points: jnp.ndarray,
    quads: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Use hard split choices in the forward pass and soft gradients in backward."""
    current_quality, candidate_quality = quad_split_candidate_qualities(points, quads)
    return _straight_through_energy_from_qualities(
        current_quality,
        candidate_quality,
        logits,
        temperature=temperature,
        quality_target=0.7,
    )


def straight_through_triangle_connectivity_energy(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    candidate_patches: jnp.ndarray | None = None,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Use hard triangle flip choices in the forward pass and soft gradients in backward."""
    patches = triangle_flip_candidate_patches(elements) if candidate_patches is None else jnp.asarray(candidate_patches, dtype=jnp.int32)
    current_quality, candidate_quality = triangle_flip_candidate_qualities(points, patches)
    return _straight_through_energy_from_qualities(
        current_quality,
        candidate_quality,
        logits,
        temperature=temperature,
        quality_target=0.65,
    )


def straight_through_tet_connectivity_energy(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Use hard tet split choices in the forward pass and soft gradients in backward."""
    current_quality, candidate_quality = tet_split_candidate_qualities(points, elements)
    return _straight_through_energy_from_qualities(
        current_quality,
        candidate_quality,
        logits,
        temperature=temperature,
        quality_target=0.35,
    )


class Mode4OptimizationResult(NamedTuple):
    points: jnp.ndarray
    elements: jnp.ndarray
    logits: jnp.ndarray
    hard_weights: jnp.ndarray
    objective_history: jnp.ndarray
    grad_norm_history: jnp.ndarray
    temperature: float
    topology_kind: str
    candidate_kind: str


def _optimize_straight_through_connectivity(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    logits: jnp.ndarray | None,
    steps: int,
    step_size: float,
    temperature: float,
    topology_kind: str,
    candidate_kind: str,
    objective_fn,
) -> Mode4OptimizationResult:
    pts = jnp.asarray(points)
    elem_arr = jnp.asarray(elements, dtype=jnp.int32)
    candidate_count = int(objective_fn("candidate_count"))
    theta = jnp.zeros((candidate_count,), dtype=pts.dtype) if logits is None else jnp.asarray(logits, dtype=pts.dtype)
    if theta.shape != (candidate_count,):
        raise ValueError(f"logits must have shape ({candidate_count},), got {theta.shape}")

    objective_history: list[jax.Array] = []
    grad_norm_history: list[jax.Array] = []

    def theta_objective(local_logits: jnp.ndarray) -> jnp.ndarray:
        return objective_fn(local_logits)

    value_and_grad = jax.value_and_grad(theta_objective)
    for _ in range(steps):
        value, grad = value_and_grad(theta)
        objective_history.append(value)
        grad_norm_history.append(jnp.linalg.norm(grad))
        theta = theta - jnp.asarray(step_size, dtype=pts.dtype) * grad

    return Mode4OptimizationResult(
        points=pts,
        elements=elem_arr,
        logits=theta,
        hard_weights=_binary_straight_through_weights(theta, temperature=temperature),
        objective_history=jnp.asarray(objective_history),
        grad_norm_history=jnp.asarray(grad_norm_history),
        temperature=float(temperature),
        topology_kind=topology_kind,
        candidate_kind=candidate_kind,
    )


def optimize_straight_through_quad_connectivity(
    points: jnp.ndarray,
    quads: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode4OptimizationResult:
    quad_arr = jnp.asarray(quads, dtype=jnp.int32)
    return _optimize_straight_through_connectivity(
        points,
        quad_arr,
        logits=logits,
        steps=steps,
        step_size=step_size,
        temperature=temperature,
        topology_kind="quad",
        candidate_kind="quad-diagonal",
        objective_fn=lambda theta: quad_arr.shape[0]
        if isinstance(theta, str)
        else straight_through_quad_connectivity_energy(jnp.asarray(points), quad_arr, theta, temperature=temperature),
    )


def optimize_straight_through_triangle_connectivity(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode4OptimizationResult:
    elem_arr = jnp.asarray(elements, dtype=jnp.int32)
    patches = triangle_flip_candidate_patches(elem_arr)

    def objective(theta):
        if isinstance(theta, str):
            return int(patches.shape[0])
        return straight_through_triangle_connectivity_energy(
            jnp.asarray(points),
            elem_arr,
            theta,
            candidate_patches=patches,
            temperature=temperature,
        )

    return _optimize_straight_through_connectivity(
        points,
        elem_arr,
        logits=logits,
        steps=steps,
        step_size=step_size,
        temperature=temperature,
        topology_kind="triangle",
        candidate_kind="triangle-edge-flip",
        objective_fn=objective,
    )


def optimize_straight_through_tet_connectivity(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode4OptimizationResult:
    elem_arr = jnp.asarray(elements, dtype=jnp.int32)
    return _optimize_straight_through_connectivity(
        points,
        elem_arr,
        logits=logits,
        steps=steps,
        step_size=step_size,
        temperature=temperature,
        topology_kind="tetra",
        candidate_kind="tetra-split",
        objective_fn=lambda theta: elem_arr.shape[0]
        if isinstance(theta, str)
        else straight_through_tet_connectivity_energy(jnp.asarray(points), elem_arr, theta, temperature=temperature),
    )


def optimize_straight_through_connectivity(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode4OptimizationResult:
    pts = jnp.asarray(points)
    elem_arr = jnp.asarray(elements, dtype=jnp.int32)
    order = int(elem_arr.shape[1])
    dim = int(pts.shape[1])
    if order == 3 and dim == 2:
        return optimize_straight_through_triangle_connectivity(
            pts,
            elem_arr,
            logits=logits,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
        )
    if order == 4 and dim == 2:
        return optimize_straight_through_quad_connectivity(
            pts,
            elem_arr,
            logits=logits,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
        )
    if order == 4 and dim == 3:
        return optimize_straight_through_tet_connectivity(
            pts,
            elem_arr,
            logits=logits,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
        )
    raise ValueError("Mode 4 currently supports 2D triangle, 2D quad, and 3D tetra domains only")


def summarize_mode4_result(result: Mode4OptimizationResult) -> dict[str, Any]:
    metrics = {
        "topology_kind": result.topology_kind,
        "candidate_kind": result.candidate_kind,
        "n_elements": int(result.elements.shape[0]),
        "n_candidates": int(result.logits.shape[0]),
        "n_steps": int(result.objective_history.shape[0]),
        "initial_objective": float(result.objective_history[0]),
        "final_objective": float(result.objective_history[-1]),
        "initial_grad_norm": float(result.grad_norm_history[0]),
        "final_grad_norm": float(result.grad_norm_history[-1]),
        "temperature": result.temperature,
    }
    if result.topology_kind == "triangle":
        metrics["n_triangles"] = int(result.elements.shape[0])
    elif result.topology_kind == "quad":
        metrics["n_quads"] = int(result.elements.shape[0])
    elif result.topology_kind == "tetra":
        metrics["n_tets"] = int(result.elements.shape[0])
    return metrics


def mode4_history_payload(result: Mode4OptimizationResult) -> dict[str, np.ndarray]:
    return {
        "step": np.arange(1, int(result.objective_history.shape[0]) + 1, dtype=np.int32),
        "objective_history": np.asarray(result.objective_history),
        "grad_norm_history": np.asarray(result.grad_norm_history),
    }


def export_mode4_artifacts(
    output_dir: str | Path,
    result: Mode4OptimizationResult,
    *,
    prefix: str = "mode4",
    extra_element_blocks: tuple[GmshElementBlock, ...] | None = None,
    physical_names: dict[tuple[int, int], str] | None = None,
) -> dict[str, Path]:
    from topojax.visualization import build_mode4_visualization_payload

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize_mode4_result(result)
    history = mode4_history_payload(result)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    history_path = out_dir / f"{prefix}_history.npz"
    history_json_path = out_dir / f"{prefix}_history.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    mesh_path = out_dir / f"{prefix}_mesh.msh"
    export_metrics_json(metrics_path, metrics)
    np.savez(history_path, **history)
    history_rows = [
        {"step": int(step), "objective": float(obj), "grad_norm": float(grad)}
        for step, obj, grad in zip(history["step"], history["objective_history"], history["grad_norm_history"])
    ]
    history_json_path.write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
    viewer_payload = build_mode4_visualization_payload(
        points=result.points,
        elements=result.elements,
        metrics=metrics,
        candidate_logits=result.logits,
        implementation_status="implemented",
    )
    viewer_path.write_text(json.dumps(viewer_payload, indent=2, sort_keys=True), encoding="utf-8")
    export_gmsh_msh(
        mesh_path,
        result.points,
        result.elements,
        element_kind=result.topology_kind,
        extra_element_blocks=extra_element_blocks,
        physical_names=physical_names,
    )
    return {
        "metrics": metrics_path,
        "history": history_path,
        "history_json": history_json_path,
        "viewer_payload": viewer_path,
        "mesh": mesh_path,
    }

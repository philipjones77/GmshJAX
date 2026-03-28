"""Differentiable connectivity surrogates on fixed candidate graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
import json

import jax
import jax.numpy as jnp
import numpy as np

from topojax.io.exports import GmshElementBlock, export_gmsh_msh, export_metrics_json


def _tri_area2(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _tri_quality(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    e01 = b - a
    e12 = c - b
    e20 = a - c
    area2 = _tri_area2(a, b, c)
    edge2 = jnp.sum(e01 * e01, axis=-1) + jnp.sum(e12 * e12, axis=-1) + jnp.sum(e20 * e20, axis=-1)
    return 2.0 * jnp.sqrt(3.0) * jnp.abs(area2) / jnp.maximum(edge2, 1.0e-12)


def _tet_quality(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    e1 = b - a
    e2 = c - a
    e3 = d - a
    det = (
        e1[..., 0] * (e2[..., 1] * e3[..., 2] - e2[..., 2] * e3[..., 1])
        - e1[..., 1] * (e2[..., 0] * e3[..., 2] - e2[..., 2] * e3[..., 0])
        + e1[..., 2] * (e2[..., 0] * e3[..., 1] - e2[..., 1] * e3[..., 0])
    )
    fro2 = jnp.sum(e1 * e1, axis=-1) + jnp.sum(e2 * e2, axis=-1) + jnp.sum(e3 * e3, axis=-1)
    signed_pow = jnp.sign(det) * jnp.power(jnp.maximum(jnp.abs(det), 1.0e-12), 2.0 / 3.0)
    return 3.0 * signed_pow / jnp.maximum(fro2, 1.0e-12)


def _binary_soft_weights(logits: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    pair_logits = jnp.stack([logits, jnp.zeros_like(logits)], axis=-1)
    return jax.nn.softmax(pair_logits / jnp.maximum(temperature, 1.0e-6), axis=-1)


def triangle_flip_candidate_patches(elements: jnp.ndarray) -> jnp.ndarray:
    """Return one 4-node patch per interior triangle edge.

    Patch layout is `[opp_a, shared_0, opp_b, shared_1]`.
    """
    elems = np.asarray(elements, dtype=np.int32)
    edge_to_info: dict[tuple[int, int], tuple[int, int, int]] = {}
    patches: list[list[int]] = []

    for elem_idx, tri in enumerate(elems.tolist()):
        a, b, c = [int(v) for v in tri]
        for shared, opposite in (((a, b), c), ((b, c), a), ((c, a), b)):
            key = tuple(sorted(shared))
            if key in edge_to_info:
                _, other_opposite, _ = edge_to_info[key]
                if len({other_opposite, opposite, key[0], key[1]}) == 4:
                    patches.append([other_opposite, key[0], opposite, key[1]])
            else:
                edge_to_info[key] = (elem_idx, opposite, len(patches))

    if not patches:
        return jnp.zeros((0, 4), dtype=jnp.int32)
    return jnp.asarray(np.asarray(patches, dtype=np.int32), dtype=jnp.int32)


def triangle_flip_candidate_qualities(points: jnp.ndarray, patches: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return current and flipped quality for triangle edge-flip candidates."""
    if patches.shape[0] == 0:
        empty = jnp.zeros((0,), dtype=points.dtype)
        return empty, empty

    p = points[patches]
    u = p[:, 0, :]
    s0 = p[:, 1, :]
    v = p[:, 2, :]
    s1 = p[:, 3, :]

    current_quality = 0.5 * (_tri_quality(u, s0, s1) + _tri_quality(v, s0, s1))
    flipped_quality = 0.5 * (_tri_quality(u, s0, v) + _tri_quality(u, v, s1))
    return current_quality, flipped_quality


def quad_split_candidate_qualities(points: jnp.ndarray, quads: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-quad quality for the two diagonal split candidates."""
    if quads.shape[0] == 0:
        empty = jnp.zeros((0,), dtype=points.dtype)
        return empty, empty

    p = points[quads]
    a = p[:, 0, :]
    b = p[:, 1, :]
    c = p[:, 2, :]
    d = p[:, 3, :]

    ac_quality = 0.5 * (_tri_quality(a, b, c) + _tri_quality(a, c, d))
    bd_quality = 0.5 * (_tri_quality(a, b, d) + _tri_quality(b, c, d))
    return ac_quality, bd_quality


def tet_split_candidate_qualities(points: jnp.ndarray, tets: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return current and split quality for tetrahedral split candidates."""
    if tets.shape[0] == 0:
        empty = jnp.zeros((0,), dtype=points.dtype)
        return empty, empty

    p = points[tets]
    a = p[:, 0, :]
    b = p[:, 1, :]
    c = p[:, 2, :]
    d = p[:, 3, :]
    center = 0.25 * (a + b + c + d)

    keep_quality = _tet_quality(a, b, c, d)
    split_quality = 0.25 * (
        _tet_quality(a, b, c, center)
        + _tet_quality(a, b, center, d)
        + _tet_quality(a, center, c, d)
        + _tet_quality(center, b, c, d)
    )
    return keep_quality, split_quality


def soft_quad_diagonal_weights(logits: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Differentiable weights over the two quad diagonal choices."""
    return _binary_soft_weights(logits, temperature=temperature)


def soft_triangle_flip_weights(logits: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Differentiable weights over keep-vs-flip triangle edge candidates."""
    return _binary_soft_weights(logits, temperature=temperature)


def soft_tet_split_weights(logits: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Differentiable weights over keep-vs-split tetra candidates."""
    return _binary_soft_weights(logits, temperature=temperature)


def _soft_connectivity_energy_from_qualities(
    current_quality: jnp.ndarray,
    candidate_quality: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float,
    quality_target: float,
    entropy_weight: float = 1.0e-3,
) -> jnp.ndarray:
    if current_quality.shape[0] == 0:
        return jnp.asarray(0.0, dtype=logits.dtype if logits.shape[0] else jnp.float32)

    weights = _binary_soft_weights(logits, temperature=temperature)
    mixed_quality = weights[:, 0] * candidate_quality + weights[:, 1] * current_quality
    entropy = -jnp.sum(weights * jnp.log(jnp.maximum(weights, 1.0e-12)), axis=-1)
    return jnp.mean(jax.nn.softplus((quality_target - mixed_quality) * 10.0)) + entropy_weight * jnp.mean(entropy)


def soft_quad_connectivity_energy(
    points: jnp.ndarray,
    quads: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Differentiable surrogate energy over a fixed quad candidate graph."""
    current_quality, candidate_quality = quad_split_candidate_qualities(points, quads)
    return _soft_connectivity_energy_from_qualities(
        current_quality,
        candidate_quality,
        logits,
        temperature=temperature,
        quality_target=0.7,
    )


def soft_triangle_connectivity_energy(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    candidate_patches: jnp.ndarray | None = None,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Differentiable surrogate energy over interior triangle edge-flip candidates."""
    patches = triangle_flip_candidate_patches(elements) if candidate_patches is None else jnp.asarray(candidate_patches, dtype=jnp.int32)
    current_quality, candidate_quality = triangle_flip_candidate_qualities(points, patches)
    return _soft_connectivity_energy_from_qualities(
        current_quality,
        candidate_quality,
        logits,
        temperature=temperature,
        quality_target=0.65,
    )


def soft_tet_connectivity_energy(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Differentiable surrogate energy over tetra keep-vs-split candidates."""
    current_quality, candidate_quality = tet_split_candidate_qualities(points, elements)
    return _soft_connectivity_energy_from_qualities(
        current_quality,
        candidate_quality,
        logits,
        temperature=temperature,
        quality_target=0.35,
    )


class Mode3OptimizationResult(NamedTuple):
    points: jnp.ndarray
    elements: jnp.ndarray
    logits: jnp.ndarray
    weights: jnp.ndarray
    objective_history: jnp.ndarray
    grad_norm_history: jnp.ndarray
    temperature: float
    topology_kind: str
    candidate_kind: str


def _optimize_soft_connectivity(
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
) -> Mode3OptimizationResult:
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

    return Mode3OptimizationResult(
        points=pts,
        elements=elem_arr,
        logits=theta,
        weights=_binary_soft_weights(theta, temperature=temperature),
        objective_history=jnp.asarray(objective_history),
        grad_norm_history=jnp.asarray(grad_norm_history),
        temperature=float(temperature),
        topology_kind=topology_kind,
        candidate_kind=candidate_kind,
    )


def optimize_soft_quad_connectivity(
    points: jnp.ndarray,
    quads: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode3OptimizationResult:
    quad_arr = jnp.asarray(quads, dtype=jnp.int32)
    return _optimize_soft_connectivity(
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
        else soft_quad_connectivity_energy(jnp.asarray(points), quad_arr, theta, temperature=temperature),
    )


def optimize_soft_triangle_connectivity(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode3OptimizationResult:
    elem_arr = jnp.asarray(elements, dtype=jnp.int32)
    patches = triangle_flip_candidate_patches(elem_arr)

    def objective(theta):
        if isinstance(theta, str):
            return int(patches.shape[0])
        return soft_triangle_connectivity_energy(jnp.asarray(points), elem_arr, theta, candidate_patches=patches, temperature=temperature)

    return _optimize_soft_connectivity(
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


def optimize_soft_tet_connectivity(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode3OptimizationResult:
    elem_arr = jnp.asarray(elements, dtype=jnp.int32)
    return _optimize_soft_connectivity(
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
        else soft_tet_connectivity_energy(jnp.asarray(points), elem_arr, theta, temperature=temperature),
    )


def optimize_soft_connectivity(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    logits: jnp.ndarray | None = None,
    steps: int = 40,
    step_size: float = 0.1,
    temperature: float = 0.25,
) -> Mode3OptimizationResult:
    pts = jnp.asarray(points)
    elem_arr = jnp.asarray(elements, dtype=jnp.int32)
    order = int(elem_arr.shape[1])
    dim = int(pts.shape[1])
    if order == 3 and dim == 2:
        return optimize_soft_triangle_connectivity(pts, elem_arr, logits=logits, steps=steps, step_size=step_size, temperature=temperature)
    if order == 4 and dim == 2:
        return optimize_soft_quad_connectivity(pts, elem_arr, logits=logits, steps=steps, step_size=step_size, temperature=temperature)
    if order == 4 and dim == 3:
        return optimize_soft_tet_connectivity(pts, elem_arr, logits=logits, steps=steps, step_size=step_size, temperature=temperature)
    raise ValueError("Mode 3 currently supports 2D triangle, 2D quad, and 3D tetra domains only")


def summarize_mode3_result(result: Mode3OptimizationResult) -> dict[str, Any]:
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


def mode3_history_payload(result: Mode3OptimizationResult) -> dict[str, np.ndarray]:
    return {
        "step": np.arange(1, int(result.objective_history.shape[0]) + 1, dtype=np.int32),
        "objective_history": np.asarray(result.objective_history),
        "grad_norm_history": np.asarray(result.grad_norm_history),
    }


def export_mode3_artifacts(
    output_dir: str | Path,
    result: Mode3OptimizationResult,
    *,
    prefix: str = "mode3",
    extra_element_blocks: tuple[GmshElementBlock, ...] | None = None,
    physical_names: dict[tuple[int, int], str] | None = None,
) -> dict[str, Path]:
    from topojax.visualization import build_mode3_visualization_payload

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize_mode3_result(result)
    history = mode3_history_payload(result)
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
    viewer_payload = build_mode3_visualization_payload(
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

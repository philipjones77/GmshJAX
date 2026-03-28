"""Relaxed Mode 5 dynamic workflows for triangle and tetrahedral meshes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
import json

import jax
import jax.numpy as jnp
import numpy as np

from topojax.ad._common import coerce_runtime_points, fit_node_mask
from topojax.ad.compiled import build_quality_value_and_grad
from topojax.ad.restart import optimize_points_fixed_topology, tet_topology_from_elements, triangle_topology_from_elements
from topojax.ad.straight_through import optimize_straight_through_connectivity
from topojax.ad.surrogate import optimize_soft_connectivity
from topojax.io.exports import GmshElementBlock, export_gmsh_msh, export_metrics_json
from topojax.mesh.adaptive import AdaptiveHistory, adaptive_remesh_tri
from topojax.mesh.adaptive_tet import TetAdaptiveHistory, adaptive_remesh_tet, tet_volume_magnitudes
from topojax.mesh.mutation import active_elements, active_points
from topojax.mesh.mutation_qt import active_tet_elements, active_tet_points
from topojax.mesh.operators import tet_icn, triangle_icn
from topojax.mesh.refine import triangle_area_magnitudes


NodeFieldMap = dict[str, jnp.ndarray]
ElementFieldMap = dict[str, jnp.ndarray]


class Mode5Phase(NamedTuple):
    cycle: int
    start_energy: float
    final_energy: float
    surrogate_initial_objective: float
    surrogate_final_objective: float
    n_nodes: int
    n_elements: int
    remeshed: bool


class Mode5ControllerDecision(NamedTuple):
    cycle: int
    topology_kind: str
    min_quality: float
    max_measure: float
    surrogate_initial_objective: float
    surrogate_final_objective: float
    remesh_triggered: bool
    reason: str


class Mode5TransferSummary(NamedTuple):
    cycle: int
    old_n_nodes: int
    new_n_nodes: int
    old_n_elements: int
    new_n_elements: int
    transferred_node_fields: tuple[str, ...]
    transferred_element_fields: tuple[str, ...]


class Mode5OptimizationResult(NamedTuple):
    points: jnp.ndarray
    elements: jnp.ndarray
    phases: list[Mode5Phase]
    controller_history: list[Mode5ControllerDecision]
    transfer_history: list[Mode5TransferSummary]
    node_fields: NodeFieldMap
    element_fields: ElementFieldMap
    topology_kind: str
    surrogate_variant: str
    implementation_status: str


def _topology_kind(points: jnp.ndarray, elements: jnp.ndarray) -> str:
    order = int(elements.shape[1])
    dim = int(points.shape[1])
    if order == 3 and dim == 2:
        return "triangle"
    if order == 4 and dim == 3:
        return "tetra"
    raise ValueError("Mode 5 currently supports 2D triangle and 3D tetra domains only")


def _topology_from_elements(points: jnp.ndarray, elements: jnp.ndarray):
    if int(elements.shape[1]) == 3:
        return triangle_topology_from_elements(elements, n_nodes=int(points.shape[0]))
    return tet_topology_from_elements(elements, n_nodes=int(points.shape[0]))


def _element_centroids(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(points[elements], axis=1)


def _nearest_source_indices(source_points: jnp.ndarray, target_points: jnp.ndarray) -> jnp.ndarray:
    src = np.asarray(source_points)
    tgt = np.asarray(target_points)
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return jnp.zeros((tgt.shape[0],), dtype=jnp.int32)
    diff = tgt[:, None, :] - src[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return jnp.asarray(np.argmin(dist2, axis=1), dtype=jnp.int32)


def transfer_node_fields_nearest(
    old_points: jnp.ndarray,
    new_points: jnp.ndarray,
    node_fields: NodeFieldMap | None,
) -> NodeFieldMap:
    if not node_fields:
        return {}
    nearest = _nearest_source_indices(old_points, new_points)
    transferred: NodeFieldMap = {}
    for name, values in node_fields.items():
        arr = jnp.asarray(values)
        if arr.shape[0] != int(old_points.shape[0]):
            raise ValueError(f"Node field '{name}' must have leading dimension {old_points.shape[0]}")
        transferred[name] = arr[nearest]
    return transferred


def transfer_element_fields_nearest(
    old_points: jnp.ndarray,
    old_elements: jnp.ndarray,
    new_points: jnp.ndarray,
    new_elements: jnp.ndarray,
    element_fields: ElementFieldMap | None,
) -> ElementFieldMap:
    if not element_fields:
        return {}
    nearest = _nearest_source_indices(_element_centroids(old_points, old_elements), _element_centroids(new_points, new_elements))
    transferred: ElementFieldMap = {}
    for name, values in element_fields.items():
        arr = jnp.asarray(values)
        if arr.shape[0] != int(old_elements.shape[0]):
            raise ValueError(f"Element field '{name}' must have leading dimension {old_elements.shape[0]}")
        transferred[name] = arr[nearest]
    return transferred


def _transfer_mask_nearest(old_points: jnp.ndarray, new_points: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray | None:
    if mask is None:
        return None
    fitted = fit_node_mask(mask, int(old_points.shape[0]))
    if fitted is None:
        return None
    nearest = _nearest_source_indices(old_points, new_points)
    return fitted[nearest]


def _quality_state(points: jnp.ndarray, elements: jnp.ndarray) -> tuple[float, float]:
    if int(elements.shape[1]) == 3:
        quality = triangle_icn(points, elements)
        measure = triangle_area_magnitudes(points, elements)
    else:
        quality = tet_icn(points, elements)
        measure = tet_volume_magnitudes(points, elements)
    return float(jnp.min(quality)), float(jnp.max(measure))


def _run_surrogate_phase(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    surrogate_variant: str,
    surrogate_steps: int,
    surrogate_step_size: float,
    temperature: float,
):
    if surrogate_variant == "soft":
        return optimize_soft_connectivity(
            points,
            elements,
            steps=surrogate_steps,
            step_size=surrogate_step_size,
            temperature=temperature,
        )
    if surrogate_variant == "straight-through":
        return optimize_straight_through_connectivity(
            points,
            elements,
            steps=surrogate_steps,
            step_size=surrogate_step_size,
            temperature=temperature,
        )
    raise ValueError("surrogate_variant must be 'soft' or 'straight-through'")


def _dynamic_controller(
    *,
    cycle: int,
    cycles: int,
    topology_kind: str,
    min_quality: float,
    max_measure: float,
    target_mean_icn: float,
    target_measure: float,
    surrogate_initial_objective: float,
    surrogate_final_objective: float,
) -> tuple[bool, str]:
    if cycle >= cycles - 1:
        return False, "final-cycle"
    if min_quality < target_mean_icn:
        return True, "quality-threshold"
    if max_measure > target_measure:
        return True, "size-threshold"
    if surrogate_final_objective < surrogate_initial_objective - 1.0e-8:
        return True, "surrogate-improvement"
    return False, f"{topology_kind}-stable"


def optimize_dynamic_topology(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    cycles: int = 2,
    optimization_steps: int = 40,
    optimization_step_size: float = 0.03,
    surrogate_variant: str = "soft",
    surrogate_steps: int = 8,
    surrogate_step_size: float = 0.1,
    temperature: float = 0.25,
    max_nodes: int,
    max_elements: int,
    remesh_max_iters: int = 2,
    target_area: float | None = None,
    target_volume: float | None = None,
    target_mean_icn: float | None = None,
    smoothing_alpha: float | None = None,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
    node_fields: NodeFieldMap | None = None,
    element_fields: ElementFieldMap | None = None,
) -> Mode5OptimizationResult:
    if cycles <= 0:
        raise ValueError("cycles must be positive")

    pts = coerce_runtime_points(points)
    elems = jnp.asarray(elements, dtype=jnp.int32)
    topology_kind = _topology_kind(pts, elems)
    phases: list[Mode5Phase] = []
    controller_history: list[Mode5ControllerDecision] = []
    transfer_history: list[Mode5TransferSummary] = []
    current_mask = fit_node_mask(movable_mask, int(pts.shape[0]))
    current_node_fields = {} if node_fields is None else {name: jnp.asarray(values) for name, values in node_fields.items()}
    current_element_fields = {} if element_fields is None else {name: jnp.asarray(values) for name, values in element_fields.items()}

    min_quality0, max_measure0 = _quality_state(pts, elems)
    target_measure = (
        (0.8 * max_measure0) if topology_kind == "triangle" else (0.8 * max_measure0)
    ) if (target_area is None and target_volume is None) else (target_area if topology_kind == "triangle" else target_volume)
    if target_measure is None:
        raise ValueError("Provide target_area for triangle workflows or target_volume for tetra workflows")
    target_quality = (
        0.50 if topology_kind == "triangle" else 0.35
    ) if target_mean_icn is None else target_mean_icn
    smooth_alpha = (0.15 if topology_kind == "triangle" else 0.12) if smoothing_alpha is None else smoothing_alpha

    for cycle in range(cycles):
        topology = _topology_from_elements(pts, elems)
        value_and_grad = build_quality_value_and_grad(topology)
        start_energy = float(value_and_grad(pts)[0])
        pts, losses = optimize_points_fixed_topology(
            pts,
            topology,
            steps=optimization_steps,
            step_size=optimization_step_size,
            movable_mask=current_mask,
        )
        final_energy = float(losses[-1]) if losses.size else start_energy

        surrogate_result = _run_surrogate_phase(
            pts,
            elems,
            surrogate_variant=surrogate_variant,
            surrogate_steps=surrogate_steps,
            surrogate_step_size=surrogate_step_size,
            temperature=temperature,
        )
        surrogate_initial = float(surrogate_result.objective_history[0]) if surrogate_result.objective_history.size else 0.0
        surrogate_final = float(surrogate_result.objective_history[-1]) if surrogate_result.objective_history.size else surrogate_initial

        min_quality, max_measure = _quality_state(pts, elems)
        remesh_triggered, reason = _dynamic_controller(
            cycle=cycle,
            cycles=cycles,
            topology_kind=topology_kind,
            min_quality=min_quality,
            max_measure=max_measure,
            target_mean_icn=target_quality,
            target_measure=float(target_measure),
            surrogate_initial_objective=surrogate_initial,
            surrogate_final_objective=surrogate_final,
        )

        old_points = pts
        old_elements = elems
        if remesh_triggered and topology_kind == "triangle":
            buffer, _ = adaptive_remesh_tri(
                pts,
                elems,
                max_nodes=max_nodes,
                max_elements=max_elements,
                max_iters=remesh_max_iters,
                target_area=float(target_measure),
                target_mean_icn=target_quality,
                smoothing_alpha=float(smooth_alpha),
                smoothing_steps=smoothing_steps,
                movable_mask=current_mask,
            )
            pts = active_points(buffer)
            elems = active_elements(buffer)
        elif remesh_triggered and topology_kind == "tetra":
            buffer, _ = adaptive_remesh_tet(
                pts,
                elems,
                max_nodes=max_nodes,
                max_elements=max_elements,
                max_iters=remesh_max_iters,
                target_volume=float(target_measure),
                target_mean_icn=target_quality,
                smoothing_alpha=float(smooth_alpha),
                smoothing_steps=smoothing_steps,
                movable_mask=current_mask,
            )
            pts = active_tet_points(buffer)
            elems = active_tet_elements(buffer)

        if remesh_triggered:
            current_node_fields = transfer_node_fields_nearest(old_points, pts, current_node_fields)
            current_element_fields = transfer_element_fields_nearest(old_points, old_elements, pts, elems, current_element_fields)
            current_mask = _transfer_mask_nearest(old_points, pts, current_mask)
            transfer_history.append(
                Mode5TransferSummary(
                    cycle=cycle,
                    old_n_nodes=int(old_points.shape[0]),
                    new_n_nodes=int(pts.shape[0]),
                    old_n_elements=int(old_elements.shape[0]),
                    new_n_elements=int(elems.shape[0]),
                    transferred_node_fields=tuple(sorted(current_node_fields)),
                    transferred_element_fields=tuple(sorted(current_element_fields)),
                )
            )
        else:
            transfer_history.append(
                Mode5TransferSummary(
                    cycle=cycle,
                    old_n_nodes=int(old_points.shape[0]),
                    new_n_nodes=int(old_points.shape[0]),
                    old_n_elements=int(old_elements.shape[0]),
                    new_n_elements=int(old_elements.shape[0]),
                    transferred_node_fields=tuple(),
                    transferred_element_fields=tuple(),
                )
            )

        controller_history.append(
            Mode5ControllerDecision(
                cycle=cycle,
                topology_kind=topology_kind,
                min_quality=min_quality,
                max_measure=max_measure,
                surrogate_initial_objective=surrogate_initial,
                surrogate_final_objective=surrogate_final,
                remesh_triggered=remesh_triggered,
                reason=reason,
            )
        )
        phases.append(
            Mode5Phase(
                cycle=cycle,
                start_energy=start_energy,
                final_energy=final_energy,
                surrogate_initial_objective=surrogate_initial,
                surrogate_final_objective=surrogate_final,
                n_nodes=int(pts.shape[0]),
                n_elements=int(elems.shape[0]),
                remeshed=remesh_triggered,
            )
        )

    return Mode5OptimizationResult(
        points=pts,
        elements=elems,
        phases=phases,
        controller_history=controller_history,
        transfer_history=transfer_history,
        node_fields=current_node_fields,
        element_fields=current_element_fields,
        topology_kind=topology_kind,
        surrogate_variant=surrogate_variant,
        implementation_status="implemented-relaxed-dynamic",
    )


def summarize_mode5_result(result: Mode5OptimizationResult) -> dict[str, Any]:
    remesh_count = int(sum(1 for phase in result.phases if phase.remeshed))
    return {
        "topology_kind": result.topology_kind,
        "surrogate_variant": result.surrogate_variant,
        "implementation_status": result.implementation_status,
        "n_cycles": len(result.phases),
        "n_nodes": int(result.points.shape[0]),
        "n_elements": int(result.elements.shape[0]),
        "remesh_count": remesh_count,
        "final_energy": float(result.phases[-1].final_energy),
    }


def export_mode5_artifacts(
    output_dir: str | Path,
    result: Mode5OptimizationResult,
    *,
    prefix: str = "mode5",
    extra_element_blocks: tuple[GmshElementBlock, ...] | None = None,
    physical_names: dict[tuple[int, int], str] | None = None,
) -> dict[str, Path]:
    from topojax.visualization import build_mode5_visualization_payload

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize_mode5_result(result)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    phases_path = out_dir / f"{prefix}_phases.json"
    controller_path = out_dir / f"{prefix}_controller.json"
    transfer_path = out_dir / f"{prefix}_transfers.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    mesh_path = out_dir / f"{prefix}_mesh.msh"

    export_metrics_json(metrics_path, metrics)
    phases_path.write_text(json.dumps([phase._asdict() for phase in result.phases], indent=2), encoding="utf-8")
    controller_path.write_text(json.dumps([entry._asdict() for entry in result.controller_history], indent=2), encoding="utf-8")
    transfer_path.write_text(json.dumps([entry._asdict() for entry in result.transfer_history], indent=2), encoding="utf-8")
    viewer_payload = build_mode5_visualization_payload(
        points=result.points,
        elements=result.elements,
        metrics=metrics,
        metadata={
            "controller_history": [entry._asdict() for entry in result.controller_history],
            "transfer_history": [entry._asdict() for entry in result.transfer_history],
            "surrogate_variant": result.surrogate_variant,
        },
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
        "phases": phases_path,
        "controller": controller_path,
        "transfers": transfer_path,
        "viewer_payload": viewer_path,
        "mesh": mesh_path,
    }

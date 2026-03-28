"""Mode 1 helpers: fixed-topology AD optimization, diagnostics, IO, and benchmarks."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from topojax.ad._common import cached_build, coerce_runtime_points, fit_node_mask, mesh_topology_metrics, topology_cache_key
from topojax.ad.compiled import build_quality_value_and_grad
from topojax.io.exports import GmshElementBlock, export_binary_stl, export_gmsh_msh, export_metrics_json, export_snapshot_npz
from topojax.io.topo_snapshot import export_topo_snapshot
from topojax.mesh.diagnostics import element_diagnostic_fields
from topojax.mesh.topology import MeshTopology
from topojax.runtime import get_runtime_precision


class Mode1StepDiagnostics(NamedTuple):
    step: int
    energy: float
    grad_norm: float
    metrics: dict[str, float | int]


class Mode1OptimizationResult(NamedTuple):
    points: jnp.ndarray
    topology: MeshTopology
    energy_history: jnp.ndarray
    grad_norm_history: jnp.ndarray
    step_diagnostics: tuple[Mode1StepDiagnostics, ...]


class Mode1BenchmarkResult(NamedTuple):
    first_call_ms: float
    steady_state_ms_per_step: float
    final_energy: float
    final_grad_norm: float
    steps: int


class Mode1JaxDiagnostics(NamedTuple):
    runtime_precision: str
    point_dtype: str
    point_shape: tuple[int, ...]
    element_shape: tuple[int, ...]
    point_dim: int
    element_order: int
    value_and_grad_cache_size: int | None
    optimizer_cache_size: int | None


_MODE1_SCAN_CACHE: OrderedDict[tuple[object, ...], object] = OrderedDict()


def _build_mode1_scan(
    topology: MeshTopology,
    *,
    steps: int,
    diagnostic_indices: tuple[int, ...] = (),
):
    normalized_indices = tuple(int(index) for index in diagnostic_indices)
    key = ("mode1_scan", topology_cache_key(topology), int(steps), normalized_indices)

    def _build():
        value_and_grad = build_quality_value_and_grad(topology)

        if normalized_indices:
            diagnostic_steps = jnp.asarray(normalized_indices, dtype=jnp.int32)
            n_diagnostics = len(normalized_indices)

            @jax.jit
            def run(points: jnp.ndarray, step_size, mask: jnp.ndarray):
                pts0 = coerce_runtime_points(points)
                mask_local = jnp.asarray(mask, dtype=pts0.dtype)
                step_size_local = jnp.asarray(step_size, dtype=pts0.dtype)
                empty_snaps = jnp.zeros((n_diagnostics, pts0.shape[0], pts0.shape[1]), dtype=pts0.dtype)

                def body(carry, step_index: jnp.ndarray):
                    pts, snapshots, snapshot_slot = carry
                    value, grad = value_and_grad(pts)
                    masked_grad = grad * mask_local[:, None]
                    next_pts = pts - step_size_local * masked_grad
                    target_slot = jnp.minimum(snapshot_slot, n_diagnostics - 1)
                    target_step = diagnostic_steps[target_slot]
                    should_store = jnp.logical_and(snapshot_slot < n_diagnostics, step_index == target_step)

                    def _store(args):
                        local_snaps, local_slot = args
                        return local_snaps.at[local_slot].set(next_pts), local_slot + 1

                    snapshots, snapshot_slot = jax.lax.cond(
                        should_store,
                        _store,
                        lambda args: args,
                        (snapshots, snapshot_slot),
                    )
                    return (next_pts, snapshots, snapshot_slot), (value, jnp.linalg.norm(masked_grad))

                (final_points, snapshots, _), outputs = jax.lax.scan(
                    body,
                    (pts0, empty_snaps, jnp.asarray(0, dtype=jnp.int32)),
                    xs=jnp.arange(steps, dtype=jnp.int32),
                )
                return final_points, outputs, snapshots

        else:

            @jax.jit
            def run(points: jnp.ndarray, step_size, mask: jnp.ndarray):
                pts0 = coerce_runtime_points(points)
                mask_local = jnp.asarray(mask, dtype=pts0.dtype)
                step_size_local = jnp.asarray(step_size, dtype=pts0.dtype)

                def body(pts: jnp.ndarray, _):
                    value, grad = value_and_grad(pts)
                    masked_grad = grad * mask_local[:, None]
                    next_pts = pts - step_size_local * masked_grad
                    return next_pts, (value, jnp.linalg.norm(masked_grad))

                final_points, outputs = jax.lax.scan(body, pts0, xs=None, length=steps)
                return final_points, outputs

        return run

    return cached_build(_MODE1_SCAN_CACHE, key, _build)


def _diagnostic_step_indices(steps: int, diagnostics_every: int) -> tuple[int, ...]:
    if steps <= 0 or diagnostics_every <= 0:
        return ()
    indices = {0, steps - 1}
    indices.update(step - 1 for step in range(diagnostics_every, steps + 1, diagnostics_every))
    return tuple(sorted(index for index in indices if 0 <= index < steps))


def build_mode1_optimizer(
    topology: MeshTopology,
    *,
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
):
    """Build a pure-JAX fixed-topology optimizer for mode 1.

    The returned callable is jitted and executes the optimization loop through
    `jax.lax.scan`, so the hot path contains no Python step loop.
    """
    run = _build_mode1_scan(
        topology,
        steps=steps,
    )

    def wrapped(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pts0 = coerce_runtime_points(points)
        mask = fit_node_mask(movable_mask, int(pts0.shape[0]))
        if mask is None:
            mask = jnp.ones((pts0.shape[0],), dtype=bool)
        final_points, outputs = run(pts0, step_size, mask)
        energy_history, grad_norm_history = outputs
        return final_points, energy_history, grad_norm_history

    wrapped._compiled_run = run  # type: ignore[attr-defined]
    if hasattr(run, "_cache_size"):
        wrapped._cache_size = run._cache_size  # type: ignore[attr-defined]
    if hasattr(run, "clear_cache"):
        wrapped.clear_cache = run.clear_cache  # type: ignore[attr-defined]
    return wrapped


def _topology_metrics(points: jnp.ndarray, topology: MeshTopology) -> dict[str, float | int]:
    return mesh_topology_metrics(points, topology)


def _mode1_step_diagnostics_payload(result: Mode1OptimizationResult) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for entry in result.step_diagnostics:
        row: dict[str, float | int] = {
            "step": int(entry.step),
            "energy": float(entry.energy),
            "grad_norm": float(entry.grad_norm),
        }
        row.update({str(key): value for key, value in entry.metrics.items()})
        rows.append(row)
    return rows


def mode1_history_payload(result: Mode1OptimizationResult) -> dict[str, np.ndarray]:
    """Return the stable Mode 1 history arrays."""
    steps = np.arange(1, int(result.energy_history.shape[0]) + 1, dtype=np.int32)
    return {
        "step": steps,
        "energy_history": np.asarray(result.energy_history),
        "grad_norm_history": np.asarray(result.grad_norm_history),
    }


def mode1_metrics_payload(result: Mode1OptimizationResult) -> dict[str, float | int | bool | str]:
    """Return the stable scalar diagnostics payload for a Mode 1 result."""
    final_metrics = _topology_metrics(result.points, result.topology)
    initial_energy = float(result.energy_history[0])
    final_energy = float(result.energy_history[-1])
    initial_grad_norm = float(result.grad_norm_history[0])
    final_grad_norm = float(result.grad_norm_history[-1])
    energy_drop = initial_energy - final_energy
    grad_norm_drop = initial_grad_norm - final_grad_norm
    relative_energy_drop = energy_drop / max(abs(initial_energy), 1.0e-12)
    relative_grad_norm_drop = grad_norm_drop / max(abs(initial_grad_norm), 1.0e-12)
    if final_grad_norm <= 1.0e-6 or relative_grad_norm_drop >= 0.99:
        status = "converged"
    elif relative_energy_drop <= 1.0e-8:
        status = "stalled"
    else:
        status = "improving"
    return {
        **final_metrics,
        "schema_name": "topojax.mode1.metrics",
        "schema_version": "1.0",
        "final_energy": final_energy,
        "initial_energy": initial_energy,
        "final_grad_norm": final_grad_norm,
        "initial_grad_norm": initial_grad_norm,
        "energy_drop": energy_drop,
        "grad_norm_drop": grad_norm_drop,
        "relative_energy_drop": relative_energy_drop,
        "relative_grad_norm_drop": relative_grad_norm_drop,
        "n_steps": int(result.energy_history.shape[0]),
        "n_diagnostic_samples": int(len(result.step_diagnostics)),
        "runtime_precision": get_runtime_precision(),
        "point_dim": int(result.points.shape[1]),
        "element_order": int(result.topology.elements.shape[1]),
        "n_nodes": int(result.topology.n_nodes),
        "n_elements": int(result.topology.elements.shape[0]),
        "status": status,
        "converged": status == "converged",
        "stalled": status == "stalled",
    }


def _write_history_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def optimize_mode1_fixed_topology(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
    diagnostics_every: int = 10,
) -> Mode1OptimizationResult:
    """Run fixed-topology AD optimization with diagnostics collection."""
    pts0 = coerce_runtime_points(points)
    step_diagnostics: list[Mode1StepDiagnostics] = []

    if diagnostics_every > 0:
        diagnostic_steps = _diagnostic_step_indices(steps, diagnostics_every)
        history_optimizer = _build_mode1_scan(
            topology,
            steps=steps,
            diagnostic_indices=diagnostic_steps,
        )
        mask = fit_node_mask(movable_mask, int(pts0.shape[0]))
        if mask is None:
            mask = jnp.ones((pts0.shape[0],), dtype=bool)
        if diagnostic_steps:
            pts, outputs, diagnostic_snapshots = history_optimizer(pts0, step_size, mask)
        else:
            pts, outputs = history_optimizer(pts0, step_size, mask)
            diagnostic_snapshots = jnp.zeros((0, pts0.shape[0], pts0.shape[1]), dtype=pts0.dtype)
        energy_history, grad_norm_history = outputs
        for snap_index, step in enumerate(diagnostic_steps):
            snap = diagnostic_snapshots[snap_index]
            metrics = _topology_metrics(snap, topology)
            step_diagnostics.append(
                Mode1StepDiagnostics(
                    step=step,
                    energy=float(energy_history[step]),
                    grad_norm=float(grad_norm_history[step]),
                    metrics=metrics,
                )
            )
    else:
        optimizer = build_mode1_optimizer(topology, steps=steps, step_size=step_size, movable_mask=movable_mask)
        pts, energy_history, grad_norm_history = optimizer(pts0)

    return Mode1OptimizationResult(
        points=pts,
        topology=topology,
        energy_history=energy_history,
        grad_norm_history=grad_norm_history,
        step_diagnostics=tuple(step_diagnostics),
    )


def export_mode1_artifacts(
    output_dir: str | Path,
    result: Mode1OptimizationResult,
    *,
    prefix: str = "mode1",
    export_stl_surface: bool = False,
    extra_element_blocks: tuple[GmshElementBlock, ...] | None = None,
    physical_names: dict[tuple[int, int], str] | None = None,
) -> dict[str, Path]:
    """Export final mode-1 snapshot, metrics, and mesh artifacts."""
    from topojax.visualization import build_mode1_visualization_payload, export_mode1_visualization_payload

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = mode1_metrics_payload(result)
    history_payload = mode1_history_payload(result)
    step_rows = _mode1_step_diagnostics_payload(result)
    element_fields = {
        key: np.asarray(value)
        for key, value in element_diagnostic_fields(result.points, result.topology.elements).items()
    }
    visualization_payload = build_mode1_visualization_payload(
        result.points,
        result.topology,
        title=f"{prefix} final mesh",
        metrics=final_metrics,
    )
    snap_path = out_dir / f"{prefix}_final_snapshot.npz"
    topo_snap_path = out_dir / f"{prefix}_final_snapshot.topo.npz"
    json_path = out_dir / f"{prefix}_final_metrics.json"
    msh_path = out_dir / f"{prefix}_final_mesh.msh"
    hist_path = out_dir / f"{prefix}_history.npz"
    hist_json_path = out_dir / f"{prefix}_history.json"
    hist_csv_path = out_dir / f"{prefix}_history.csv"
    viewer_path = out_dir / f"{prefix}_viewer_payload.json"
    stl_path = out_dir / f"{prefix}_final_surface.stl"

    export_snapshot_npz(snap_path, result.points, result.topology.elements, metrics=final_metrics)
    export_topo_snapshot(
        topo_snap_path,
        result.points,
        result.topology,
        metrics=final_metrics,
        history=history_payload,
        step_diagnostics=step_rows,
        element_fields=element_fields,
        visualization_payload=visualization_payload,
        physical_names=physical_names,
        extra_element_blocks=extra_element_blocks,
        metadata={"artifact_kind": "mode1-final"},
    )
    export_metrics_json(json_path, final_metrics)
    export_gmsh_msh(
        msh_path,
        result.points,
        result.topology.elements,
        element_entity_tags=result.topology.element_entity_tags,
        extra_element_blocks=extra_element_blocks,
        physical_names=physical_names,
    )
    np.savez(
        hist_path,
        **history_payload,
    )
    hist_json_path.write_text(json.dumps(step_rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_history_csv(hist_csv_path, step_rows)
    export_mode1_visualization_payload(viewer_path, result.points, result.topology, title=f"{prefix} final mesh", metrics=final_metrics)
    artifacts = {
        "snapshot": snap_path,
        "topo_snapshot": topo_snap_path,
        "metrics": json_path,
        "mesh": msh_path,
        "history": hist_path,
        "history_json": hist_json_path,
        "history_csv": hist_csv_path,
        "viewer_payload": viewer_path,
    }
    if export_stl_surface:
        export_binary_stl(stl_path, result.points, result.topology.elements)
        artifacts["stl"] = stl_path
    return artifacts


def benchmark_mode1_fixed_topology(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    steps: int = 50,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
) -> Mode1BenchmarkResult:
    """Benchmark compile cost and steady-state cost for mode 1."""
    pts = coerce_runtime_points(points)
    optimizer = build_mode1_optimizer(topology, steps=steps, step_size=step_size, movable_mask=movable_mask)

    t0 = perf_counter()
    _, energy_history, grad_norm_history = optimizer(pts)
    _ = jax.block_until_ready((energy_history, grad_norm_history))
    t1 = perf_counter()

    t2 = perf_counter()
    _, energy_history_steady, grad_norm_history_steady = optimizer(pts)
    _ = jax.block_until_ready((energy_history_steady, grad_norm_history_steady))
    t3 = perf_counter()

    return Mode1BenchmarkResult(
        first_call_ms=(t1 - t0) * 1.0e3,
        steady_state_ms_per_step=((t3 - t2) / max(steps, 1)) * 1.0e3,
        final_energy=float(energy_history_steady[-1]),
        final_grad_norm=float(grad_norm_history_steady[-1]),
        steps=steps,
    )


def collect_mode1_jax_diagnostics(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
) -> Mode1JaxDiagnostics:
    """Compile the core Mode-1 paths and report cache and shape diagnostics."""
    pts = coerce_runtime_points(points)
    value_and_grad = build_quality_value_and_grad(topology)
    optimizer = build_mode1_optimizer(topology, steps=steps, step_size=step_size, movable_mask=movable_mask)

    value, grad = value_and_grad(pts)
    final_points, energy_history, grad_norm_history = optimizer(pts)
    _ = jax.block_until_ready((value, grad, final_points, energy_history, grad_norm_history))

    value_and_grad_cache_size = value_and_grad._cache_size() if hasattr(value_and_grad, "_cache_size") else None
    optimizer_cache_size = optimizer._cache_size() if hasattr(optimizer, "_cache_size") else None
    return Mode1JaxDiagnostics(
        runtime_precision=get_runtime_precision(),
        point_dtype=str(pts.dtype),
        point_shape=tuple(int(v) for v in pts.shape),
        element_shape=tuple(int(v) for v in topology.elements.shape),
        point_dim=int(pts.shape[1]),
        element_order=int(topology.elements.shape[1]),
        value_and_grad_cache_size=value_and_grad_cache_size,
        optimizer_cache_size=optimizer_cache_size,
    )


def summarize_mode1_result(result: Mode1OptimizationResult) -> dict[str, Any]:
    """Return a compact scalar summary of a mode-1 optimization run."""
    final_metrics = mode1_metrics_payload(result)
    return {
        "final_energy": float(result.energy_history[-1]),
        "final_grad_norm": float(result.grad_norm_history[-1]),
        "steps": int(result.energy_history.shape[0]),
        "status": final_metrics["status"],
        "converged": final_metrics["converged"],
        **_topology_metrics(result.points, result.topology),
    }

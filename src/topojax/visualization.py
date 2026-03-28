"""Visualization helpers for TopoJAX mesh movement modes.

Native Gmsh remains the default external viewer for Mode 1 mesh inspection.
PyVista and Viser support a shared payload/interface layer across Modes 1-5.
Modes 2-5 expose stable payloads and viewer entry points even where the full
algorithmic implementation is still incomplete.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from topojax.mesh.topology import MeshTopology, mesh_topology_from_points_and_elements


class TopoVisualizationState(NamedTuple):
    mode: int
    points: jnp.ndarray
    topology: MeshTopology
    title: str
    metadata: dict[str, Any]


def _points3(points: jnp.ndarray) -> np.ndarray:
    arr = np.asarray(points)
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
    return arr


def _tet_boundary_faces(elements: np.ndarray) -> np.ndarray:
    face_counts: dict[tuple[int, int, int], int] = {}
    face_owner: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in elements.tolist():
        faces = [
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ]
        for face in faces:
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_owner[key] = tuple(face)
    boundary = [face_owner[key] for key, count in face_counts.items() if count == 1]
    return np.asarray(boundary, dtype=np.int32)


def _pyvista_lines(elements: np.ndarray) -> np.ndarray:
    counts = np.full((elements.shape[0], 1), 2, dtype=np.int32)
    return np.hstack([counts, elements]).reshape(-1)


def _surface_faces(points: jnp.ndarray, topology: MeshTopology) -> np.ndarray | None:
    elems = np.asarray(topology.elements, dtype=np.int32)
    order = int(elems.shape[1])
    dim = int(np.asarray(points).shape[1])
    if order == 3:
        return elems
    if order == 4 and dim == 2:
        return np.vstack([elems[:, [0, 1, 2]], elems[:, [0, 2, 3]]])
    if order == 4 and dim == 3:
        return _tet_boundary_faces(elems)
    return None


def _topology_kind(points: jnp.ndarray, topology: MeshTopology) -> str:
    order = int(topology.elements.shape[1])
    dim = int(np.asarray(points).shape[1])
    if order == 2:
        return "line"
    if order == 3:
        return "triangle"
    if order == 4 and dim == 2:
        return "quad"
    if order == 4 and dim == 3:
        return "tetra"
    return "unknown"


def _coerce_topology(points: jnp.ndarray, topology: MeshTopology | None, elements: jnp.ndarray | None) -> MeshTopology:
    if topology is not None:
        return topology
    if elements is None:
        raise ValueError("Provide either topology or elements")
    return mesh_topology_from_points_and_elements(points, jnp.asarray(elements, dtype=jnp.int32))


def _state_from_result(
    result: object,
    *,
    mode: int,
    title: str,
    points: jnp.ndarray | None = None,
    topology: MeshTopology | None = None,
    elements: jnp.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> TopoVisualizationState:
    if hasattr(result, "points") and hasattr(result, "topology"):
        pts = jnp.asarray(getattr(result, "points"))
        topo = getattr(result, "topology")
        return TopoVisualizationState(mode=mode, points=pts, topology=topo, title=title, metadata=dict(metadata or {}))
    if hasattr(result, "points") and hasattr(result, "elements"):
        pts = jnp.asarray(getattr(result, "points"))
        topo = _coerce_topology(pts, None, jnp.asarray(getattr(result, "elements"), dtype=jnp.int32))
        meta = dict(metadata or {})
        if hasattr(result, "phases"):
            meta["phases"] = [phase._asdict() for phase in getattr(result, "phases")]
        return TopoVisualizationState(mode=mode, points=pts, topology=topo, title=title, metadata=meta)
    if points is None:
        raise TypeError("Provide a compatible result object or explicit points/topology/elements")
    return TopoVisualizationState(
        mode=mode,
        points=jnp.asarray(points),
        topology=_coerce_topology(jnp.asarray(points), topology, elements),
        title=title,
        metadata=dict(metadata or {}),
    )


def build_pyvista_dataset(points: jnp.ndarray, topology: MeshTopology):
    """Build a PyVista dataset for lines, triangles, quads, or tetrahedra."""
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyvista is not installed; install pyvista to enable this backend") from exc

    pts = _points3(points)
    elems = np.asarray(topology.elements, dtype=np.int32)
    order = int(elems.shape[1])
    dim = int(np.asarray(points).shape[1])

    if order == 2:
        return pv.PolyData(pts, lines=_pyvista_lines(elems))
    if order == 3:
        faces = np.hstack([np.full((elems.shape[0], 1), 3, dtype=np.int32), elems]).reshape(-1)
        return pv.PolyData(pts, faces)
    if order == 4 and dim == 2:
        faces = np.hstack([np.full((elems.shape[0], 1), 4, dtype=np.int32), elems]).reshape(-1)
        return pv.PolyData(pts, faces)
    if order == 4 and dim == 3:
        cell_sizes = np.full((elems.shape[0], 1), 4, dtype=np.int32)
        cells = np.hstack([cell_sizes, elems]).reshape(-1)
        celltypes = np.full((elems.shape[0],), pv.CellType.TETRA, dtype=np.uint8)
        return pv.UnstructuredGrid(cells, celltypes, pts)
    raise ValueError("Unsupported topology for PyVista visualization")


def plot_mode1_matplotlib(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    title: str = "Mode 1 Fixed-Topology Mesh",
):
    """Return a Matplotlib figure for a mesh state."""
    backend = import_module("topojax._visualization_backends.matplotlib_backend")
    return backend.plot_mode1_matplotlib(points, topology, title=title)


def plot_mode1_pyvista(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    title: str = "Mode 1 Fixed-Topology Mesh",
    show: bool = False,
):
    """Build a PyVista plotter for a mode-1 mesh state."""
    state = TopoVisualizationState(mode=1, points=points, topology=topology, title=title, metadata={})
    return plot_topo_pyvista(state, show=show)


def plot_topo_pyvista(
    state: TopoVisualizationState,
    *,
    show: bool = False,
    show_nodes: bool = True,
    show_edges: bool = True,
    background_color: str = "white",
):
    """Build a PyVista plotter for a Topo mode state."""
    backend = import_module("topojax._visualization_backends.pyvista_backend")
    return backend.plot_topo_pyvista(
        state,
        show=show,
        show_nodes=show_nodes,
        show_edges=show_edges,
        background_color=background_color,
    )


def build_topo_visualization_payload(
    state: TopoVisualizationState,
    *,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable viewer-neutral payload for Topo visualization artifacts."""
    pts = np.asarray(state.points)
    topology = state.topology
    return {
        "schema_name": "topojax.visualization.payload",
        "schema_version": "1.0",
        "title": state.title,
        "mode": int(state.mode),
        "points": pts.tolist(),
        "elements": np.asarray(topology.elements, dtype=np.int32).tolist(),
        "edges": np.asarray(topology.edges, dtype=np.int32).tolist(),
        "element_order": int(topology.elements.shape[1]),
        "point_dim": int(pts.shape[1]),
        "n_nodes": int(topology.n_nodes),
        "n_elements": int(topology.elements.shape[0]),
        "topology_kind": _topology_kind(state.points, topology),
        "bounds": {"min": pts.min(axis=0).tolist(), "max": pts.max(axis=0).tolist()},
        "metadata": state.metadata,
        "metrics": {
            str(key): (value.item() if isinstance(value, np.generic) else value)
            for key, value in (metrics or {}).items()
        },
    }


def build_mode1_visualization_payload(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    title: str = "Mode 1 Fixed-Topology Mesh",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable payload for Mode 1 visualization artifacts."""
    state = TopoVisualizationState(mode=1, points=points, topology=topology, title=title, metadata={})
    payload = build_topo_visualization_payload(state, metrics=metrics)
    payload["backend"] = "viewer-neutral"
    return payload


def build_mode2_visualization_payload(
    result: object,
    *,
    title: str = "Mode 2 Remesh-Restart Result",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable payload contract for Mode 2 visualization."""
    state = _state_from_result(result, mode=2, title=title)
    payload = build_topo_visualization_payload(state, metrics=metrics)
    payload["implementation_status"] = "implemented"
    return payload


def build_mode3_visualization_payload(
    *,
    points: jnp.ndarray,
    topology: MeshTopology | None = None,
    elements: jnp.ndarray | None = None,
    metrics: dict[str, Any] | None = None,
    title: str = "Mode 3 Soft-Connectivity State",
    candidate_logits: jnp.ndarray | None = None,
    implementation_status: str = "stubbed-interface",
) -> dict[str, Any]:
    """Build the stable payload contract for Mode 3 visualization.

    This defines the payload and interface now even though the full viewer-specific
    overlays for soft connectivity are still evolving.
    """
    state = _state_from_result(
        object(),
        mode=3,
        title=title,
        points=points,
        topology=topology,
        elements=elements,
        metadata={"candidate_logits": None if candidate_logits is None else np.asarray(candidate_logits).tolist()},
    )
    payload = build_topo_visualization_payload(state, metrics=metrics)
    payload["implementation_status"] = implementation_status
    return payload


def build_mode4_visualization_payload(
    *,
    points: jnp.ndarray,
    topology: MeshTopology | None = None,
    elements: jnp.ndarray | None = None,
    metrics: dict[str, Any] | None = None,
    title: str = "Mode 4 Straight-Through State",
    candidate_logits: jnp.ndarray | None = None,
    implementation_status: str = "stubbed-interface",
) -> dict[str, Any]:
    """Build the stable payload contract for Mode 4 visualization."""
    state = _state_from_result(
        object(),
        mode=4,
        title=title,
        points=points,
        topology=topology,
        elements=elements,
        metadata={"candidate_logits": None if candidate_logits is None else np.asarray(candidate_logits).tolist()},
    )
    payload = build_topo_visualization_payload(state, metrics=metrics)
    payload["implementation_status"] = implementation_status
    return payload


def build_mode5_visualization_payload(
    *,
    points: jnp.ndarray,
    topology: MeshTopology | None = None,
    elements: jnp.ndarray | None = None,
    metrics: dict[str, Any] | None = None,
    title: str = "Mode 5 Fully-Dynamic State",
    metadata: dict[str, Any] | None = None,
    implementation_status: str = "implemented",
) -> dict[str, Any]:
    """Build the stable payload contract for Mode 5 visualization."""
    state = _state_from_result(
        object(),
        mode=5,
        title=title,
        points=points,
        topology=topology,
        elements=elements,
        metadata=metadata,
    )
    payload = build_topo_visualization_payload(state, metrics=metrics)
    payload["implementation_status"] = implementation_status
    return payload


def export_mode1_visualization_payload(
    path: str | Path,
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    title: str = "Mode 1 Fixed-Topology Mesh",
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write the stable Mode 1 visualization payload to JSON."""
    payload = build_mode1_visualization_payload(points, topology, title=title, metrics=metrics)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def plot_mode1_gmsh(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    mesh_path: str | Path | None = None,
    gmsh_executable: str = "gmsh",
    wait: bool = False,
    extra_args: list[str] | None = None,
):
    """Export a temporary `.msh` and open it in the native Gmsh GUI."""
    backend = import_module("topojax._visualization_backends.gmsh_backend")
    return backend.plot_mode1_gmsh(
        points,
        topology,
        mesh_path=mesh_path,
        gmsh_executable=gmsh_executable,
        wait=wait,
        extra_args=extra_args,
    )


def plot_topo_viser(
    state: TopoVisualizationState,
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    block: bool = False,
):
    """Launch a minimal Viser scene for a Topo mode state."""
    backend = import_module("topojax._visualization_backends.viser_backend")
    return backend.plot_topo_viser(state, host=host, port=port, block=block)


def visualize_mode1(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    backend: str = "gmsh",
    title: str = "Mode 1 Fixed-Topology Mesh",
    show: bool = False,
    mesh_path: str | Path | None = None,
    gmsh_executable: str = "gmsh",
    gmsh_extra_args: list[str] | None = None,
    wait: bool = False,
):
    """Render a Mode 1 mesh through the selected backend."""
    state = TopoVisualizationState(mode=1, points=points, topology=topology, title=title, metadata={})
    if backend == "matplotlib":
        return plot_mode1_matplotlib(points, topology, title=title)
    if backend == "pyvista":
        return plot_topo_pyvista(state, show=show)
    if backend == "gmsh":
        return plot_mode1_gmsh(
            points,
            topology,
            mesh_path=mesh_path,
            gmsh_executable=gmsh_executable,
            wait=wait,
            extra_args=gmsh_extra_args,
        )
    if backend == "viser":
        return plot_topo_viser(state)
    raise ValueError(f"Unsupported visualization backend: {backend}")


def visualize_mode1_result(
    result,
    *,
    backend: str = "gmsh",
    title: str = "Mode 1 Fixed-Topology Result",
    show: bool = False,
    mesh_path: str | Path | None = None,
    gmsh_executable: str = "gmsh",
    gmsh_extra_args: list[str] | None = None,
    wait: bool = False,
):
    """Render a Mode 1 result through the selected backend."""
    return visualize_mode1(
        result.points,
        result.topology,
        backend=backend,
        title=title,
        show=show,
        mesh_path=mesh_path,
        gmsh_executable=gmsh_executable,
        gmsh_extra_args=gmsh_extra_args,
        wait=wait,
    )


def visualize_mode2_result(
    result,
    *,
    backend: str = "gmsh",
    title: str = "Mode 2 Remesh-Restart Result",
    show: bool = False,
    mesh_path: str | Path | None = None,
    gmsh_executable: str = "gmsh",
    gmsh_extra_args: list[str] | None = None,
    wait: bool = False,
):
    """Render a Mode 2 result through the selected backend."""
    state = _state_from_result(result, mode=2, title=title)
    return visualize_topo_state(
        state,
        backend=backend,
        show=show,
        mesh_path=mesh_path,
        gmsh_executable=gmsh_executable,
        gmsh_extra_args=gmsh_extra_args,
        wait=wait,
    )


def visualize_mode3_state(
    *,
    points: jnp.ndarray,
    topology: MeshTopology | None = None,
    elements: jnp.ndarray | None = None,
    backend: str = "gmsh",
    title: str = "Mode 3 Soft-Connectivity State",
    show: bool = False,
):
    """Render a Mode 3 state through the selected backend."""
    state = _state_from_result(object(), mode=3, title=title, points=points, topology=topology, elements=elements)
    return visualize_topo_state(state, backend=backend, show=show)


def visualize_mode4_state(
    *,
    points: jnp.ndarray,
    topology: MeshTopology | None = None,
    elements: jnp.ndarray | None = None,
    backend: str = "gmsh",
    title: str = "Mode 4 Straight-Through State",
    show: bool = False,
):
    """Render a Mode 4 state through the selected backend."""
    state = _state_from_result(object(), mode=4, title=title, points=points, topology=topology, elements=elements)
    return visualize_topo_state(state, backend=backend, show=show)


def visualize_mode5_state(
    *,
    points: jnp.ndarray,
    topology: MeshTopology | None = None,
    elements: jnp.ndarray | None = None,
    backend: str = "gmsh",
    title: str = "Mode 5 Fully-Dynamic State",
    show: bool = False,
):
    """Render a Mode 5 state through the selected backend."""
    state = _state_from_result(object(), mode=5, title=title, points=points, topology=topology, elements=elements)
    return visualize_topo_state(state, backend=backend, show=show)


def visualize_topo_state(
    state: TopoVisualizationState,
    *,
    backend: str = "gmsh",
    show: bool = False,
    mesh_path: str | Path | None = None,
    gmsh_executable: str = "gmsh",
    gmsh_extra_args: list[str] | None = None,
    wait: bool = False,
):
    """Render a generic Topo visualization state through the selected backend."""
    if backend == "matplotlib":
        return plot_mode1_matplotlib(state.points, state.topology, title=state.title)
    if backend == "pyvista":
        return plot_topo_pyvista(state, show=show)
    if backend == "gmsh":
        return plot_mode1_gmsh(
            state.points,
            state.topology,
            mesh_path=mesh_path,
            gmsh_executable=gmsh_executable,
            wait=wait,
            extra_args=gmsh_extra_args,
        )
    if backend == "viser":
        return plot_topo_viser(state)
    raise ValueError(f"Unsupported visualization backend: {backend}")

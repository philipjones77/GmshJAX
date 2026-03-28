"""NumPy-native mesh runtime with RF77 bridge, IO, diagnostics, and visualization hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from topojax.ad.modes import MeshMovementMode
from topojax.io.exports import GmshElementBlock
from topojax.mesh.domains import DomainMeshMetadata
from topojax.mesh.topology import MeshTopology, mesh_topology_from_points_and_elements
from topojax.rf77 import (
    RandomFields77ModeBridge,
    build_mode1_randomfields77_bridge,
    build_mode2_randomfields77_bridge,
    build_mode3_randomfields77_bridge,
    build_mode4_randomfields77_bridge,
    build_mode5_randomfields77_bridge,
)
from topojax.visualization import (
    build_mode1_visualization_payload as topo_build_mode1_visualization_payload,
    build_mode2_visualization_payload as topo_build_mode2_visualization_payload,
    build_mode3_visualization_payload as topo_build_mode3_visualization_payload,
    build_mode4_visualization_payload as topo_build_mode4_visualization_payload,
    build_mode5_visualization_payload as topo_build_mode5_visualization_payload,
    visualize_mode1,
    visualize_mode2_result,
    visualize_mode3_state,
    visualize_mode4_state,
    visualize_mode5_state,
)

from .diagnostics import to_jsonable
from .io import atomic_write_json, atomic_write_npz
from .movement import MeshMovementTransform, apply_mesh_movement_numpy


__all__ = [
    "NumpyMeshDiagnostics",
    "NumpyMeshRuntime",
    "build_mode_bridge",
    "build_visualization_payload",
    "create_mode1_runtime",
    "create_mode2_runtime",
    "create_mode3_runtime",
    "create_mode4_runtime",
    "create_mode5_runtime",
    "create_runtime",
    "export_mode1_artifacts",
    "export_mode2_artifacts",
    "export_mode3_artifacts",
    "export_mode4_artifacts",
    "export_mode5_artifacts",
    "export_mode_artifacts",
    "load_runtime",
    "move_runtime",
    "save_runtime",
    "visualize_runtime",
]


_MODE_TITLES = {
    MeshMovementMode.FIXED_TOPOLOGY: "NumPy Mode 1 Mesh",
    MeshMovementMode.REMESH_RESTART: "NumPy Mode 2 Mesh",
    MeshMovementMode.SOFT_CONNECTIVITY: "NumPy Mode 3 Mesh",
    MeshMovementMode.STRAIGHT_THROUGH: "NumPy Mode 4 Mesh",
    MeshMovementMode.FULLY_DYNAMIC: "NumPy Mode 5 Mesh",
}


class NumpyMeshDiagnostics(NamedTuple):
    mode: str
    element_kind: str
    point_dtype: str
    point_dim: int
    n_nodes: int
    n_elements: int
    n_edges: int
    bounds_min: tuple[float, ...]
    bounds_max: tuple[float, ...]
    n_boundary_blocks: int
    metadata_keys: tuple[str, ...]
    mode_payload_keys: tuple[str, ...]


def _as_numpy_points(points: Any) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n_points, 2) or (n_points, 3)")
    return arr


def _as_numpy_elements(elements: Any) -> np.ndarray:
    arr = np.asarray(elements, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3, 4):
        raise ValueError("elements must have shape (n_elements, 2), (n_elements, 3), or (n_elements, 4)")
    return arr


def _infer_element_kind(points: np.ndarray, elements: np.ndarray, element_kind: str | None) -> str:
    if element_kind is not None:
        return str(element_kind)
    if int(elements.shape[1]) == 2:
        return "line"
    if int(elements.shape[1]) == 3:
        return "triangle"
    if int(points.shape[1]) == 2:
        return "quad"
    return "tetra"


def _domain_metadata(
    boundary_element_blocks: tuple[GmshElementBlock, ...],
    physical_names: dict[tuple[int, int], str],
) -> DomainMeshMetadata | None:
    if not boundary_element_blocks and not physical_names:
        return None
    return DomainMeshMetadata(boundary_element_blocks=boundary_element_blocks, physical_names=physical_names)


def _serialize_boundary_blocks(blocks: tuple[GmshElementBlock, ...]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    manifest: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, block in enumerate(blocks):
        elem_key = f"boundary_{index}_elements"
        arrays[elem_key] = np.asarray(block.elements, dtype=np.int32)
        entry: dict[str, Any] = {"element_kind": block.element_kind, "elements_key": elem_key}
        if block.physical_tags is not None:
            phys_key = f"boundary_{index}_physical_tags"
            arrays[phys_key] = np.asarray(block.physical_tags, dtype=np.int32)
            entry["physical_tags_key"] = phys_key
        if block.geometrical_tags is not None:
            geom_key = f"boundary_{index}_geometrical_tags"
            arrays[geom_key] = np.asarray(block.geometrical_tags, dtype=np.int32)
            entry["geometrical_tags_key"] = geom_key
        manifest.append(entry)
    return manifest, arrays


def _deserialize_boundary_blocks(manifest: list[dict[str, Any]], archive: Mapping[str, Any]) -> tuple[GmshElementBlock, ...]:
    blocks: list[GmshElementBlock] = []
    for entry in manifest:
        blocks.append(
            GmshElementBlock(
                elements=np.asarray(archive[entry["elements_key"]], dtype=np.int32),
                element_kind=str(entry["element_kind"]),
                physical_tags=None
                if "physical_tags_key" not in entry
                else np.asarray(archive[entry["physical_tags_key"]], dtype=np.int32),
                geometrical_tags=None
                if "geometrical_tags_key" not in entry
                else np.asarray(archive[entry["geometrical_tags_key"]], dtype=np.int32),
            )
        )
    return tuple(blocks)


@dataclass
class NumpyMeshRuntime:
    points: np.ndarray
    elements: np.ndarray
    mode: MeshMovementMode = MeshMovementMode.FIXED_TOPOLOGY
    metadata: dict[str, Any] = field(default_factory=dict)
    mode_payload: dict[str, Any] = field(default_factory=dict)
    element_kind: str | None = None
    element_entity_tags: np.ndarray | None = None
    boundary_element_blocks: tuple[GmshElementBlock, ...] = ()
    physical_names: dict[tuple[int, int], str] = field(default_factory=dict)
    geometry_fn: Any | None = None
    name: str | None = None
    _topology_cache: MeshTopology | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.points = _as_numpy_points(self.points)
        self.elements = _as_numpy_elements(self.elements)
        self.mode = MeshMovementMode(self.mode)
        self.element_kind = _infer_element_kind(self.points, self.elements, self.element_kind)
        self.metadata = dict(self.metadata)
        self.mode_payload = dict(self.mode_payload)
        self.physical_names = {tuple(map(int, key)): str(value) for key, value in self.physical_names.items()}
        if self.element_entity_tags is not None:
            tags = np.asarray(self.element_entity_tags, dtype=np.int32)
            if tags.shape != (int(self.elements.shape[0]),):
                raise ValueError("element_entity_tags must have shape (n_elements,)")
            self.element_entity_tags = tags

    @property
    def topology(self) -> MeshTopology:
        if self._topology_cache is None:
            tags = None if self.element_entity_tags is None else jnp.asarray(self.element_entity_tags, dtype=jnp.int32)
            self._topology_cache = mesh_topology_from_points_and_elements(
                jnp.asarray(self.points),
                jnp.asarray(self.elements, dtype=jnp.int32),
                element_entity_tags=tags,
            )
        return self._topology_cache

    @property
    def domain_metadata(self) -> DomainMeshMetadata | None:
        return _domain_metadata(self.boundary_element_blocks, self.physical_names)

    def diagnostics(self) -> NumpyMeshDiagnostics:
        bounds_min = tuple(float(v) for v in np.min(self.points, axis=0))
        bounds_max = tuple(float(v) for v in np.max(self.points, axis=0))
        topology = self.topology
        return NumpyMeshDiagnostics(
            mode=self.mode.value,
            element_kind=str(self.element_kind),
            point_dtype=str(self.points.dtype),
            point_dim=int(self.points.shape[1]),
            n_nodes=int(self.points.shape[0]),
            n_elements=int(self.elements.shape[0]),
            n_edges=int(topology.edges.shape[0]),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            n_boundary_blocks=len(self.boundary_element_blocks),
            metadata_keys=tuple(sorted(self.metadata)),
            mode_payload_keys=tuple(sorted(self.mode_payload)),
        )

    def build_mode_bridge(self, **kwargs: Any) -> RandomFields77ModeBridge:
        return build_mode_bridge(self, **kwargs)

    def to_randomfields77_mesh_payload(self, **kwargs: Any) -> dict[str, Any]:
        return self.build_mode_bridge(**kwargs).to_randomfields77_mesh_payload()

    def build_visualization_payload(self, **kwargs: Any) -> dict[str, Any]:
        return build_visualization_payload(self, **kwargs)

    def visualize(self, **kwargs: Any):
        return visualize_runtime(self, **kwargs)

    def save(self, path: str | Path) -> Path:
        return save_runtime(path, self)

    def export_artifacts(self, output_dir: str | Path, *, prefix: str = "numpy_mesh") -> dict[str, Path]:
        return export_mode_artifacts(output_dir, self, prefix=prefix)

    def apply_transform(self, transform: MeshMovementTransform) -> "NumpyMeshRuntime":
        return move_runtime(self, transform)


def create_runtime(
    points: Any,
    elements: Any,
    *,
    mode: MeshMovementMode | str = MeshMovementMode.FIXED_TOPOLOGY,
    metadata: Mapping[str, Any] | None = None,
    mode_payload: Mapping[str, Any] | None = None,
    element_kind: str | None = None,
    element_entity_tags: Any | None = None,
    boundary_element_blocks: tuple[GmshElementBlock, ...] = (),
    physical_names: dict[tuple[int, int], str] | None = None,
    geometry_fn=None,
    name: str | None = None,
) -> NumpyMeshRuntime:
    return NumpyMeshRuntime(
        points=np.asarray(points),
        elements=np.asarray(elements, dtype=np.int32),
        mode=MeshMovementMode(mode),
        metadata={} if metadata is None else dict(metadata),
        mode_payload={} if mode_payload is None else dict(mode_payload),
        element_kind=element_kind,
        element_entity_tags=None if element_entity_tags is None else np.asarray(element_entity_tags, dtype=np.int32),
        boundary_element_blocks=tuple(boundary_element_blocks),
        physical_names={} if physical_names is None else dict(physical_names),
        geometry_fn=geometry_fn,
        name=name,
    )


def create_mode1_runtime(points: Any, elements: Any, **kwargs: Any) -> NumpyMeshRuntime:
    return create_runtime(points, elements, mode=MeshMovementMode.FIXED_TOPOLOGY, **kwargs)


def create_mode2_runtime(points: Any, elements: Any, *, restart_phases: list[dict[str, Any]] | None = None, **kwargs: Any) -> NumpyMeshRuntime:
    payload = dict(kwargs.pop("mode_payload", {}))
    if restart_phases is not None:
        payload["restart_phases"] = restart_phases
    return create_runtime(points, elements, mode=MeshMovementMode.REMESH_RESTART, mode_payload=payload, **kwargs)


def create_mode3_runtime(
    points: Any,
    elements: Any,
    *,
    candidate_graph: dict[str, Any] | None = None,
    soft_weights: Any | None = None,
    candidate_logits: Any | None = None,
    **kwargs: Any,
) -> NumpyMeshRuntime:
    payload = dict(kwargs.pop("mode_payload", {}))
    if candidate_graph is not None:
        payload["candidate_graph"] = candidate_graph
    if soft_weights is not None:
        payload["soft_weights"] = np.asarray(soft_weights).tolist()
    if candidate_logits is not None:
        payload["candidate_logits"] = np.asarray(candidate_logits).tolist()
    payload.setdefault("implementation_status", "implemented-numpy")
    return create_runtime(points, elements, mode=MeshMovementMode.SOFT_CONNECTIVITY, mode_payload=payload, **kwargs)


def create_mode4_runtime(
    points: Any,
    elements: Any,
    *,
    candidate_graph: dict[str, Any] | None = None,
    forward_state: dict[str, Any] | None = None,
    backward_surrogate: dict[str, Any] | None = None,
    candidate_logits: Any | None = None,
    **kwargs: Any,
) -> NumpyMeshRuntime:
    payload = dict(kwargs.pop("mode_payload", {}))
    if candidate_graph is not None:
        payload["candidate_graph"] = candidate_graph
    if forward_state is not None:
        payload["forward_state"] = forward_state
    if backward_surrogate is not None:
        payload["backward_surrogate"] = backward_surrogate
    if candidate_logits is not None:
        payload["candidate_logits"] = np.asarray(candidate_logits).tolist()
    payload.setdefault("implementation_status", "implemented-numpy")
    return create_runtime(points, elements, mode=MeshMovementMode.STRAIGHT_THROUGH, mode_payload=payload, **kwargs)


def create_mode5_runtime(
    points: Any,
    elements: Any,
    *,
    controller_history: list[dict[str, Any]] | None = None,
    transfer_history: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> NumpyMeshRuntime:
    payload = dict(kwargs.pop("mode_payload", {}))
    if controller_history is not None:
        payload["controller_history"] = controller_history
    if transfer_history is not None:
        payload["transfer_history"] = transfer_history
    payload.setdefault("implementation_status", "implemented-numpy")
    return create_runtime(points, elements, mode=MeshMovementMode.FULLY_DYNAMIC, mode_payload=payload, **kwargs)


def build_mode_bridge(runtime: NumpyMeshRuntime, *, mode: MeshMovementMode | str | None = None, **kwargs: Any) -> RandomFields77ModeBridge:
    normalized = runtime.mode if mode is None else MeshMovementMode(mode)
    topology = runtime.topology
    metadata = runtime.domain_metadata
    builder_options = {"backend": "numpy", "storage": "numpy-array-runtime", **runtime.metadata, **kwargs.pop("builder_options", {})}
    geometry_fn = runtime.geometry_fn
    if geometry_fn is not None:
        geometry_fn = lambda params=None, _geometry_fn=geometry_fn: jnp.asarray(_geometry_fn(params))

    if normalized == MeshMovementMode.FIXED_TOPOLOGY:
        source = type("NumpyMode1Source", (), {"points": jnp.asarray(runtime.points), "topology": topology, "metadata": metadata})()
        return build_mode1_randomfields77_bridge(source, geometry_fn=geometry_fn, builder_options=builder_options, **kwargs)
    if normalized == MeshMovementMode.REMESH_RESTART:
        source = type("NumpyMode2Source", (), {"points": jnp.asarray(runtime.points), "topology": topology, "metadata": metadata})()
        return build_mode2_randomfields77_bridge(
            source,
            geometry_fn=geometry_fn,
            builder_options={**builder_options, "restart_phases": runtime.mode_payload.get("restart_phases")},
            **kwargs,
        )
    if normalized == MeshMovementMode.SOFT_CONNECTIVITY:
        return build_mode3_randomfields77_bridge(
            jnp.asarray(runtime.points),
            topology,
            metadata=metadata,
            geometry_fn=geometry_fn,
            candidate_graph=runtime.mode_payload.get("candidate_graph"),
            soft_weights=runtime.mode_payload.get("soft_weights"),
            builder_options=builder_options,
            **kwargs,
        )
    if normalized == MeshMovementMode.STRAIGHT_THROUGH:
        return build_mode4_randomfields77_bridge(
            jnp.asarray(runtime.points),
            topology,
            metadata=metadata,
            geometry_fn=geometry_fn,
            candidate_graph=runtime.mode_payload.get("candidate_graph"),
            forward_state=runtime.mode_payload.get("forward_state"),
            backward_surrogate=runtime.mode_payload.get("backward_surrogate"),
            builder_options=builder_options,
            **kwargs,
        )
    if normalized == MeshMovementMode.FULLY_DYNAMIC:
        return build_mode5_randomfields77_bridge(
            jnp.asarray(runtime.points),
            topology,
            metadata=metadata,
            geometry_fn=geometry_fn,
            builder_options={
                **builder_options,
                "controller_history": runtime.mode_payload.get("controller_history"),
                "transfer_history": runtime.mode_payload.get("transfer_history"),
                "implementation_status": runtime.mode_payload.get("implementation_status", "implemented-numpy"),
            },
            **kwargs,
        )
    raise KeyError(f"Unsupported mode: {mode}")


def _result_like(runtime: NumpyMeshRuntime):
    return type("NumpyResultLike", (), {"points": jnp.asarray(runtime.points), "elements": jnp.asarray(runtime.elements, dtype=jnp.int32)})()


def build_visualization_payload(
    runtime: NumpyMeshRuntime,
    *,
    metrics: dict[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    active_title = title or _MODE_TITLES[runtime.mode]
    topo = runtime.topology
    if runtime.mode == MeshMovementMode.FIXED_TOPOLOGY:
        return topo_build_mode1_visualization_payload(jnp.asarray(runtime.points), topo, title=active_title, metrics=metrics)
    if runtime.mode == MeshMovementMode.REMESH_RESTART:
        return topo_build_mode2_visualization_payload(_result_like(runtime), title=active_title, metrics=metrics)
    if runtime.mode == MeshMovementMode.SOFT_CONNECTIVITY:
        return topo_build_mode3_visualization_payload(
            points=jnp.asarray(runtime.points),
            topology=topo,
            title=active_title,
            metrics=metrics,
            candidate_logits=None
            if "candidate_logits" not in runtime.mode_payload
            else jnp.asarray(runtime.mode_payload["candidate_logits"]),
            implementation_status=str(runtime.mode_payload.get("implementation_status", "implemented-numpy")),
        )
    if runtime.mode == MeshMovementMode.STRAIGHT_THROUGH:
        return topo_build_mode4_visualization_payload(
            points=jnp.asarray(runtime.points),
            topology=topo,
            title=active_title,
            metrics=metrics,
            candidate_logits=None
            if "candidate_logits" not in runtime.mode_payload
            else jnp.asarray(runtime.mode_payload["candidate_logits"]),
            implementation_status=str(runtime.mode_payload.get("implementation_status", "implemented-numpy")),
        )
    return topo_build_mode5_visualization_payload(
        points=jnp.asarray(runtime.points),
        topology=topo,
        title=active_title,
        metrics=metrics,
        metadata=to_jsonable(runtime.mode_payload),
        implementation_status=str(runtime.mode_payload.get("implementation_status", "implemented-numpy")),
    )


def visualize_runtime(runtime: NumpyMeshRuntime, *, backend: str = "gmsh", title: str | None = None, show: bool = False, **kwargs: Any):
    active_title = title or _MODE_TITLES[runtime.mode]
    topo = runtime.topology
    if runtime.mode == MeshMovementMode.FIXED_TOPOLOGY:
        return visualize_mode1(jnp.asarray(runtime.points), topo, backend=backend, title=active_title, show=show, **kwargs)
    if runtime.mode == MeshMovementMode.REMESH_RESTART:
        return visualize_mode2_result(_result_like(runtime), backend=backend, title=active_title, show=show, **kwargs)
    if runtime.mode == MeshMovementMode.SOFT_CONNECTIVITY:
        return visualize_mode3_state(points=jnp.asarray(runtime.points), topology=topo, backend=backend, title=active_title, show=show)
    if runtime.mode == MeshMovementMode.STRAIGHT_THROUGH:
        return visualize_mode4_state(points=jnp.asarray(runtime.points), topology=topo, backend=backend, title=active_title, show=show)
    return visualize_mode5_state(points=jnp.asarray(runtime.points), topology=topo, backend=backend, title=active_title, show=show)


def save_runtime(path: str | Path, runtime: NumpyMeshRuntime) -> Path:
    boundary_manifest, boundary_arrays = _serialize_boundary_blocks(runtime.boundary_element_blocks)
    physical_names = [
        {"dim": int(dim), "tag": int(tag), "name": str(name)}
        for (dim, tag), name in sorted(runtime.physical_names.items())
    ]
    arrays: dict[str, Any] = {
        "points": runtime.points,
        "elements": runtime.elements,
        "metadata_json": np.asarray(json.dumps(to_jsonable(runtime.metadata)), dtype=np.str_),
        "mode_payload_json": np.asarray(json.dumps(to_jsonable(runtime.mode_payload)), dtype=np.str_),
        "mode": np.asarray(runtime.mode.value, dtype=np.str_),
        "element_kind": np.asarray(str(runtime.element_kind), dtype=np.str_),
        "physical_names_json": np.asarray(json.dumps(physical_names), dtype=np.str_),
        "boundary_manifest_json": np.asarray(json.dumps(boundary_manifest), dtype=np.str_),
        "name": np.asarray("" if runtime.name is None else runtime.name, dtype=np.str_),
    }
    if runtime.element_entity_tags is not None:
        arrays["element_entity_tags"] = runtime.element_entity_tags
    arrays.update(boundary_arrays)
    atomic_write_npz(path, **arrays)
    return Path(path)


def load_runtime(path: str | Path) -> NumpyMeshRuntime:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    mode_payload = json.loads(str(archive["mode_payload_json"]))
    physical_names_payload = json.loads(str(archive["physical_names_json"]))
    physical_names = {(int(entry["dim"]), int(entry["tag"])): str(entry["name"]) for entry in physical_names_payload}
    boundary_manifest = json.loads(str(archive["boundary_manifest_json"]))
    boundary_blocks = _deserialize_boundary_blocks(boundary_manifest, archive)
    return create_runtime(
        np.asarray(archive["points"]),
        np.asarray(archive["elements"], dtype=np.int32),
        mode=str(archive["mode"]),
        metadata=metadata,
        mode_payload=mode_payload,
        element_kind=str(archive["element_kind"]),
        element_entity_tags=None if "element_entity_tags" not in archive else np.asarray(archive["element_entity_tags"], dtype=np.int32),
        boundary_element_blocks=boundary_blocks,
        physical_names=physical_names,
        name=None if not str(archive["name"]) else str(archive["name"]),
    )


def move_runtime(runtime: NumpyMeshRuntime, transform: MeshMovementTransform) -> NumpyMeshRuntime:
    moved_points = apply_mesh_movement_numpy(runtime.points, transform)
    return create_runtime(
        moved_points,
        runtime.elements,
        mode=runtime.mode,
        metadata=runtime.metadata,
        mode_payload=runtime.mode_payload,
        element_kind=runtime.element_kind,
        element_entity_tags=runtime.element_entity_tags,
        boundary_element_blocks=runtime.boundary_element_blocks,
        physical_names=runtime.physical_names,
        geometry_fn=runtime.geometry_fn,
        name=runtime.name,
    )


def export_mode_artifacts(output_dir: str | Path, runtime: NumpyMeshRuntime, *, prefix: str = "numpy_mesh") -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / f"{prefix}_mesh.npz"
    metrics_path = out_dir / f"{prefix}_metrics.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    bridge_path = out_dir / f"{prefix}_rf77_summary.json"

    metrics = {"schema_name": "common.numpy_mesh.metrics", "schema_version": "1.0", **to_jsonable(runtime.diagnostics())}
    viewer_payload = build_visualization_payload(runtime, metrics=metrics)
    bridge = build_mode_bridge(runtime)
    bridge_summary = {
        "mode": bridge.mode.value,
        "shape_signature": to_jsonable(bridge.shape_signature()),
        "physical_groups": to_jsonable(bridge.physical_groups()),
        "boundary_tags": to_jsonable(bridge.boundary_tags()),
    }

    save_runtime(mesh_path, runtime)
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(viewer_path, viewer_payload)
    atomic_write_json(bridge_path, bridge_summary)
    return {
        "mesh": mesh_path,
        "metrics": metrics_path,
        "viewer_payload": viewer_path,
        "rf77_summary": bridge_path,
    }


def export_mode1_artifacts(output_dir: str | Path, runtime: NumpyMeshRuntime, *, prefix: str = "numpy_mode1") -> dict[str, Path]:
    return export_mode_artifacts(output_dir, runtime, prefix=prefix)


def export_mode2_artifacts(output_dir: str | Path, runtime: NumpyMeshRuntime, *, prefix: str = "numpy_mode2") -> dict[str, Path]:
    return export_mode_artifacts(output_dir, runtime, prefix=prefix)


def export_mode3_artifacts(output_dir: str | Path, runtime: NumpyMeshRuntime, *, prefix: str = "numpy_mode3") -> dict[str, Path]:
    return export_mode_artifacts(output_dir, runtime, prefix=prefix)


def export_mode4_artifacts(output_dir: str | Path, runtime: NumpyMeshRuntime, *, prefix: str = "numpy_mode4") -> dict[str, Path]:
    return export_mode_artifacts(output_dir, runtime, prefix=prefix)


def export_mode5_artifacts(output_dir: str | Path, runtime: NumpyMeshRuntime, *, prefix: str = "numpy_mode5") -> dict[str, Path]:
    return export_mode_artifacts(output_dir, runtime, prefix=prefix)

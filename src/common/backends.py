"""Backend registry for the common TopoSmplJAX track."""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import Any, NamedTuple

from topojax.ad.modes import MeshMovementMode, get_mesh_movement_mode


class BackendName(str, Enum):
    TOPO = "topo"
    SMPL = "smpl"


class BackendSpec(NamedTuple):
    backend: BackendName
    summary: str
    package: str
    implemented_modes: tuple[MeshMovementMode, ...]
    review_stub_modes: tuple[MeshMovementMode, ...]


_BACKEND_SPECS: tuple[BackendSpec, ...] = (
    BackendSpec(
        backend=BackendName.TOPO,
        summary="Topo track implementation for differentiable mesh generation and mesh operators.",
        package="topojax",
        implemented_modes=(
            MeshMovementMode.FIXED_TOPOLOGY,
            MeshMovementMode.REMESH_RESTART,
            MeshMovementMode.SOFT_CONNECTIVITY,
            MeshMovementMode.STRAIGHT_THROUGH,
            MeshMovementMode.FULLY_DYNAMIC,
        ),
        review_stub_modes=(),
    ),
    BackendSpec(
        backend=BackendName.SMPL,
        summary="SMPL track implementation for fixed-topology body-model meshes and parameter-routing workflows.",
        package="smpljax",
        implemented_modes=(
            MeshMovementMode.FIXED_TOPOLOGY,
            MeshMovementMode.REMESH_RESTART,
            MeshMovementMode.SOFT_CONNECTIVITY,
            MeshMovementMode.STRAIGHT_THROUGH,
            MeshMovementMode.FULLY_DYNAMIC,
        ),
        review_stub_modes=(),
    ),
)

__all__ = [
    "BackendName",
    "BackendSpec",
    "build_mode_bridge",
    "get_backend_mode_report",
    "get_backend_spec",
    "get_backends",
    "repair_print_mesh",
]


def get_backends() -> tuple[BackendSpec, ...]:
    return _BACKEND_SPECS


def get_backend_spec(backend: BackendName | str) -> BackendSpec:
    needle = BackendName(backend)
    for spec in _BACKEND_SPECS:
        if spec.backend == needle:
            return spec
    raise KeyError(f"Unknown backend: {backend}")


def get_backend_mode_report(backend: BackendName | str) -> dict[str, Any]:
    spec = get_backend_spec(backend)
    return {
        "backend": spec.backend.value,
        "package": spec.package,
        "summary": spec.summary,
        "implemented_modes": [get_mesh_movement_mode(mode)._asdict() for mode in spec.implemented_modes],
        "review_stub_modes": [get_mesh_movement_mode(mode)._asdict() for mode in spec.review_stub_modes],
    }


def build_mode_bridge(backend: BackendName | str, source: Any, mode: MeshMovementMode | str, **kwargs: Any):
    normalized_backend = BackendName(backend)
    if normalized_backend == BackendName.TOPO:
        topo_backend = import_module(".topo", __package__)
        return topo_backend.build_mode_bridge(source, mode, **kwargs)
    if normalized_backend == BackendName.SMPL:
        smpl_backend = import_module(".smpl", __package__)
        return smpl_backend.build_mode_bridge(source, mode, **kwargs)
    raise KeyError(f"Unknown backend: {backend}")


def repair_print_mesh(backend: BackendName | str, source: Any, **kwargs: Any):
    normalized_backend = BackendName(backend)
    if normalized_backend == BackendName.TOPO:
        topo_backend = import_module(".topo", __package__)
        if hasattr(source, "points") and hasattr(source, "topology"):
            return topo_backend.repair_print_mesh(source.points, source.topology.elements, **kwargs)
        return topo_backend.repair_print_mesh(source[0], source[1], **kwargs)
    if normalized_backend == BackendName.SMPL:
        smpl_backend = import_module(".smpl", __package__)
        return smpl_backend.repair_print_mesh(source, **kwargs)
    raise KeyError(f"Unknown backend: {backend}")

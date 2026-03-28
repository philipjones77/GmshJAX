"""Topo track adapters that obey the common TopoSmplJAX protocols."""

from __future__ import annotations

from typing import Any

from topojax.ad import workflow as topo_workflow
from topojax.ad.modes import MeshMovementMode
from topojax.rf77 import (
    RandomFields77ModeBridge,
    build_mode1_randomfields77_bridge,
    build_mode2_randomfields77_bridge,
    build_mode3_randomfields77_bridge,
    build_mode4_randomfields77_bridge,
    build_mode5_randomfields77_bridge,
)

from .mesh_repair import MeshRepairResult, RepairBackend, repair_topo_mesh_for_printing


Mode1Domain = topo_workflow.Mode1Domain
Mode1WorkflowRun = topo_workflow.Mode1WorkflowRun
Mode2Domain = topo_workflow.Mode2Domain
Mode2WorkflowRun = topo_workflow.Mode2WorkflowRun
Mode3Domain = topo_workflow.Mode3Domain
Mode3WorkflowRun = topo_workflow.Mode3WorkflowRun
Mode4Domain = topo_workflow.Mode4Domain
Mode4WorkflowRun = topo_workflow.Mode4WorkflowRun
Mode5Domain = topo_workflow.Mode5Domain
Mode5WorkflowRun = topo_workflow.Mode5WorkflowRun

__all__ = [
    "Mode1Domain",
    "Mode1WorkflowRun",
    "Mode2Domain",
    "Mode2WorkflowRun",
    "Mode3Domain",
    "Mode3WorkflowRun",
    "Mode4Domain",
    "Mode4WorkflowRun",
    "Mode5Domain",
    "Mode5WorkflowRun",
    "build_mode_bridge",
    "initialize_mode1_domain",
    "initialize_mode2_domain",
    "initialize_mode3_domain",
    "initialize_mode4_domain",
    "initialize_mode5_domain",
    "repair_print_mesh",
    "run_mode1_workflow",
    "run_mode3_workflow",
    "run_mode4_workflow",
    "run_mode5_workflow",
    "run_mode2_restart_workflow",
    "run_mode2_workflow",
]


def initialize_mode1_domain(kind: str, *, progress: bool = True, **kwargs: Any) -> Mode1Domain:
    """Initialize a Topo mode-1 domain through the common track."""
    return topo_workflow.initialize_mode1_domain(kind, progress=progress, **kwargs)


def initialize_mode2_domain(kind: str, *, progress: bool = True, **kwargs: Any) -> Mode2Domain:
    """Initialize a Topo mode-2 domain through the common track."""
    return topo_workflow.initialize_mode2_domain(kind, progress=progress, **kwargs)


def initialize_mode3_domain(kind: str, *, progress: bool = True, **kwargs: Any) -> Mode3Domain:
    """Initialize a Topo mode-3 domain through the common track."""
    return topo_workflow.initialize_mode3_domain(kind, progress=progress, **kwargs)


def initialize_mode4_domain(kind: str, *, progress: bool = True, **kwargs: Any) -> Mode4Domain:
    """Initialize a Topo mode-4 domain through the common track."""
    return topo_workflow.initialize_mode4_domain(kind, progress=progress, **kwargs)


def initialize_mode5_domain(kind: str, *, progress: bool = True, **kwargs: Any) -> Mode5Domain:
    """Initialize a Topo mode-5 domain through the common track."""
    return topo_workflow.initialize_mode5_domain(kind, progress=progress, **kwargs)


def run_mode1_workflow(domain: Mode1Domain, **kwargs: Any) -> Mode1WorkflowRun:
    """Run the Topo mode-1 workflow through the common track."""
    return topo_workflow.run_mode1_workflow(domain, **kwargs)


def run_mode2_restart_workflow(domain: Mode2Domain, **kwargs: Any) -> Mode2WorkflowRun:
    """Run the Topo mode-2 remesh-restart workflow through the common track."""
    return topo_workflow.run_mode2_restart_workflow(domain, **kwargs)


def run_mode2_workflow(domain: Mode2Domain, **kwargs: Any) -> Mode2WorkflowRun:
    """Convenience alias for the Topo mode-2 remesh-restart workflow."""
    return run_mode2_restart_workflow(domain, **kwargs)


def run_mode3_workflow(domain: Mode3Domain, **kwargs: Any) -> Mode3WorkflowRun:
    """Run the Topo mode-3 workflow through the common track."""
    return topo_workflow.run_mode3_workflow(domain, **kwargs)


def run_mode4_workflow(domain: Mode4Domain, **kwargs: Any) -> Mode4WorkflowRun:
    """Run the Topo mode-4 workflow through the common track."""
    return topo_workflow.run_mode4_workflow(domain, **kwargs)


def run_mode5_workflow(domain: Mode5Domain, **kwargs: Any) -> Mode5WorkflowRun:
    """Run the Topo mode-5 workflow through the common track."""
    return topo_workflow.run_mode5_workflow(domain, **kwargs)


def build_mode_bridge(source: Any, mode: MeshMovementMode | str, **kwargs: Any) -> RandomFields77ModeBridge:
    normalized = MeshMovementMode(mode)
    if normalized == MeshMovementMode.FIXED_TOPOLOGY:
        return build_mode1_randomfields77_bridge(source, **kwargs)
    if normalized == MeshMovementMode.REMESH_RESTART:
        return build_mode2_randomfields77_bridge(source, **kwargs)
    if normalized == MeshMovementMode.SOFT_CONNECTIVITY:
        if hasattr(source, "domain") and hasattr(source, "result"):
            return build_mode3_randomfields77_bridge(source, **kwargs)
        if hasattr(source, "points") and hasattr(source, "topology"):
            return build_mode3_randomfields77_bridge(source.points, source.topology, metadata=getattr(source, "metadata", None), **kwargs)
        raise TypeError("Mode 3 topo bridge expects a source with `.points` and `.topology`.")
    if normalized == MeshMovementMode.STRAIGHT_THROUGH:
        if hasattr(source, "domain") and hasattr(source, "result"):
            return build_mode4_randomfields77_bridge(source, **kwargs)
        if hasattr(source, "points") and hasattr(source, "topology"):
            return build_mode4_randomfields77_bridge(source.points, source.topology, metadata=getattr(source, "metadata", None), **kwargs)
        raise TypeError("Mode 4 topo bridge expects a source with `.points` and `.topology`.")
    if normalized == MeshMovementMode.FULLY_DYNAMIC:
        if hasattr(source, "domain") and hasattr(source, "result"):
            return build_mode5_randomfields77_bridge(source, **kwargs)
        if hasattr(source, "points") and hasattr(source, "topology"):
            return build_mode5_randomfields77_bridge(source.points, source.topology, metadata=getattr(source, "metadata", None), **kwargs)
        raise TypeError("Mode 5 topo bridge expects a source with `.points` and `.topology`.")
    raise KeyError(f"Unsupported topo mode: {mode}")


def repair_print_mesh(
    points: Any,
    elements: Any,
    *,
    element_kind: str | None = None,
    backend: RepairBackend = "auto",
    **kwargs: Any,
) -> MeshRepairResult:
    return repair_topo_mesh_for_printing(points, elements, element_kind=element_kind, backend=backend, **kwargs)

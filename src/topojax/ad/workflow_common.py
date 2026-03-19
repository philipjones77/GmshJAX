"""Shared workflow-domain initialization helpers for Mode 1 and Mode 2."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from topojax.io.imports import load_gmsh_msh
from topojax.mesh.domains import (
    DomainMeshMetadata,
    box_volume_tet_mesh_tagged,
    extruded_polygon_tet_mesh,
    implicit_volume_tet_mesh_tagged,
    polygon_domain_quad_mesh_tagged,
    polygon_domain_tri_mesh_tagged,
    sphere_surface_tri_mesh_tagged,
    sphere_volume_tet_mesh_tagged,
)
from topojax.mesh.topology import (
    MeshTopology,
    mapped_quad_mesh,
    polyline_mesh,
    unit_cube_tet_mesh,
    unit_interval_line_mesh,
    unit_square_quad_mesh,
    unit_square_tri_mesh,
)


class MeshWorkflowDomain(NamedTuple):
    topology: MeshTopology
    points: jnp.ndarray
    metadata: DomainMeshMetadata | None = None


def initialize_workflow_domain(kind: str, **kwargs: Any) -> MeshWorkflowDomain:
    """Initialize a supported domain for fixed-topology or restart workflows."""
    if kind in ("line", "interval", "polyline"):
        if "points" in kwargs:
            topo, points = polyline_mesh(kwargs["points"], closed=bool(kwargs.get("closed", False)))
        else:
            topo, points = unit_interval_line_mesh(int(kwargs.get("n", 16)))
        return MeshWorkflowDomain(topology=topo, points=points)
    if kind == "square":
        nx = int(kwargs.get("nx", 16))
        ny = int(kwargs.get("ny", 16))
        family = str(kwargs.get("family", "tri"))
        dtype = jnp.asarray(0.0).dtype
        bbox_min = jnp.asarray(kwargs.get("bbox_min", [0.0, 0.0]), dtype=dtype)
        bbox_max = jnp.asarray(kwargs.get("bbox_max", [1.0, 1.0]), dtype=dtype)
        if "map_fn" in kwargs:
            if family != "quad":
                raise ValueError("square with map_fn currently supports family='quad' only")
            topo, points = mapped_quad_mesh(kwargs["map_fn"], nx, ny)
        elif family == "tri":
            topo, points = unit_square_tri_mesh(nx, ny)
            points = bbox_min[None, :] + points * (bbox_max - bbox_min)[None, :]
        elif family == "quad":
            topo, points = unit_square_quad_mesh(nx, ny)
            points = bbox_min[None, :] + points * (bbox_max - bbox_min)[None, :]
        else:
            raise ValueError("square family must be 'tri' or 'quad'")
        return MeshWorkflowDomain(topo, points)
    if kind == "polygon":
        topo, points, metadata = polygon_domain_tri_mesh_tagged(
            kwargs["outer_boundary"],
            holes=kwargs.get("holes"),
            target_edge_size=kwargs.get("target_edge_size"),
            interior_relaxation=float(kwargs.get("interior_relaxation", 0.35)),
        )
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "polygon-quad":
        topo, points, metadata = polygon_domain_quad_mesh_tagged(
            kwargs["outer_boundary"],
            holes=kwargs.get("holes"),
            target_edge_size=kwargs.get("target_edge_size"),
            interior_relaxation=float(kwargs.get("interior_relaxation", 0.35)),
        )
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "extruded":
        topo, points, metadata = extruded_polygon_tet_mesh(
            kwargs["outer_boundary"],
            holes=kwargs.get("holes"),
            target_edge_size=kwargs.get("target_edge_size"),
            height=float(kwargs.get("height", 1.0)),
            layers=int(kwargs.get("layers", 4)),
        )
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "box-volume":
        topo, points, metadata = box_volume_tet_mesh_tagged(
            kwargs["bbox_min"],
            kwargs["bbox_max"],
            int(kwargs["nx"]),
            int(kwargs["ny"]),
            int(kwargs["nz"]),
        )
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "box":
        if "bbox_min" in kwargs or "bbox_max" in kwargs:
            topo, points, metadata = box_volume_tet_mesh_tagged(
                kwargs.get("bbox_min", jnp.asarray([0.0, 0.0, 0.0])),
                kwargs.get("bbox_max", jnp.asarray([1.0, 1.0, 1.0])),
                int(kwargs["nx"]),
                int(kwargs["ny"]),
                int(kwargs["nz"]),
            )
        else:
            topo, points = unit_cube_tet_mesh(int(kwargs["nx"]), int(kwargs["ny"]), int(kwargs["nz"]))
            metadata = None
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "implicit-volume":
        topo, points, metadata = implicit_volume_tet_mesh_tagged(
            kwargs["level_set_fn"],
            kwargs["bbox_min"],
            kwargs["bbox_max"],
            int(kwargs["nx"]),
            int(kwargs["ny"]),
            int(kwargs["nz"]),
        )
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "sphere-volume":
        topo, points, metadata = sphere_volume_tet_mesh_tagged(
            kwargs["center"],
            float(kwargs["radius"]),
            int(kwargs["nx"]),
            int(kwargs["ny"]),
            int(kwargs["nz"]),
            padding=float(kwargs.get("padding", 0.0)),
        )
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "sphere-surface":
        topo, points, metadata = sphere_surface_tri_mesh_tagged(
            kwargs["center"],
            float(kwargs["radius"]),
            int(kwargs.get("n_lat", 12)),
            int(kwargs.get("n_lon", 24)),
        )
        return MeshWorkflowDomain(topo, points, metadata)
    if kind == "import-msh":
        imported = load_gmsh_msh(kwargs["path"], primary_element_kind=kwargs.get("primary_element_kind"))
        metadata = DomainMeshMetadata(
            boundary_element_blocks=imported.extra_element_blocks,
            physical_names=imported.physical_names,
        )
        return MeshWorkflowDomain(imported.topology, imported.points, metadata)
    raise ValueError("Unsupported workflow domain kind")

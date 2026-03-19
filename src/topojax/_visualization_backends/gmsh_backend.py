"""Native Gmsh visualization backend."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import jax.numpy as jnp

from topojax.io.exports import export_gmsh_msh
from topojax.io.gmsh_viewer import launch_gmsh_viewer
from topojax.mesh.topology import MeshTopology


def plot_mode1_gmsh(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    mesh_path: str | Path | None = None,
    gmsh_executable: str = "gmsh",
    wait: bool = False,
    extra_args: list[str] | None = None,
):
    """Export a `.msh` file and open it in native Gmsh."""
    if mesh_path is None:
        with NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
            target = Path(tmp.name)
    else:
        target = Path(mesh_path)
    export_gmsh_msh(target, points, topology.elements, element_entity_tags=topology.element_entity_tags)
    proc = launch_gmsh_viewer(target, gmsh_executable=gmsh_executable, wait=wait, extra_args=extra_args)
    try:
        setattr(proc, "_topojax_mesh_path", str(target))
    except Exception:
        pass
    return proc

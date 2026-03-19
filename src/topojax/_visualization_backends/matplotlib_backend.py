"""Matplotlib visualization backend."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from topojax.mesh.topology import MeshTopology
from topojax.visualization import _points3, _tet_boundary_faces


def plot_mode1_matplotlib(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    title: str = "Mode 1 Fixed-Topology Mesh",
):
    """Return a Matplotlib figure for a mesh state."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    pts = np.asarray(points)
    edges = np.asarray(topology.edges, dtype=np.int32)
    order = int(topology.elements.shape[1])
    dim = int(pts.shape[1])

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        segs = pts[edges]
        ax.add_collection(LineCollection(segs, colors="black", linewidths=0.8))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c="tab:blue")
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        return fig

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    pts3 = _points3(points)
    if order == 3:
        poly = Poly3DCollection(
            pts3[np.asarray(topology.elements, dtype=np.int32)],
            alpha=0.25,
            facecolor="tab:blue",
            edgecolor="black",
        )
        ax.add_collection3d(poly)
    elif order == 4:
        faces = _tet_boundary_faces(np.asarray(topology.elements, dtype=np.int32))
        poly = Poly3DCollection(pts3[faces], alpha=0.25, facecolor="tab:blue", edgecolor="black")
        ax.add_collection3d(poly)
    lines = pts3[edges]
    ax.add_collection3d(Line3DCollection(lines, colors="black", linewidths=0.6))
    ax.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], s=8, c="tab:red")
    ax.set_title(title)
    return fig

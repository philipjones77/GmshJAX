from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from smpljax.visualization.common import skeleton_connections_from_parents

if TYPE_CHECKING:
    from smpljax.mode1 import SMPLMode1OptimizationResult


def plot_mode1_matplotlib(
    result: SMPLMode1OptimizationResult,
    *,
    faces: np.ndarray,
    title: str = "SMPL Mode 1 Result",
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = np.asarray(result.output.vertices[0], dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(verts[tris], alpha=0.35, facecolor="#dcaa78", edgecolor="black")
    ax.add_collection3d(poly)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=4, c="#1f77b4")
    joints = np.asarray(result.output.joints[0], dtype=np.float32)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=10, c="#d62728")
    ax.set_title(title)
    return fig


def plot_mode1_pyvista(
    result: SMPLMode1OptimizationResult,
    *,
    faces: np.ndarray,
    parents: np.ndarray,
    title: str = "SMPL Mode 1 Result",
    show: bool = False,
):
    import pyvista as pv

    verts = np.asarray(result.output.vertices[0], dtype=np.float32)
    joints = np.asarray(result.output.joints[0], dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    face_cells = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int32), tris]).reshape(-1)
    mesh = pv.PolyData(verts, face_cells)

    plotter = pv.Plotter(off_screen=not show)
    plotter.add_mesh(mesh, color="#dcaa78", smooth_shading=True, show_edges=False)
    plotter.add_points(joints, color="#d62728", point_size=10, render_points_as_spheres=True)
    skeleton = pv.PolyData(joints)
    connections = skeleton_connections_from_parents(np.asarray(parents, dtype=np.int32))
    if connections:
        skeleton.lines = np.asarray([[2, parent, child] for parent, child in connections], dtype=np.int64).reshape(-1)
        plotter.add_mesh(skeleton, color="#1f77b4", line_width=3)
    plotter.add_title(title)
    if show:
        plotter.show()
    return plotter


def plot_mode1_viser(
    result: SMPLMode1OptimizationResult,
    *,
    faces: np.ndarray,
    host: str = "127.0.0.1",
    port: int = 8090,
):
    import viser

    verts = np.asarray(result.output.vertices[0], dtype=np.float32)
    joints = np.asarray(result.output.joints[0], dtype=np.float32)
    tris = np.asarray(faces, dtype=np.uint32)

    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", plane="xz")
    server.scene.add_mesh_simple("/smpl/mode1_mesh", vertices=verts, faces=tris, color=(220, 170, 120))
    server.scene.add_point_cloud(
        "/smpl/mode1_joints",
        points=joints,
        colors=np.tile(np.array([[214, 39, 40]], dtype=np.uint8), (joints.shape[0], 1)),
        point_size=0.02,
    )
    return server


def visualize_mode1_result(
    result: SMPLMode1OptimizationResult,
    *,
    backend: str = "matplotlib",
    title: str = "SMPL Mode 1 Result",
    show: bool = False,
    host: str = "127.0.0.1",
    port: int = 8090,
):
    faces = np.asarray(result.faces, dtype=np.int32)
    parents = np.asarray(result.parents, dtype=np.int32)
    if backend == "matplotlib":
        return plot_mode1_matplotlib(result, faces=faces, title=title)
    if backend == "pyvista":
        return plot_mode1_pyvista(result, faces=faces, parents=parents, title=title, show=show)
    if backend == "viser":
        return plot_mode1_viser(result, faces=faces, host=host, port=port)
    raise ValueError(f"Unsupported visualization backend: {backend}")

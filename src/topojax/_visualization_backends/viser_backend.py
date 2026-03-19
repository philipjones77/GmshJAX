"""Viser visualization backend."""

from __future__ import annotations

import time

import numpy as np

from topojax.visualization import TopoVisualizationState, _points3, _surface_faces


def plot_topo_viser(
    state: TopoVisualizationState,
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    block: bool = False,
):
    """Launch a minimal Viser scene for a Topo mode state."""
    try:
        import viser
    except Exception as exc:
        raise ModuleNotFoundError("viser is not installed; install viser to enable this backend") from exc

    server = viser.ViserServer(host=host, port=port)
    scene = server.scene
    scene.set_up_direction("+z")
    scene.add_grid("/grid", plane="xy")
    pts = _points3(state.points).astype(np.float32)
    faces = _surface_faces(state.points, state.topology)

    if faces is None:
        scene.add_point_cloud(
            "/topo/points",
            points=pts,
            colors=np.tile(np.array([[30, 144, 255]], dtype=np.uint8), (pts.shape[0], 1)),
            point_size=0.02,
        )
    else:
        scene.add_mesh_simple("/topo/mesh", vertices=pts, faces=np.asarray(faces, dtype=np.uint32), color=(173, 216, 230))
        scene.add_point_cloud(
            "/topo/nodes",
            points=pts,
            colors=np.tile(np.array([[220, 20, 60]], dtype=np.uint8), (pts.shape[0], 1)),
            point_size=0.012,
        )
    if hasattr(scene, "add_line_segments"):
        edge_pts = pts[np.asarray(state.topology.edges, dtype=np.int32)]
        edge_colors = np.tile(np.array([[[40, 40, 40], [40, 40, 40]]], dtype=np.uint8), (edge_pts.shape[0], 1, 1))
        scene.add_line_segments("/topo/edges", points=edge_pts, colors=edge_colors)

    if block:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return server

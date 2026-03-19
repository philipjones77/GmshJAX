"""PyVista visualization backend."""

from __future__ import annotations

import os

from topojax.visualization import TopoVisualizationState, _points3, build_pyvista_dataset


def plot_topo_pyvista(
    state: TopoVisualizationState,
    *,
    show: bool = False,
    show_nodes: bool = True,
    show_edges: bool = True,
    background_color: str = "white",
):
    """Build a PyVista plotter for a Topo mode state."""
    dataset = build_pyvista_dataset(state.points, state.topology)
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyvista is not installed; install pyvista to enable this backend") from exc

    off_screen = not show or os.environ.get("PYVISTA_OFF_SCREEN", "").lower() in {"1", "true", "yes"}
    plotter = pv.Plotter(off_screen=off_screen)
    if hasattr(plotter, "set_background"):
        plotter.set_background(background_color)
    plotter.add_mesh(dataset, show_edges=show_edges, color="lightblue")
    if show_nodes:
        plotter.add_points(_points3(state.points), color="tomato", point_size=8, render_points_as_spheres=True)
    if hasattr(plotter, "view_isometric"):
        plotter.view_isometric()
    plotter.add_title(state.title)
    if hasattr(plotter, "render"):
        plotter.render()
    if show:
        plotter.show()
    return plotter

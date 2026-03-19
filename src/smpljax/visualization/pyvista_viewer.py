from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smpljax.body_models import SMPLJAXModel
from smpljax.diagnostics import DiagnosticsLogger, write_runtime_diagnostics
from smpljax.optimized import OptimizedSMPLJAX
from smpljax.visualization.common import (
    ViewerConfig,
    ViewerState,
    ViewerAppearance,
    add_point_labels_compat,
    available_presets,
    camera_from_triplet,
    create_polydata_from_vertices_faces,
    create_skeleton_polydata,
    load_viewer_state,
    preset_named,
    save_viewer_state,
    evaluate_model,
    resolve_runtime,
    skeleton_connections_from_parents,
    summarize_viewer_state,
)


@dataclass(frozen=True)
class PyVistaViewerConfig(ViewerConfig):
    window_size: tuple[int, int] = (1280, 900)
    show_joints: bool = True
    show_skeleton: bool = True
    joint_labels: bool = False
    off_screen: bool = False
    screenshot_path: Path | None = None
    mesh_opacity: float = 1.0
    appearance: ViewerAppearance = ViewerAppearance()


def run_pyvista_viewer(
    config: PyVistaViewerConfig,
    model: SMPLJAXModel | None = None,
    runtime: OptimizedSMPLJAX | None = None,
) -> Any:
    """Run the PyVista SMPL viewer.

    Portions of the mesh/joint/skeleton viewer surface are adapted from ideas in
    the MIT-licensed `smplx-toolbox` visualization module, reworked for this
    repository's runtime and data model.
    """
    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError("pyvista is required. Install with: python -m pip install pyvista") from exc

    runtime_obj = resolve_runtime(config=config, model=model, runtime=runtime)
    if runtime_obj.data.faces_tensor is None:
        raise ValueError("Model file must include `faces_tensor` (or `f`/`faces`) for visualization.")

    total_joints = int(np.asarray(runtime_obj.data.parents).shape[0])
    num_betas = int(
        min(config.max_betas, runtime_obj.data.num_betas or np.asarray(runtime_obj.data.shapedirs).shape[-1])
    )
    num_expr = int(min(config.max_expression, runtime_obj.data.num_expression_coeffs or 0))

    if config.state_json is not None:
        state, camera = load_viewer_state(config.state_json)
    else:
        preset = preset_named(
            config.preset,
            runtime_obj,
            max_betas=config.max_betas,
            max_expression=config.max_expression,
        )
        state, camera = preset.state, preset.camera
    betas = state.betas.copy()
    expr = None if state.expression is None else state.expression.copy()
    full_pose = state.full_pose_aa.copy()
    transl = state.transl.copy()

    faces = np.asarray(runtime_obj.data.faces_tensor, dtype=np.int32)
    mesh = create_polydata_from_vertices_faces(np.asarray(runtime_obj.data.v_template, dtype=np.float32), faces)
    skeleton_connections = skeleton_connections_from_parents(np.asarray(runtime_obj.data.parents, dtype=np.int32))

    plotter = pv.Plotter(window_size=config.window_size, off_screen=config.off_screen)
    update_counter = {"value": 0}
    diagnostics_logger = DiagnosticsLogger(config.diagnostics_jsonl) if config.diagnostics_jsonl is not None else None
    actors: dict[str, Any] = {}
    plotter.add_axes()
    plotter.set_background(config.appearance.background_color)
    actor = plotter.add_mesh(
        mesh,
        color=config.appearance.mesh_color,
        smooth_shading=True,
        show_edges=False,
        opacity=float(config.mesh_opacity),
    )
    actors["mesh"] = actor
    plotter.add_text("smplJAX + pyvista", position="upper_left", font_size=10)
    if camera is not None:
        plotter.camera_position = [camera.position, camera.focal_point, camera.up]

    def _compute_vertices() -> np.ndarray:
        out = evaluate_model(
            runtime_obj,
            state=ViewerState(
                betas=betas,
                expression=expr,
                full_pose_aa=full_pose,
                transl=transl,
            ),
        )
        return np.asarray(out.vertices[0], dtype=np.float32)

    def _compute_joints() -> np.ndarray:
        out = evaluate_model(
            runtime_obj,
            state=ViewerState(
                betas=betas,
                expression=expr,
                full_pose_aa=full_pose,
                transl=transl,
            ),
        )
        return np.asarray(out.joints[0], dtype=np.float32)

    def _reset_overlay(name: str) -> None:
        actor_obj = actors.pop(name, None)
        if actor_obj is not None:
            try:
                plotter.remove_actor(actor_obj)
            except Exception:
                pass

    def _refresh_mesh() -> None:
        vertices = _compute_vertices()
        mesh.points = vertices
        joints = _compute_joints()
        _reset_overlay("joints")
        _reset_overlay("joint_labels")
        _reset_overlay("skeleton")
        if config.show_joints:
            actors["joints"] = plotter.add_points(
                joints,
                color=config.appearance.joint_color,
                point_size=10,
                render_points_as_spheres=True,
            )
            if config.joint_labels:
                joint_names = [f"j{idx}" for idx in range(joints.shape[0])]
                actors["joint_labels"] = add_point_labels_compat(plotter, joints, joint_names, font_size=12)
        if config.show_skeleton:
            skeleton = create_skeleton_polydata(joints, skeleton_connections)
            actors["skeleton"] = plotter.add_mesh(
                skeleton,
                color=config.appearance.skeleton_color,
                line_width=3,
            )
        if config.export_state_json is not None:
            save_viewer_state(
                config.export_state_json,
                ViewerState(betas=betas, expression=expr, full_pose_aa=full_pose, transl=transl),
                camera=camera_from_triplet(plotter.camera_position),
            )
        update_counter["value"] += 1
        if update_counter["value"] % max(config.diagnostics_every_n_updates, 1) == 0:
            viewer_diag = summarize_viewer_state(
                state=ViewerState(betas=betas, expression=expr, full_pose_aa=full_pose, transl=transl),
                num_vertices=int(vertices.shape[0]),
                num_joints=int(np.asarray(runtime_obj.data.parents).shape[0]),
                preset=config.preset,
                use_optimized_runtime=isinstance(runtime_obj, OptimizedSMPLJAX),
                update_index=update_counter["value"],
            )
            if isinstance(runtime_obj, OptimizedSMPLJAX) and config.diagnostics_json is not None:
                write_runtime_diagnostics(
                    config.diagnostics_json,
                    runtime=runtime_obj,
                    extra={"viewer": viewer_diag},
                )
            if diagnostics_logger is not None:
                diagnostics_logger.append(
                    {
                        "event": "viewer_update",
                        "backend": "pyvista",
                        "viewer": viewer_diag,
                        "runtime": (runtime_obj.diagnostics() if isinstance(runtime_obj, OptimizedSMPLJAX) else None),
                    }
                )
        plotter.render()

    def _slider_callback_factory(target: np.ndarray, row: int, col: int):
        def _callback(value: float) -> None:
            target[row, col] = float(value)
            _refresh_mesh()

        return _callback

    ypos = 0.95
    for beta_idx in range(num_betas):
        plotter.add_slider_widget(
            _slider_callback_factory(betas, 0, beta_idx),
            rng=[-3.0, 3.0],
            value=0.0,
            title=f"beta_{beta_idx}",
            pointa=(0.02, ypos),
            pointb=(0.28, ypos),
            style="modern",
        )
        ypos -= 0.035

    expr_y = 0.95
    for expr_idx in range(num_expr):
        plotter.add_slider_widget(
            _slider_callback_factory(expr, 0, expr_idx),
            rng=[-3.0, 3.0],
            value=0.0,
            title=f"expr_{expr_idx}",
            pointa=(0.32, expr_y),
            pointb=(0.58, expr_y),
            style="modern",
        )
        expr_y -= 0.035

    pose_y = 0.95
    for joint_idx in range(min(total_joints, 8)):
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            plotter.add_slider_widget(
                _slider_callback_factory(full_pose[0], joint_idx, axis_idx),
                rng=[-config.joint_limit, config.joint_limit],
                value=0.0,
                title=f"j{joint_idx}_{axis_name}",
                pointa=(0.64, pose_y),
                pointb=(0.94, pose_y),
                style="modern",
            )
            pose_y -= 0.03

    for axis_idx, axis_name in enumerate(("tx", "ty", "tz")):
        pos = max(0.08 - 0.03 * axis_idx, 0.02)
        plotter.add_slider_widget(
            _slider_callback_factory(transl, 0, axis_idx),
            rng=[-2.0, 2.0],
            value=0.0,
            title=axis_name,
            pointa=(0.02, pos),
            pointb=(0.28, pos),
            style="modern",
        )

    def _toggle_wireframe(state: bool) -> None:
        actor.prop.show_edges = bool(state)
        actor.prop.style = "wireframe" if state else "surface"
        plotter.render()

    plotter.add_checkbox_button_widget(_toggle_wireframe, value=False, position=(15, 15), size=28)
    _refresh_mesh()
    if config.screenshot_path is not None:
        config.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(config.screenshot_path))
    if not config.off_screen:
        plotter.show(auto_close=False)
    return plotter


def _parse_args() -> PyVistaViewerConfig:
    parser = argparse.ArgumentParser(description="Run smplJAX visualizer with pyvista.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-betas", type=int, default=10)
    parser.add_argument("--max-expression", type=int, default=10)
    parser.add_argument("--joint-limit", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--preset", type=str, choices=list(available_presets()), default="neutral")
    parser.add_argument("--state-json", type=Path, default=None)
    parser.add_argument("--export-state-json", type=Path, default=None)
    parser.add_argument("--no-optimized-runtime", action="store_true")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--hide-joints", action="store_true")
    parser.add_argument("--hide-skeleton", action="store_true")
    parser.add_argument("--joint-labels", action="store_true")
    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--screenshot-path", type=Path, default=None)
    args = parser.parse_args()
    return PyVistaViewerConfig(
        model_path=args.model_path,
        max_betas=args.max_betas,
        max_expression=args.max_expression,
        joint_limit=args.joint_limit,
        fps=args.fps,
        preset=args.preset,
        state_json=args.state_json,
        export_state_json=args.export_state_json,
        use_optimized_runtime=not args.no_optimized_runtime,
        window_size=(args.window_width, args.window_height),
        show_joints=not args.hide_joints,
        show_skeleton=not args.hide_skeleton,
        joint_labels=args.joint_labels,
        off_screen=args.off_screen,
        screenshot_path=args.screenshot_path,
    )


def main() -> None:
    run_pyvista_viewer(_parse_args())


if __name__ == "__main__":
    main()

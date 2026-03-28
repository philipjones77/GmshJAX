import json
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from smpljax import (
    SMPLJAXModel,
    build_mode3_visualization_payload,
    build_mode4_visualization_payload,
    build_mode5_visualization_payload,
    export_mode3_artifacts,
    export_mode4_artifacts,
    export_mode5_artifacts,
    mode3_history_payload,
    mode3_metrics_payload,
    mode4_history_payload,
    mode4_metrics_payload,
    mode5_history_payload,
    mode5_metrics_payload,
    optimize_mode3,
    optimize_mode4,
    optimize_mode5,
    visualize_mode4_result,
    visualize_mode5_result,
)
from smpljax.utils import SMPLModelData


def _toy_model() -> SMPLJAXModel:
    return SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
            shapedirs=jnp.zeros((4, 3, 10), dtype=jnp.float32),
            posedirs=jnp.zeros((27, 12), dtype=jnp.float32),
            j_regressor=jnp.ones((4, 4), dtype=jnp.float32) / 4.0,
            parents=jnp.array([-1, 0, 1, 2], dtype=jnp.int32),
            lbs_weights=jnp.ones((4, 4), dtype=jnp.float32) / 4.0,
            num_betas=10,
            num_body_joints=3,
            num_hand_joints=0,
            num_face_joints=0,
            faces_tensor=jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
        )
    )


def _mode_params() -> dict[str, jnp.ndarray]:
    return {
        "betas": jnp.zeros((1, 10), dtype=jnp.float32),
        "body_pose": jnp.zeros((1, 3, 3), dtype=jnp.float32),
        "global_orient": jnp.zeros((1, 1, 3), dtype=jnp.float32),
        "transl": jnp.array([[0.5, -0.25, 0.1]], dtype=jnp.float32),
    }


def test_mode3_soft_routing_reduces_objective_and_exports_artifacts(tmp_path: Path) -> None:
    result = optimize_mode3(
        _toy_model(),
        _mode_params(),
        steps=6,
        step_size=0.2,
        logit_step_size=0.1,
        diagnostics_every=2,
        activation_weight=0.0,
    )
    metrics = mode3_metrics_payload(result)
    history = mode3_history_payload(result)
    viewer = build_mode3_visualization_payload(result)
    artifacts = export_mode3_artifacts(tmp_path, result, prefix="mode3")
    groups = json.loads(artifacts["groups"].read_text(encoding="utf-8"))

    assert result.implementation_status == "implemented-soft-routing"
    assert float(result.objective_history[-1]) <= float(result.objective_history[0]) + 1.0e-8
    assert metrics["schema_name"] == "smpljax.mode3.metrics"
    assert metrics["n_groups"] >= 1
    assert history["step"].tolist() == list(range(1, 7))
    assert viewer["schema_name"] == "smpljax.mode3.visualization"
    assert artifacts["metrics"].exists()
    assert artifacts["history"].exists()
    assert artifacts["history_json"].exists()
    assert artifacts["history_csv"].exists()
    assert groups[0]["group_name"] == "translation"


def test_mode4_straight_through_reduces_objective_and_visualizes(monkeypatch, tmp_path: Path) -> None:
    class _FakeScene:
        def set_up_direction(self, direction):
            self.direction = direction

        def add_grid(self, path, position=(0.0, 0.0, 0.0), plane="xz"):
            self.grid = (path, position, plane)

        def add_mesh_simple(self, path, *, vertices, faces, color):
            self.mesh = (path, vertices, faces, color)

        def add_point_cloud(self, path, *, points, colors, point_size):
            self.points = (path, points, colors, point_size)

        def add_line_segments(self, path, *, points, colors, line_width):
            self.lines = (path, points, colors, line_width)

        def add_label(self, path, text, position):
            self.label = (path, text, position)

    class _FakeServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.scene = _FakeScene()

    monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=_FakeServer))
    result = optimize_mode4(
        _toy_model(),
        _mode_params(),
        steps=6,
        step_size=0.2,
        logit_step_size=0.1,
        diagnostics_every=2,
        activation_weight=0.0,
    )
    metrics = mode4_metrics_payload(result)
    history = mode4_history_payload(result)
    viewer = build_mode4_visualization_payload(result)
    artifacts = export_mode4_artifacts(tmp_path, result, prefix="mode4")
    server = visualize_mode4_result(result, backend="viser")

    assert result.implementation_status == "implemented-straight-through-routing"
    assert float(result.objective_history[-1]) <= float(result.objective_history[0]) + 1.0e-8
    assert metrics["schema_name"] == "smpljax.mode4.metrics"
    assert history["step"].tolist() == list(range(1, 7))
    assert viewer["schema_name"] == "smpljax.mode4.visualization"
    assert np.all(np.isin(np.asarray(result.hard_weights), np.array([0.0, 1.0], dtype=np.float32)))
    assert artifacts["groups"].exists()
    assert server.host == "127.0.0.1"
    assert server.scene.lines[0] == "/smpl/mode1_skeleton"


def test_mode5_dynamic_controller_reduces_objective_and_exports_histories(monkeypatch, tmp_path: Path) -> None:
    class _FakeScene:
        def set_up_direction(self, direction):
            self.direction = direction

        def add_grid(self, path, position=(0.0, 0.0, 0.0), plane="xz"):
            self.grid = (path, position, plane)

        def add_mesh_simple(self, path, *, vertices, faces, color):
            self.mesh = (path, vertices, faces, color)

        def add_point_cloud(self, path, *, points, colors, point_size):
            self.points = (path, points, colors, point_size)

        def add_line_segments(self, path, *, points, colors, line_width):
            self.lines = (path, points, colors, line_width)

        def add_label(self, path, text, position):
            self.label = (path, text, position)

    class _FakeServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.scene = _FakeScene()

    monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=_FakeServer))
    result = optimize_mode5(
        _toy_model(),
        _mode_params(),
        cycles=2,
        surrogate_steps=2,
        refinement_steps=4,
        surrogate_step_size=0.1,
        refinement_step_size=0.2,
        diagnostics_every=1,
        activation_weight=0.0,
    )
    metrics = mode5_metrics_payload(result)
    history = mode5_history_payload(result)
    viewer = build_mode5_visualization_payload(result)
    artifacts = export_mode5_artifacts(tmp_path, result, prefix="mode5")
    controller = json.loads(artifacts["controller"].read_text(encoding="utf-8"))
    transfer = json.loads(artifacts["transfer"].read_text(encoding="utf-8"))
    server = visualize_mode5_result(result, backend="viser")

    assert result.implementation_status == "implemented-dynamic-controller"
    assert float(result.objective_history[-1]) <= float(result.objective_history[0]) + 1.0e-8
    assert len(result.phase_summaries) == 2
    assert len(result.controller_history) == 2
    assert len(result.transfer_history) == 2
    assert metrics["schema_name"] == "smpljax.mode5.metrics"
    assert history["step"].tolist() == list(range(1, 9))
    assert viewer["schema_name"] == "smpljax.mode5.visualization"
    assert len(controller) == 2
    assert len(transfer) == 2
    assert artifacts["phases"].exists()
    assert artifacts["history_csv"].exists()
    assert server.host == "127.0.0.1"
    assert server.scene.lines[0] == "/smpl/mode1_skeleton"

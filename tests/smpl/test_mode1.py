import json
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from smpljax import (
    SMPLJAXModel,
    SMPLMode1Provision,
    SMPLMode2StageConfig,
    SMPLMode2OptimizationResult,
    build_mode1_visualization_payload,
    build_mode2_visualization_payload,
    default_mode1_params,
    export_mode1_artifacts,
    export_mode2_artifacts,
    initialize_mode1_model,
    load_mode1_snapshot,
    mode1_history_payload,
    mode1_metrics_payload,
    mode2_history_payload,
    mode2_metrics_payload,
    optimize_mode1,
    optimize_mode2,
    plot_mode1_matplotlib,
    plot_mode1_pyvista,
    plot_mode1_viser,
    run_mode1_workflow,
    visualize_mode1_result,
    visualize_mode2_result,
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


def _mode1_params() -> dict[str, jnp.ndarray]:
    return {
        "betas": jnp.zeros((1, 10), dtype=jnp.float32),
        "body_pose": jnp.zeros((1, 3, 3), dtype=jnp.float32),
        "global_orient": jnp.zeros((1, 1, 3), dtype=jnp.float32),
        "transl": jnp.array([[0.5, -0.25, 0.1]], dtype=jnp.float32),
    }


def test_mode1_optimization_reduces_translation_energy() -> None:
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=8, step_size=0.2, diagnostics_every=2)
    metrics = mode1_metrics_payload(result)
    history = mode1_history_payload(result)

    assert float(result.objective_history[-1]) <= float(result.objective_history[0]) + 1.0e-8
    assert metrics["schema_name"] == "smpljax.mode1.metrics"
    assert metrics["n_steps"] == 8
    assert history["step"].tolist() == list(range(1, 9))


def test_default_mode1_params_shapes() -> None:
    model = _toy_model()
    params = default_mode1_params(model, batch_size=2)
    assert params["betas"].shape == (2, 10)
    assert params["body_pose"].shape == (2, 3, 3)
    assert params["global_orient"].shape == (2, 1, 3)
    assert params["transl"].shape == (2, 3)


def test_initialize_mode1_model_with_existing_model() -> None:
    provision = initialize_mode1_model(model=_toy_model(), batch_size=1, progress=False)
    assert isinstance(provision, SMPLMode1Provision)
    assert provision.runtime_mode == "SMPLJAXModel"
    assert provision.params["betas"].shape == (1, 10)
    assert provision.model_path is None


def test_mode1_exports_artifacts(tmp_path: Path) -> None:
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=4, step_size=0.2, diagnostics_every=2)
    artifacts = export_mode1_artifacts(tmp_path, result, prefix="toy")
    payload = json.loads(artifacts["viewer_payload"].read_text(encoding="utf-8"))

    assert artifacts["snapshot"].exists()
    assert artifacts["mode1_snapshot"].exists()
    assert artifacts["metrics"].exists()
    assert artifacts["history"].exists()
    assert artifacts["history_json"].exists()
    assert artifacts["history_csv"].exists()
    assert payload["schema_name"] == "smpljax.mode1.visualization"
    assert payload["metrics"]["n_steps"] == 4
    snapshot = load_mode1_snapshot(artifacts["mode1_snapshot"])
    assert snapshot.schema_name == "smpljax.mode1.snapshot"
    assert snapshot.schema_version == "1.0"
    assert snapshot.faces.shape == (2, 3)


def test_mode1_workflow_runs_end_to_end(tmp_path: Path) -> None:
    provision = initialize_mode1_model(model=_toy_model(), params=_mode1_params(), progress=False)
    run = run_mode1_workflow(
        provision,
        output_dir=tmp_path,
        prefix="workflow",
        steps=4,
        step_size=0.2,
        diagnostics_every=2,
        progress=False,
    )
    assert run.artifacts["metrics"].exists()
    assert run.artifacts["history_json"].exists()
    assert run.result.output.vertices.shape[0] == 1


def test_mode1_visualization_payload_is_stable() -> None:
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=3, step_size=0.2, diagnostics_every=1)
    payload = build_mode1_visualization_payload(result)

    assert payload["schema_name"] == "smpljax.mode1.visualization"
    assert len(payload["vertices"]) == 4
    assert len(payload["joints"]) == 4


def test_mode1_matplotlib_plot() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=2, step_size=0.2, diagnostics_every=1)
    fig = plot_mode1_matplotlib(result, faces=np.asarray(model.data.faces_tensor))
    assert fig is not None
    fig.canvas.draw()


def test_mode1_pyvista_plot_with_fake_backend(monkeypatch) -> None:
    class _FakePolyData:
        def __init__(self, points, faces=None):
            self.points = np.asarray(points)
            self.faces = None if faces is None else np.asarray(faces)
            self.lines = None

    class _FakePlotter:
        def __init__(self, off_screen):
            self.off_screen = off_screen
            self.meshes = []
            self.points = []
            self.axes = False
            self.background = None
            self.rendered = False

        def add_mesh(self, dataset, **kwargs):
            self.meshes.append((dataset, kwargs))

        def add_points(self, points, **kwargs):
            self.points.append((points, kwargs))

        def add_title(self, title):
            self.title = title

        def add_axes(self):
            self.axes = True

        def set_background(self, value):
            self.background = value

        def reset_camera(self):
            self.camera_reset = True

        def render(self):
            self.rendered = True

        def show(self):
            return None

    monkeypatch.setitem(sys.modules, "pyvista", types.SimpleNamespace(PolyData=_FakePolyData, Plotter=_FakePlotter))
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=2, step_size=0.2, diagnostics_every=1)
    plotter = plot_mode1_pyvista(
        result,
        faces=np.asarray(model.data.faces_tensor),
        parents=np.asarray(model.data.parents),
        show=False,
    )
    assert plotter.off_screen is True
    assert len(plotter.meshes) >= 1
    assert len(plotter.points) == 1
    assert plotter.axes is True
    assert plotter.background == "white"
    assert plotter.rendered is True


def test_mode1_viser_plot_with_fake_backend(monkeypatch) -> None:
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
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=2, step_size=0.2, diagnostics_every=1)
    server = plot_mode1_viser(
        result,
        faces=np.asarray(model.data.faces_tensor),
        parents=np.asarray(model.data.parents),
    )
    assert server.host == "127.0.0.1"
    assert server.port == 8090
    assert server.scene.label[1] == "SMPL Mode 1 Result"
    assert server.scene.lines[0] == "/smpl/mode1_skeleton"


def test_mode1_visualize_result_dispatch(monkeypatch) -> None:
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
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=2, step_size=0.2, diagnostics_every=1)
    server = visualize_mode1_result(result, backend="viser")
    assert server.host == "127.0.0.1"
    assert server.scene.lines[0] == "/smpl/mode1_skeleton"


def test_mode2_payload_helpers_handle_incomplete_result(tmp_path: Path) -> None:
    result = SMPLMode2OptimizationResult(
        implementation_status="incomplete-result",
        summary="Mode 2 result without geometry payloads",
        metadata={"mode": 2},
    )
    metrics = mode2_metrics_payload(result)
    viewer = build_mode2_visualization_payload(result)
    artifacts = export_mode2_artifacts(tmp_path, result, prefix="mode2")

    assert metrics["schema_name"] == "smpljax.mode2.metrics"
    assert viewer["schema_name"] == "smpljax.mode2.visualization"
    assert artifacts["metrics"].exists()
    assert artifacts["viewer_payload"].exists()
    try:
        visualize_mode2_result(result, backend="pyvista")
    except NotImplementedError as exc:
        assert "requires geometry payloads" in str(exc) or "requires objective and gradient history" in str(exc)
    else:
        raise AssertionError("Expected incomplete Mode 2 visualization to raise NotImplementedError")


def test_mode2_staged_optimization_reduces_objective_and_exports_artifacts(tmp_path: Path) -> None:
    model = _toy_model()
    result = optimize_mode2(
        model,
        _mode1_params(),
        stages=(
            SMPLMode2StageConfig(name="transl_only", steps=4, step_size=0.2, trainable_keys=("transl",)),
            SMPLMode2StageConfig(name="full_refine", steps=4, step_size=0.1),
        ),
        diagnostics_every=2,
    )
    metrics = mode2_metrics_payload(result)
    history = mode2_history_payload(result)
    artifacts = export_mode2_artifacts(tmp_path, result, prefix="mode2_real")
    phases = json.loads(artifacts["phases"].read_text(encoding="utf-8"))

    assert result.implementation_status == "implemented-staged-workflow"
    assert result.output is not None
    assert result.objective_history is not None
    assert float(result.objective_history[-1]) <= float(result.objective_history[0]) + 1.0e-8
    assert metrics["n_stages"] == 2
    assert metrics["n_steps"] == 8
    assert history["step"].tolist() == list(range(1, 9))
    assert artifacts["history"].exists()
    assert artifacts["history_json"].exists()
    assert artifacts["history_csv"].exists()
    assert len(phases) == 2
    assert phases[0]["stage_name"] == "transl_only"


def test_mode2_visualization_dispatch_for_implemented_result(monkeypatch) -> None:
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
    result = optimize_mode2(_toy_model(), _mode1_params(), diagnostics_every=1)
    server = visualize_mode2_result(result, backend="viser")
    assert server.host == "127.0.0.1"
    assert server.scene.lines[0] == "/smpl/mode1_skeleton"

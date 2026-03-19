import json
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from smpljax import (
    SMPLJAXModel,
    SMPLMode2OptimizationResult,
    build_mode1_visualization_payload,
    build_mode2_visualization_payload,
    export_mode1_artifacts,
    export_mode2_artifacts,
    load_mode1_snapshot,
    mode1_history_payload,
    mode1_metrics_payload,
    mode2_metrics_payload,
    optimize_mode1,
    plot_mode1_matplotlib,
    plot_mode1_pyvista,
    plot_mode1_viser,
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


def test_mode1_exports_artifacts(tmp_path: Path) -> None:
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=4, step_size=0.2, diagnostics_every=2)
    artifacts = export_mode1_artifacts(tmp_path, result, prefix="toy")
    payload = json.loads(artifacts["viewer_payload"].read_text(encoding="utf-8"))

    assert artifacts["snapshot"].exists()
    assert artifacts["mode1_snapshot"].exists()
    assert artifacts["metrics"].exists()
    assert artifacts["history"].exists()
    assert artifacts["history_csv"].exists()
    assert payload["schema_name"] == "smpljax.mode1.visualization"
    assert payload["metrics"]["n_steps"] == 4
    snapshot = load_mode1_snapshot(artifacts["mode1_snapshot"])
    assert snapshot.schema_name == "smpljax.mode1.snapshot"
    assert snapshot.schema_version == "1.0"
    assert snapshot.faces.shape == (2, 3)


def test_mode1_visualization_payload_is_stable() -> None:
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=3, step_size=0.2, diagnostics_every=1)
    payload = build_mode1_visualization_payload(result)

    assert payload["schema_name"] == "smpljax.mode1.visualization"
    assert len(payload["vertices"]) == 4
    assert len(payload["joints"]) == 4


def test_mode1_matplotlib_plot() -> None:
    import matplotlib

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

        def add_mesh(self, dataset, **kwargs):
            self.meshes.append((dataset, kwargs))

        def add_points(self, points, **kwargs):
            self.points.append((points, kwargs))

        def add_title(self, title):
            self.title = title

        def show(self):
            return None

    monkeypatch.setitem(sys.modules, "pyvista", types.SimpleNamespace(PolyData=_FakePolyData, Plotter=_FakePlotter))
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=2, step_size=0.2, diagnostics_every=1)
    plotter = plot_mode1_pyvista(result, faces=np.asarray(model.data.faces_tensor), parents=np.asarray(model.data.parents), show=False)
    assert plotter.off_screen is True
    assert len(plotter.meshes) >= 1
    assert len(plotter.points) == 1


def test_mode1_viser_plot_with_fake_backend(monkeypatch) -> None:
    class _FakeScene:
        def set_up_direction(self, direction):
            self.direction = direction

        def add_grid(self, path, plane="xz"):
            self.grid = (path, plane)

        def add_mesh_simple(self, path, *, vertices, faces, color):
            self.mesh = (path, vertices, faces, color)

        def add_point_cloud(self, path, *, points, colors, point_size):
            self.points = (path, points, colors, point_size)

    class _FakeServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.scene = _FakeScene()

    monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=_FakeServer))
    model = _toy_model()
    result = optimize_mode1(model, _mode1_params(), steps=2, step_size=0.2, diagnostics_every=1)
    server = plot_mode1_viser(result, faces=np.asarray(model.data.faces_tensor))
    assert server.host == "127.0.0.1"
    assert server.port == 8090


def test_mode1_visualize_result_dispatch(monkeypatch) -> None:
    class _FakeScene:
        def set_up_direction(self, direction):
            self.direction = direction

        def add_grid(self, path, plane="xz"):
            self.grid = (path, plane)

        def add_mesh_simple(self, path, *, vertices, faces, color):
            self.mesh = (path, vertices, faces, color)

        def add_point_cloud(self, path, *, points, colors, point_size):
            self.points = (path, points, colors, point_size)

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


def test_mode2_stub_contracts(tmp_path: Path) -> None:
    result = SMPLMode2OptimizationResult(
        implementation_status="stubbed-interface",
        summary="Mode 2 contract stub",
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
        assert "not implemented yet" in str(exc)
    else:
        raise AssertionError("Expected Mode 2 visualization stub to raise NotImplementedError")

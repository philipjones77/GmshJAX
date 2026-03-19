import csv
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import jax.numpy as jnp

from topojax.ad.mode1 import export_mode1_artifacts, mode1_history_payload, mode1_metrics_payload, optimize_mode1_fixed_topology
from topojax.io.exports import GmshElementBlock
from topojax.io.topo_snapshot import TOPO_SNAPSHOT_SCHEMA, TOPO_SNAPSHOT_VERSION, load_topo_snapshot
from topojax.mesh.topology import unit_square_tri_mesh
from topojax.visualization import build_mode1_visualization_payload, export_mode1_visualization_payload, visualize_mode1_result


def _distort(points: jnp.ndarray) -> jnp.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return points.at[:, 1].set(y + 0.08 * jnp.sin(2.0 * jnp.pi * x) * x * (1.0 - x))


def test_mode1_metrics_and_history_payload_have_stable_schema() -> None:
    topo, points = unit_square_tri_mesh(8, 6)
    result = optimize_mode1_fixed_topology(_distort(points), topo, steps=6, step_size=0.02, diagnostics_every=2)

    metrics = mode1_metrics_payload(result)
    history = mode1_history_payload(result)

    assert metrics["schema_name"] == "topojax.mode1.metrics"
    assert metrics["schema_version"] == "1.0"
    assert metrics["n_steps"] == 6
    assert metrics["status"] in {"converged", "improving", "stalled"}
    assert history["step"].tolist() == [1, 2, 3, 4, 5, 6]
    assert history["energy_history"].shape == (6,)
    assert history["grad_norm_history"].shape == (6,)


def test_mode1_exports_native_snapshot_and_visualization_payload(tmp_path: Path) -> None:
    topo, points = unit_square_tri_mesh(8, 6)
    result = optimize_mode1_fixed_topology(_distort(points), topo, steps=8, step_size=0.02, diagnostics_every=4)

    boundary_block = GmshElementBlock(
        elements=jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32),
        element_kind="line",
        physical_tags=jnp.asarray([7, 7], dtype=jnp.int32),
        geometrical_tags=jnp.asarray([70, 70], dtype=jnp.int32),
    )
    artifacts = export_mode1_artifacts(
        tmp_path,
        result,
        prefix="tri",
        extra_element_blocks=(boundary_block,),
        physical_names={(1, 7): "outer_boundary"},
    )

    assert artifacts["topo_snapshot"].exists()
    assert artifacts["viewer_payload"].exists()
    assert artifacts["history_json"].exists()
    assert artifacts["history_csv"].exists()

    snapshot = load_topo_snapshot(artifacts["topo_snapshot"])
    assert snapshot.schema_name == TOPO_SNAPSHOT_SCHEMA
    assert snapshot.schema_version == TOPO_SNAPSHOT_VERSION
    assert snapshot.topology.n_nodes == topo.n_nodes
    assert snapshot.metrics["n_steps"] == 8
    assert "icn" in snapshot.element_fields
    assert "area" in snapshot.element_fields
    assert snapshot.history["step"].tolist() == list(range(1, 9))
    assert snapshot.physical_names[(1, 7)] == "outer_boundary"
    assert len(snapshot.extra_element_blocks) == 1
    assert snapshot.extra_element_blocks[0].element_kind == "line"
    assert snapshot.visualization_payload["topology_kind"] == "triangle"

    history_rows = json.loads(artifacts["history_json"].read_text(encoding="utf-8"))
    assert history_rows
    with artifacts["history_csv"].open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "energy" in rows[0]
    assert "grad_norm" in rows[0]


def test_visualization_payload_export_and_dispatch(tmp_path: Path, monkeypatch) -> None:
    topo, points = unit_square_tri_mesh(6, 5)
    result = optimize_mode1_fixed_topology(_distort(points), topo, steps=4, step_size=0.02, diagnostics_every=2)
    metrics = mode1_metrics_payload(result)

    payload = build_mode1_visualization_payload(result.points, result.topology, title="Mode 1", metrics=metrics)
    payload_path = export_mode1_visualization_payload(tmp_path / "viewer.json", result.points, result.topology, title="Mode 1", metrics=metrics)
    exported = json.loads(payload_path.read_text(encoding="utf-8"))

    assert payload["topology_kind"] == "triangle"
    assert exported["metrics"]["n_steps"] == 4
    class _FakeScene:
        def set_up_direction(self, direction):
            self.direction = direction

        def add_grid(self, path, plane="xy"):
            self.grid = (path, plane)

        def add_mesh_simple(self, path, *, vertices, faces, color):
            self.mesh = (path, vertices, faces, color)

        def add_point_cloud(self, path, *, points, colors, point_size):
            self.cloud = (path, points, colors, point_size)

        def add_line_segments(self, path, *, points, colors):
            self.lines = (path, points, colors)

    class _FakeServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.scene = _FakeScene()

    monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=_FakeServer))
    server = visualize_mode1_result(result, backend="viser", title="Mode 1")
    assert server.host == "127.0.0.1"
    assert hasattr(server.scene, "cloud")
    assert hasattr(server.scene, "lines")


def test_visualization_result_dispatches_native_gmsh(monkeypatch, tmp_path: Path) -> None:
    topo, points = unit_square_tri_mesh(5, 4)
    result = optimize_mode1_fixed_topology(_distort(points), topo, steps=3, step_size=0.02, diagnostics_every=1)
    captured: dict[str, object] = {}

    def _fake_launch(mesh_path, *, gmsh_executable, wait, extra_args):
        captured["mesh_path"] = Path(mesh_path)
        captured["gmsh_executable"] = gmsh_executable
        captured["wait"] = wait
        captured["extra_args"] = extra_args
        return "gmsh-ok"

    monkeypatch.setattr("topojax._visualization_backends.gmsh_backend.launch_gmsh_viewer", _fake_launch)
    rv = visualize_mode1_result(
        result,
        mesh_path=tmp_path / "mode1_view.msh",
        gmsh_executable="gmsh",
        gmsh_extra_args=["-nopopup"],
        wait=True,
    )
    assert rv == "gmsh-ok"
    assert captured["mesh_path"] == tmp_path / "mode1_view.msh"
    assert captured["gmsh_executable"] == "gmsh"
    assert captured["wait"] is True
    assert captured["extra_args"] == ["-nopopup"]
    assert (tmp_path / "mode1_view.msh").exists()


def test_mode1_artifact_benchmark_script_emits_expected_fields(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_json = tmp_path / "bench.json"
    out_dir = tmp_path / "artifacts"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    proc = subprocess.run(
        [
            sys.executable,
            "benchmarks/topo/benchmark_mode1_artifacts.py",
            "--steps",
            "4",
            "--out-dir",
            str(out_dir),
            "--out",
            str(out_json),
        ],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["steps"] == 4
    assert payload["export_ms"] > 0.0
    assert payload["topo_snapshot_bytes"] > 0
    assert payload["viewer_payload_bytes"] > 0


def test_real_pyvista_backend_smoke_subprocess() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYVISTA_OFF_SCREEN"] = "true"
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "from topojax.mesh.topology import unit_square_tri_mesh; "
                "from topojax.visualization import plot_mode1_pyvista; "
                "topo, pts = unit_square_tri_mesh(4,3); "
                "print('before_plotter'); "
                "plotter = plot_mode1_pyvista(pts, topo, show=False); "
                "print('after_plotter', type(plotter).__name__); "
                "plotter.close(); "
                "print('closed')"
            ),
        ],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert "before_plotter" in proc.stdout
    assert "after_plotter Plotter" in proc.stdout
    assert "closed" in proc.stdout

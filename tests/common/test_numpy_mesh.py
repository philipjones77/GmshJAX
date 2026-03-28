from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from topojax.ad.modes import MeshMovementMode
from topojax.mesh.topology import unit_square_tri_mesh

from common import diagnostics_payload, numpy_mesh


def _tri_runtime(**kwargs):
    topology, points = unit_square_tri_mesh(4, 3)
    return numpy_mesh.create_runtime(points, topology.elements, **kwargs)


def test_numpy_mesh_runtime_diagnostics_save_and_load(tmp_path: Path) -> None:
    runtime = _tri_runtime(mode=MeshMovementMode.FIXED_TOPOLOGY, metadata={"source": "unit-square"})
    payload = diagnostics_payload(runtime=runtime)
    artifacts = numpy_mesh.export_mode1_artifacts(tmp_path, runtime, prefix="npmesh")
    loaded = numpy_mesh.load_runtime(artifacts["mesh"])
    metrics = json.loads(artifacts["metrics"].read_text(encoding="utf-8"))

    assert payload["runtime"]["n_nodes"] == 12
    assert payload["runtime"]["n_elements"] == 12
    assert metrics["schema_name"] == "common.numpy_mesh.metrics"
    assert artifacts["viewer_payload"].exists()
    assert artifacts["rf77_summary"].exists()
    assert loaded.mode == MeshMovementMode.FIXED_TOPOLOGY
    assert loaded.diagnostics().n_edges == runtime.diagnostics().n_edges


def test_numpy_mesh_rf77_bridges_cover_all_modes() -> None:
    topology, points = unit_square_tri_mesh(4, 3)
    runtimes = (
        numpy_mesh.create_mode1_runtime(points, topology.elements, metadata={"tag": "m1"}),
        numpy_mesh.create_mode2_runtime(points, topology.elements, restart_phases=[{"cycle": 0, "reason": "seed"}]),
        numpy_mesh.create_mode3_runtime(
            points,
            topology.elements,
            candidate_graph={"kind": "triangle-edge-candidates"},
            soft_weights=[[0.7, 0.3]],
            candidate_logits=[0.1],
        ),
        numpy_mesh.create_mode4_runtime(
            points,
            topology.elements,
            candidate_graph={"kind": "triangle-edge-candidates"},
            forward_state={"hard_weights": [[1.0, 0.0]]},
            backward_surrogate={"temperature": 0.25},
            candidate_logits=[0.2],
        ),
        numpy_mesh.create_mode5_runtime(
            points,
            topology.elements,
            controller_history=[{"cycle": 0, "reason": "bootstrap"}],
            transfer_history=[{"cycle": 0, "transferred": True}],
        ),
    )

    for runtime in runtimes:
        bridge = numpy_mesh.build_mode_bridge(runtime)
        payload = bridge.to_randomfields77_mesh_payload()

        assert bridge.mode == runtime.mode
        assert payload["mesh_storage"]["mesh_builder_options"]["backend"] == "numpy"
        assert payload["nodes"].shape[0] == points.shape[0]
        assert payload["cells"].shape == topology.elements.shape


def test_numpy_mesh_visualization_payload_and_viser_dispatch(monkeypatch) -> None:
    class _FakeScene:
        def __init__(self):
            self.meshes = []
            self.lines = []
            self.clouds = []

        def set_up_direction(self, direction):
            self.direction = direction

        def add_grid(self, path, plane="xy"):
            self.grid = (path, plane)

        def add_mesh_simple(self, path, *, vertices, faces, color):
            self.meshes.append((path, vertices, faces, color))

        def add_point_cloud(self, path, *, points, colors, point_size):
            self.clouds.append((path, points, colors, point_size))

        def add_line_segments(self, path, *, points, colors):
            self.lines.append((path, points, colors))

    class _FakeServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.scene = _FakeScene()

    monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=_FakeServer))
    runtime = _tri_runtime(
        mode=MeshMovementMode.FULLY_DYNAMIC,
        mode_payload={
            "controller_history": [{"cycle": 0, "reason": "bootstrap"}],
            "transfer_history": [{"cycle": 0, "transferred": True}],
            "implementation_status": "implemented-numpy",
        },
    )
    payload = numpy_mesh.build_visualization_payload(runtime)
    server = numpy_mesh.visualize_runtime(runtime, backend="viser")

    assert payload["schema_name"] == "topojax.visualization.payload"
    assert payload["mode"] == 5
    assert payload["implementation_status"] == "implemented-numpy"
    assert server.host == "127.0.0.1"
    assert server.port == 8081
    assert server.scene.meshes
    assert server.scene.clouds
    assert server.scene.lines

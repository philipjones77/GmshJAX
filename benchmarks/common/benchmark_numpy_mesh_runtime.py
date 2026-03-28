from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from common import numpy_mesh
from topojax.mesh.topology import unit_square_tri_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the common NumPy mesh runtime surface.")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--mode", choices=("1", "3", "5"), default="1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    topology, points = unit_square_tri_mesh(args.nx, args.ny)
    if args.mode == "1":
        runtime = numpy_mesh.create_mode1_runtime(points, topology.elements, metadata={"source": "benchmark"})
    elif args.mode == "3":
        runtime = numpy_mesh.create_mode3_runtime(
            points,
            topology.elements,
            candidate_graph={"kind": "triangle-edge-candidates", "candidate_count": 1},
            soft_weights=[[0.7, 0.3]],
            candidate_logits=[0.1],
            metadata={"source": "benchmark"},
        )
    else:
        runtime = numpy_mesh.create_mode5_runtime(
            points,
            topology.elements,
            controller_history=[{"cycle": 0, "reason": "benchmark"}],
            transfer_history=[{"cycle": 0, "transferred": True}],
            metadata={"source": "benchmark"},
        )

    t0 = time.perf_counter()
    diagnostics = runtime.diagnostics()
    bridge = numpy_mesh.build_mode_bridge(runtime)
    artifacts = numpy_mesh.export_mode_artifacts(args.out_dir, runtime, prefix=f"numpy_mode{args.mode}")
    payload = bridge.to_randomfields77_mesh_payload()
    runtime_s = time.perf_counter() - t0

    summary = {
        "benchmark": "common_numpy_mesh_runtime",
        "api_namespace": "common.numpy_mesh",
        "mode": int(args.mode),
        "n_nodes": diagnostics.n_nodes,
        "n_elements": diagnostics.n_elements,
        "n_edges": diagnostics.n_edges,
        "runtime_s": runtime_s,
        "rf77_cells": int(payload["cells"].shape[0]),
        "viewer_payload_path": str(artifacts["viewer_payload"]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

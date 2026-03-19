"""Benchmark Mode 1 artifact export, including native Topo snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from topojax.ad.mode1 import export_mode1_artifacts, optimize_mode1_fixed_topology
from topojax.mesh.topology import unit_square_tri_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mode 1 artifact export.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    topo, points = unit_square_tri_mesh(24, 18)
    result = optimize_mode1_fixed_topology(points, topo, steps=args.steps, step_size=0.02, diagnostics_every=4)

    t0 = perf_counter()
    artifacts = export_mode1_artifacts(args.out_dir, result, prefix="mode1_bench", export_stl_surface=True)
    elapsed_ms = (perf_counter() - t0) * 1.0e3
    payload = {
        "steps": args.steps,
        "export_ms": elapsed_ms,
        "topo_snapshot_bytes": artifacts["topo_snapshot"].stat().st_size,
        "viewer_payload_bytes": artifacts["viewer_payload"].stat().st_size,
        "history_csv_bytes": artifacts["history_csv"].stat().st_size,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for key, value in payload.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

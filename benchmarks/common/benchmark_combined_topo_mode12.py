"""Benchmark common-track Topo Mode 1 and Mode 2 workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from common import topo as topo_api


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark common-track Topo workflows.")
    parser.add_argument("--mode", choices=["1", "2"], default="1")
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = perf_counter()
    if args.mode == "1":
        domain = topo_api.initialize_mode1_domain("square", family="tri", nx=args.nx, ny=args.ny, progress=False)
        init_s = perf_counter() - t0
        t1 = perf_counter()
        run = topo_api.run_mode1_workflow(
            domain,
            output_dir=args.out_dir,
            prefix="bench_mode1",
            steps=args.steps,
            step_size=0.02,
            diagnostics_every=max(1, args.steps // 2),
            progress=False,
        )
        workflow_s = perf_counter() - t1
        payload = {
            "benchmark": "combined_topo_mode1",
            "api_namespace": "common.topo",
            "mode": 1,
            "init_s": init_s,
            "workflow_s": workflow_s,
            "runtime_s": init_s + workflow_s,
            "n_nodes": int(run.result.points.shape[0]),
            "n_elements": int(run.result.topology.elements.shape[0]),
            "n_steps": int(run.result.energy_history.shape[0]),
            "final_energy": float(run.result.energy_history[-1]),
            "final_grad_norm": float(run.result.grad_norm_history[-1]),
        }
    else:
        domain = topo_api.initialize_mode2_domain("square", family="tri", nx=args.nx, ny=args.ny, progress=False)
        init_s = perf_counter() - t0
        t1 = perf_counter()
        run = topo_api.run_mode2_workflow(
            domain,
            output_dir=args.out_dir,
            prefix="bench_mode2",
            cycles=args.cycles,
            optimization_steps=args.steps,
            optimization_step_size=0.02,
            remesh_max_iters=1,
            progress=False,
        )
        workflow_s = perf_counter() - t1
        payload = {
            "benchmark": "combined_topo_mode2",
            "api_namespace": "common.topo",
            "mode": 2,
            "init_s": init_s,
            "workflow_s": workflow_s,
            "runtime_s": init_s + workflow_s,
            "n_nodes": int(run.result.points.shape[0]),
            "n_elements": int(run.result.elements.shape[0]),
            "n_phases": len(run.result.phases),
            "remesh_count": sum(1 for phase in run.result.phases if phase.remeshed),
        }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    for key, value in payload.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

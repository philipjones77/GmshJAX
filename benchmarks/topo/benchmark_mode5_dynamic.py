"""Benchmark Mode 5 relaxed dynamic workflow for triangle and tetra meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from topojax.ad.workflow import initialize_mode5_domain, run_mode5_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TopoJAX Mode 5 relaxed dynamic workflow.")
    parser.add_argument("--kind", choices=["tri", "tet"], default="tri")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--optimization-steps", type=int, default=4)
    parser.add_argument("--surrogate-steps", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.kind == "tri":
        domain = initialize_mode5_domain("square", family="tri", nx=12, ny=10, progress=False)
    else:
        domain = initialize_mode5_domain("box", nx=4, ny=4, nz=4, progress=False)

    t0 = perf_counter()
    run = run_mode5_workflow(
        domain,
        output_dir=args.out_dir,
        prefix=f"bench_{args.kind}",
        cycles=args.cycles,
        optimization_steps=args.optimization_steps,
        surrogate_steps=args.surrogate_steps,
        optimization_step_size=0.02 if args.kind == "tri" else 0.012,
        surrogate_step_size=0.1,
        remesh_max_iters=1,
        progress=False,
    )
    elapsed_s = perf_counter() - t0

    payload = {
        "benchmark": "mode5_dynamic",
        "kind": args.kind,
        "cycles": args.cycles,
        "optimization_steps": args.optimization_steps,
        "surrogate_steps": args.surrogate_steps,
        "runtime_s": elapsed_s,
        "implementation_status": run.result.implementation_status,
        "n_nodes": int(run.result.points.shape[0]),
        "n_elements": int(run.result.elements.shape[0]),
        "remesh_count": sum(1 for phase in run.result.phases if phase.remeshed),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for key, value in payload.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

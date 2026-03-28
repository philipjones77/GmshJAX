"""Benchmark Mode 2 remesh-restart workflow for TopoJAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

from topojax.ad.workflow import initialize_mode2_domain, run_mode2_restart_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TopoJAX Mode 2 restart workflow.")
    parser.add_argument("--kind", choices=["tri", "quad", "tet"], default="tri")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--optimization-steps", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.kind == "tri":
        outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
        hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
        domain = initialize_mode2_domain("polygon", outer_boundary=outer, holes=[hole], target_edge_size=0.18, progress=False)
    elif args.kind == "quad":
        outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
        hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
        domain = initialize_mode2_domain("polygon-quad", outer_boundary=outer, holes=[hole], target_edge_size=0.2, progress=False)
    else:
        domain = initialize_mode2_domain(
            "sphere-volume",
            center=jnp.asarray([0.5, 0.5, 0.5]),
            radius=0.42,
            nx=7,
            ny=7,
            nz=7,
            progress=False,
        )

    t0 = perf_counter()
    run = run_mode2_restart_workflow(
        domain,
        output_dir=args.out_dir,
        prefix=f"bench_{args.kind}",
        cycles=args.cycles,
        optimization_steps=args.optimization_steps,
        optimization_step_size=0.02 if args.kind != "tet" else 0.012,
        remesh_max_iters=1,
        progress=False,
    )
    elapsed_s = perf_counter() - t0

    payload = {
        "benchmark": "mode2_restart",
        "kind": args.kind,
        "cycles": args.cycles,
        "optimization_steps": args.optimization_steps,
        "runtime_s": elapsed_s,
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

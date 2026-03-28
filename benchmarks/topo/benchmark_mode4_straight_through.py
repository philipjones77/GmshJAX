"""Benchmark Mode 4 straight-through connectivity optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

from topojax.ad.straight_through import optimize_straight_through_connectivity, summarize_mode4_result
from topojax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh


def _distort(points: jnp.ndarray) -> jnp.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return points.at[:, 1].set(y + 0.08 * jnp.sin(2.0 * jnp.pi * x) * x * (1.0 - x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TopoJAX Mode 4 straight-through optimization.")
    parser.add_argument("--kind", choices=["tri", "quad", "tet"], default="tri")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.kind == "tri":
        topo, points = unit_square_tri_mesh(24, 18)
        points = _distort(points)
    elif args.kind == "quad":
        topo, points = unit_square_quad_mesh(24, 18)
        points = _distort(points)
    else:
        topo, points = unit_cube_tet_mesh(6, 6, 5)
        points = points.at[:, 2].set(points[:, 2] + 0.05 * points[:, 0] * (1.0 - points[:, 0]))

    t0 = perf_counter()
    result = optimize_straight_through_connectivity(points, topo.elements, steps=args.steps, step_size=0.1, temperature=0.25)
    elapsed_s = perf_counter() - t0
    payload = {
        "benchmark": "mode4_straight_through",
        "kind": args.kind,
        "runtime_s": elapsed_s,
        **summarize_mode4_result(result),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for key, value in payload.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

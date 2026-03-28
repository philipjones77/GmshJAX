from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax.numpy as jnp

from smpljax.body_models import SMPLJAXModel
from smpljax.mode1 import default_mode1_params
from smpljax.mode2 import SMPLMode2StageConfig, optimize_mode2
from smpljax.utils import SMPLModelData


def _build_synthetic_model(
    *,
    num_verts: int,
    num_joints: int,
    num_betas: int,
    dtype: jnp.dtype,
) -> SMPLJAXModel:
    body_joints = max(num_joints - 1, 0)
    return SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((num_verts, 3), dtype=dtype),
            shapedirs=jnp.zeros((num_verts, 3, num_betas), dtype=dtype),
            posedirs=jnp.zeros((body_joints * 9, num_verts * 3), dtype=dtype),
            j_regressor=jnp.ones((num_joints, num_verts), dtype=dtype) / max(num_verts, 1),
            parents=jnp.array([-1] + list(range(num_joints - 1)), dtype=jnp.int32),
            lbs_weights=jnp.ones((num_verts, num_joints), dtype=dtype) / max(num_joints, 1),
            num_betas=num_betas,
            num_body_joints=body_joints,
            faces_tensor=jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark staged SMPLJAX Mode 2 optimization.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-verts", type=int, default=8)
    parser.add_argument("--num-joints", type=int, default=4)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    model = _build_synthetic_model(
        num_verts=args.num_verts,
        num_joints=args.num_joints,
        num_betas=args.num_betas,
        dtype=dtype,
    )
    params = default_mode1_params(model, batch_size=args.batch_size)
    params["transl"] = jnp.broadcast_to(jnp.asarray([[0.5, -0.25, 0.1]], dtype=dtype), (args.batch_size, 3))

    t0 = time.perf_counter()
    first = optimize_mode2(
        model,
        params,
        stages=(
            SMPLMode2StageConfig(name="translation_warmup", steps=2, step_size=0.05, trainable_keys=("transl",)),
            SMPLMode2StageConfig(name="full_refine", steps=4, step_size=0.01),
        ),
        diagnostics_every=2,
    )
    first_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.iters):
        result = optimize_mode2(
            model,
            params,
            stages=(
                SMPLMode2StageConfig(name="translation_warmup", steps=2, step_size=0.05, trainable_keys=("transl",)),
                SMPLMode2StageConfig(name="full_refine", steps=4, step_size=0.01),
            ),
            diagnostics_every=2,
        )
        assert result.objective_history is not None
    avg_runtime_s = (time.perf_counter() - t0) / max(args.iters, 1)

    payload = {
        "benchmark": "mode2",
        "runtime": "staged",
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "num_verts": args.num_verts,
        "num_joints": args.num_joints,
        "num_betas": args.num_betas,
        "iters": args.iters,
        "first_run_s": first_time_s,
        "avg_runtime_s": avg_runtime_s,
        "n_stages": len(first.phase_summaries),
        "n_steps": int(first.objective_history.shape[0]) if first.objective_history is not None else 0,
        "implementation_status": first.implementation_status,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps([payload], indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

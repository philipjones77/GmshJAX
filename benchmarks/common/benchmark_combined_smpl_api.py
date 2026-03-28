"""Benchmark common-track SMPL optimized runtime and workflow APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from common import smpl as smpl_api


def _write_toy_model(path: Path) -> None:
    np.savez(
        path,
        v_template=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        shapedirs=np.zeros((4, 3, 2), dtype=np.float32),
        posedirs=np.zeros((9, 12), dtype=np.float32),
        J_regressor=np.ones((2, 4), dtype=np.float32) / 4.0,
        weights=np.ones((4, 2), dtype=np.float32) / 2.0,
        parents=np.asarray([-1, 0], dtype=np.int32),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark common-track SMPL APIs.")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.output_json.parent / "combined_smpl_toy_model.npz"
    _write_toy_model(model_path)

    smpl_api.clear_io_cache()

    t0 = perf_counter()
    runtime = smpl_api.create_runtime(model_path, mode="optimized")
    create_first_s = perf_counter() - t0
    diag_after_first = smpl_api.io_cache_diagnostics()

    t0 = perf_counter()
    _ = smpl_api.create_runtime(model_path, mode="optimized")
    create_second_s = perf_counter() - t0
    diag_after_second = smpl_api.io_cache_diagnostics()

    inputs = runtime.prepare_inputs(batch_size=1, pose2rot=True)
    t0 = perf_counter()
    out = runtime.forward(inputs, pose2rot=True)
    _ = out.vertices.block_until_ready()
    compile_plus_first_forward_s = perf_counter() - t0

    t0 = perf_counter()
    for _ in range(args.iters):
        out = runtime.forward(inputs, pose2rot=True)
        _ = out.vertices.block_until_ready()
    steady_state_forward_s = (perf_counter() - t0) / max(args.iters, 1)

    params = smpl_api.default_mode1_params(runtime, batch_size=1)
    params["transl"] = params["transl"].at[0, 0].set(0.5)

    t0 = perf_counter()
    mode1_result = smpl_api.optimize_mode1(
        runtime,
        params,
        steps=3,
        step_size=0.1,
        diagnostics_every=1,
    )
    mode1_s = perf_counter() - t0

    t0 = perf_counter()
    mode2_result = smpl_api.optimize_mode2(runtime, params, diagnostics_every=1)
    mode2_s = perf_counter() - t0

    diagnostics = runtime.diagnostics()
    payload = {
        "benchmark": "combined_smpl_api",
        "api_namespace": "common.smpl",
        "runtime": "optimized",
        "iters": args.iters,
        "create_first_s": create_first_s,
        "create_second_s": create_second_s,
        "compile_plus_first_forward_s": compile_plus_first_forward_s,
        "steady_state_forward_s": steady_state_forward_s,
        "mode1_s": mode1_s,
        "mode2_s": mode2_s,
        "io_cache_entries": diag_after_second.entries,
        "io_cache_hits": diag_after_second.hits,
        "io_cache_misses": diag_after_second.misses,
        "io_cache_entries_after_first": diag_after_first.entries,
        "compile_count": diagnostics.compile_count,
        "compiled_entries": diagnostics.compiled_entries,
        "mode1_final_objective": float(mode1_result.objective_history[-1]),
        "mode2_final_objective": float(mode2_result.objective_history[-1]) if mode2_result.objective_history is not None else 0.0,
        "mode2_n_stages": len(mode2_result.phase_summaries),
        "mode2_status": mode2_result.implementation_status,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps([payload], indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

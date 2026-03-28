from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smpljax import OptimizedSMPLJAX
from common import smpl as smpl_api
from common import topo as topo_api
from topojax.io.imports import load_gmsh_msh


def _write_smpl_model(path: Path) -> None:
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


def test_combined_topo_mode1_workflow_runs_end_to_end(tmp_path: Path) -> None:
    domain = topo_api.initialize_mode1_domain("line", n=6, progress=False)

    run = topo_api.run_mode1_workflow(
        domain,
        output_dir=tmp_path / "topo_mode1",
        prefix="line",
        steps=4,
        step_size=0.02,
        diagnostics_every=2,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"], primary_element_kind="line")
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "line"
    assert metrics["schema_name"] == "topojax.mode1.metrics"
    assert metrics["n_steps"] == 4
    assert run.artifacts["viewer_payload"].exists()


def test_combined_topo_mode2_workflow_runs_end_to_end(tmp_path: Path) -> None:
    domain = topo_api.initialize_mode2_domain("square", family="tri", nx=5, ny=4, progress=False)

    run = topo_api.run_mode2_workflow(
        domain,
        output_dir=tmp_path / "topo_mode2",
        prefix="square_tri",
        cycles=1,
        optimization_steps=4,
        optimization_step_size=0.02,
        remesh_max_iters=1,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    phases = json.loads(run.artifacts["phases"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "triangle"
    assert metrics["n_cycles"] == 1
    assert len(phases) == 1
    assert run.artifacts["remesh_history"].exists()


def test_combined_smpl_optimized_runtime_cache_and_mode12_paths(tmp_path: Path) -> None:
    model_path = tmp_path / "toy_model.npz"
    _write_smpl_model(model_path)

    smpl_api.clear_io_cache()
    runtime = smpl_api.create_runtime(model_path, mode="optimized")
    diag_after_first = smpl_api.io_cache_diagnostics()
    runtime_again = smpl_api.create_runtime(model_path, mode="optimized")
    diag_after_second = smpl_api.io_cache_diagnostics()

    assert isinstance(runtime, OptimizedSMPLJAX)
    assert isinstance(runtime_again, OptimizedSMPLJAX)
    assert diag_after_first.entries == 1
    assert diag_after_second.entries == 1
    assert diag_after_second.hits >= diag_after_first.hits + 1

    params = smpl_api.default_mode1_params(runtime, batch_size=1)
    params["transl"] = params["transl"].at[0, 0].set(0.5)

    mode1_result = smpl_api.optimize_mode1(
        runtime,
        params,
        steps=3,
        step_size=0.1,
        diagnostics_every=1,
    )
    mode2_result = smpl_api.optimize_mode2(runtime, params, diagnostics_every=1)

    assert mode1_result.objective_history.shape == (3,)
    assert float(mode1_result.objective_history[-1]) <= float(mode1_result.objective_history[0]) + 1.0e-8
    assert mode2_result.implementation_status == "implemented-staged-workflow"
    assert mode2_result.objective_history is not None
    assert int(mode2_result.objective_history.shape[0]) >= 1

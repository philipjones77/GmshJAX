import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from topojax.ad.workflow import (
    initialize_mode3_domain,
    initialize_mode4_domain,
    run_mode3_workflow,
    run_mode4_workflow,
)
from topojax.io.imports import load_gmsh_msh


def test_mode3_square_quad_workflow_exports_artifacts(tmp_path: Path) -> None:
    domain = initialize_mode3_domain("square", family="quad", nx=6, ny=5, progress=False)

    run = run_mode3_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode3_square",
        steps=8,
        step_size=0.15,
        temperature=0.3,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    viewer = json.loads(run.artifacts["viewer_payload"].read_text(encoding="utf-8"))
    history_rows = json.loads(run.artifacts["history_json"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "quad"
    assert viewer["mode"] == 3
    assert viewer["implementation_status"] == "implemented"
    assert len(viewer["metadata"]["candidate_logits"]) == domain.topology.elements.shape[0]
    assert metrics["n_quads"] == domain.topology.elements.shape[0]
    assert metrics["n_steps"] == 8
    assert metrics["final_objective"] <= metrics["initial_objective"] + 1.0e-8
    assert len(history_rows) == 8
    assert run.result.weights.shape == (domain.topology.elements.shape[0], 2)


def test_mode3_square_tri_workflow_exports_artifacts(tmp_path: Path) -> None:
    domain = initialize_mode3_domain("square", family="tri", nx=6, ny=5, progress=False)

    run = run_mode3_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode3_square_tri",
        steps=8,
        step_size=0.15,
        temperature=0.3,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    viewer = json.loads(run.artifacts["viewer_payload"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "triangle"
    assert viewer["mode"] == 3
    assert viewer["implementation_status"] == "implemented"
    assert metrics["topology_kind"] == "triangle"
    assert metrics["candidate_kind"] == "triangle-edge-flip"
    assert metrics["final_objective"] <= metrics["initial_objective"] + 1.0e-8
    assert run.result.topology_kind == "triangle"


def test_mode4_polygon_quad_workflow_preserves_boundary_metadata(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode4_domain(
        "polygon-quad",
        outer_boundary=outer,
        holes=[hole],
        target_edge_size=0.2,
        backend="native",
        progress=False,
    )

    run = run_mode4_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode4_polygon",
        steps=8,
        step_size=0.15,
        temperature=0.25,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    viewer = json.loads(run.artifacts["viewer_payload"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "quad"
    assert any(block.element_kind == "line" for block in imported.extra_element_blocks)
    assert imported.physical_names[(1, 100)] == "outer_boundary"
    assert viewer["mode"] == 4
    assert viewer["implementation_status"] == "implemented"
    assert metrics["n_quads"] == run.result.elements.shape[0]
    assert metrics["final_objective"] <= metrics["initial_objective"] + 1.0e-8
    assert jnp.allclose(run.result.hard_weights, jnp.round(run.result.hard_weights), atol=1.0e-6)


def test_mode4_box_tet_workflow_exports_artifacts(tmp_path: Path) -> None:
    domain = initialize_mode4_domain("box", nx=3, ny=3, nz=3, progress=False)

    run = run_mode4_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode4_box_tet",
        steps=8,
        step_size=0.12,
        temperature=0.25,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    viewer = json.loads(run.artifacts["viewer_payload"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "tetra"
    assert viewer["mode"] == 4
    assert viewer["implementation_status"] == "implemented"
    assert metrics["topology_kind"] == "tetra"
    assert metrics["candidate_kind"] == "tetra-split"
    assert metrics["final_objective"] <= metrics["initial_objective"] + 1.0e-8
    assert jnp.allclose(run.result.hard_weights, jnp.round(run.result.hard_weights), atol=1.0e-6)


def test_mode34_workflows_reject_unsupported_domains(tmp_path: Path) -> None:
    domain3 = initialize_mode3_domain("line", n=8, progress=False)
    domain4 = initialize_mode4_domain(
        "sphere-surface",
        center=jnp.asarray([0.0, 0.0, 0.0]),
        radius=1.0,
        n_lat=6,
        n_lon=10,
        progress=False,
    )

    with pytest.raises(ValueError, match="2D triangle, 2D quad, and 3D tetra domains only"):
        run_mode3_workflow(domain3, output_dir=tmp_path / "mode3_bad", progress=False)

    with pytest.raises(ValueError, match="2D triangle, 2D quad, and 3D tetra domains only"):
        run_mode4_workflow(domain4, output_dir=tmp_path / "mode4_bad", progress=False)

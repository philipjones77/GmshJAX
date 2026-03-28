import json
from pathlib import Path

import jax.numpy as jnp

from topojax.ad.workflow import initialize_mode5_domain, run_mode5_workflow
from topojax.io.imports import load_gmsh_msh


def test_mode5_triangle_workflow_exports_artifacts_and_transfer_state(tmp_path: Path) -> None:
    domain = initialize_mode5_domain("square", family="tri", nx=6, ny=5, progress=False)
    node_fields = {"temperature": domain.points[:, 0]}
    element_fields = {"density": jnp.linspace(0.0, 1.0, domain.topology.elements.shape[0], dtype=domain.points.dtype)}

    run = run_mode5_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode5_tri",
        cycles=2,
        optimization_steps=4,
        surrogate_steps=4,
        optimization_step_size=0.02,
        surrogate_step_size=0.12,
        remesh_max_iters=1,
        node_fields=node_fields,
        element_fields=element_fields,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    viewer = json.loads(run.artifacts["viewer_payload"].read_text(encoding="utf-8"))
    phases = json.loads(run.artifacts["phases"].read_text(encoding="utf-8"))
    controller = json.loads(run.artifacts["controller"].read_text(encoding="utf-8"))
    transfers = json.loads(run.artifacts["transfers"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "triangle"
    assert metrics["implementation_status"] == "implemented-relaxed-dynamic"
    assert metrics["topology_kind"] == "triangle"
    assert viewer["implementation_status"] == "implemented"
    assert len(phases) == 2
    assert len(controller) == 2
    assert len(transfers) == 2
    assert transfers[0]["transferred_node_fields"] == ["temperature"]
    assert transfers[0]["transferred_element_fields"] == ["density"]
    assert run.result.node_fields["temperature"].shape[0] == run.result.points.shape[0]
    assert run.result.element_fields["density"].shape[0] == run.result.elements.shape[0]


def test_mode5_tet_workflow_exports_artifacts(tmp_path: Path) -> None:
    domain = initialize_mode5_domain("box", nx=3, ny=3, nz=3, progress=False)
    node_fields = {"temperature": domain.points[:, 2]}

    run = run_mode5_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode5_tet",
        cycles=2,
        optimization_steps=4,
        surrogate_steps=4,
        optimization_step_size=0.012,
        surrogate_step_size=0.1,
        remesh_max_iters=1,
        node_fields=node_fields,
        progress=False,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    transfers = json.loads(run.artifacts["transfers"].read_text(encoding="utf-8"))

    assert imported.primary_element_kind == "tetra"
    assert metrics["topology_kind"] == "tetra"
    assert metrics["implementation_status"] == "implemented-relaxed-dynamic"
    assert len(run.result.phases) == 2
    assert len(transfers) == 2
    assert transfers[0]["transferred_node_fields"] == ["temperature"]

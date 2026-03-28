from pathlib import Path
import types

import jax.numpy as jnp

from topojax.ad.workflow import initialize_mode1_domain, run_mode1_workflow
from topojax.io.exports import GmshElementBlock, export_gmsh_msh
from topojax.io.imports import load_gmsh_msh


def test_mode1_line_workflow_create_export_import(tmp_path: Path) -> None:
    points = jnp.asarray([[0.0, 0.0], [0.2, 0.02], [0.5, -0.03], [0.8, 0.01], [1.0, 0.0]])
    domain = initialize_mode1_domain("line", points=points)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="line", steps=8, step_size=0.02, diagnostics_every=4)
    assert run.artifacts["mesh"].exists()
    imported = load_gmsh_msh(run.artifacts["mesh"], primary_element_kind="line")
    assert imported.primary_element_kind == "line"
    assert imported.topology.elements.shape[1] == 2


def test_mode1_square_workflow(tmp_path: Path) -> None:
    domain = initialize_mode1_domain("square", nx=7, ny=5, family="quad")
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="square", steps=6, step_size=0.02, diagnostics_every=3)
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "quad"


def test_mode1_polygon_workflow_and_import_workflow(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode1_domain("polygon", outer_boundary=outer, holes=[hole], target_edge_size=0.18)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="polygon_workflow", steps=10, step_size=0.02, diagnostics_every=5)
    imported_domain = initialize_mode1_domain("import-msh", path=run.artifacts["mesh"])
    rerun = run_mode1_workflow(imported_domain, output_dir=tmp_path / "imported", prefix="polygon_imported", steps=4, step_size=0.02, diagnostics_every=2)
    assert rerun.artifacts["mesh"].exists()


def test_mode1_polygon_quad_workflow(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode1_domain("polygon-quad", outer_boundary=outer, holes=[hole], target_edge_size=0.2)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="polygon_quad_workflow", steps=8, step_size=0.015, diagnostics_every=4)
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "quad"


def test_mode1_polygon_gmsh_backend_initialization(monkeypatch, tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

    def _fake_run(cmd, capture_output, text, check):
        msh_path = Path(cmd[-1])
        points = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=jnp.float32)
        elements = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
        boundary = GmshElementBlock(
            elements=jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=jnp.int32),
            element_kind="line",
            physical_tags=jnp.asarray([100, 100, 100, 100], dtype=jnp.int32),
            geometrical_tags=jnp.asarray([10, 10, 10, 10], dtype=jnp.int32),
        )
        export_gmsh_msh(
            msh_path,
            points,
            elements,
            physical_tags=jnp.asarray([1, 1], dtype=jnp.int32),
            geometrical_tags=jnp.asarray([1, 1], dtype=jnp.int32),
            element_kind="triangle",
            extra_element_blocks=(boundary,),
            physical_names={(1, 100): "outer_boundary", (2, 1): "domain"},
        )
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("topojax.mesh.domains._load_gmsh_module", lambda: (_ for _ in ()).throw(ModuleNotFoundError("gmsh")))
    monkeypatch.setattr("topojax.mesh.domains.subprocess.run", _fake_run)
    domain = initialize_mode1_domain("polygon", outer_boundary=outer, target_edge_size=0.25, backend="gmsh", gmsh_executable="gmsh")
    assert domain.topology.elements.shape == (2, 3)
    assert domain.metadata is not None
    assert domain.metadata.physical_names[(1, 100)] == "outer_boundary"


def test_mode1_polygon_gmsh_backend_prefers_python_bindings(monkeypatch) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    called = {"write": 0, "generate": 0}
    recombine_flag = {"value": 0.0}

    class _FakeGeo:
        def __init__(self):
            self._next = 1

        def addPoint(self, x, y, z, mesh_size):
            del x, y, z, mesh_size
            tag = self._next
            self._next += 1
            return tag

        def addLine(self, a, b):
            del a, b
            tag = self._next
            self._next += 1
            return tag

        def addCurveLoop(self, line_ids):
            del line_ids
            tag = self._next
            self._next += 1
            return tag

        def addPlaneSurface(self, curve_loop_ids):
            del curve_loop_ids
            return 1

        def synchronize(self):
            return None

    class _FakeModel:
        def __init__(self):
            self.geo = _FakeGeo()
            self.mesh = types.SimpleNamespace(generate=self._generate)

        def add(self, name):
            self.name = name

        def addPhysicalGroup(self, dim, tags, tag):
            del dim, tags, tag
            return None

        def setPhysicalName(self, dim, tag, name):
            del dim, tag, name
            return None

        def _generate(self, dim):
            assert dim == 2
            called["generate"] += 1

    class _FakeGmsh:
        def __init__(self):
            self.model = _FakeModel()
            self.option = types.SimpleNamespace(setNumber=self._set_number)
            self._initialized = False

        def isInitialized(self):
            return self._initialized

        def initialize(self):
            self._initialized = True

        def finalize(self):
            self._initialized = False

        def write(self, path):
            called["write"] += 1
            points = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=jnp.float32)
            if recombine_flag["value"]:
                elements = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
                kind = "quad"
            else:
                elements = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
                kind = "triangle"
            boundary = GmshElementBlock(
                elements=jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=jnp.int32),
                element_kind="line",
                physical_tags=jnp.asarray([100, 100, 100, 100], dtype=jnp.int32),
                geometrical_tags=jnp.asarray([10, 10, 10, 10], dtype=jnp.int32),
            )
            export_gmsh_msh(
                path,
                points,
                elements,
                physical_tags=jnp.asarray([1] * elements.shape[0], dtype=jnp.int32),
                geometrical_tags=jnp.asarray([1] * elements.shape[0], dtype=jnp.int32),
                element_kind=kind,
                extra_element_blocks=(boundary,),
                physical_names={(1, 100): "outer_boundary", (2, 1): "domain"},
            )

        def _set_number(self, name, value):
            if name == "Mesh.RecombineAll":
                recombine_flag["value"] = float(value)

    monkeypatch.setattr("topojax.mesh.domains._load_gmsh_module", lambda: _FakeGmsh())
    monkeypatch.setattr(
        "topojax.mesh.domains.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CLI fallback should not be used")),
    )
    domain = initialize_mode1_domain("polygon", outer_boundary=outer, target_edge_size=0.25, backend="gmsh", progress=False)
    assert called == {"write": 1, "generate": 1}
    assert domain.topology.elements.shape == (2, 3)
    assert domain.metadata is not None
    assert domain.metadata.physical_names[(1, 100)] == "outer_boundary"


def test_mode1_polygon_default_backend_prefers_gmsh_and_falls_back_to_native(monkeypatch) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    calls: list[str] = []

    def _fake_polygon_domain_tri_mesh_tagged(*args, backend: str, **kwargs):
        del args, kwargs
        calls.append(backend)
        if backend == "gmsh":
            raise FileNotFoundError("gmsh not found")
        points = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=jnp.float32)
        elements = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
        from topojax.mesh.topology import mesh_topology_from_points_and_elements
        from topojax.mesh.domains import DomainMeshMetadata

        return (
            mesh_topology_from_points_and_elements(points, elements),
            points,
            DomainMeshMetadata(boundary_element_blocks=(), physical_names={(2, 1): "domain"}),
        )

    monkeypatch.setattr("topojax.ad.workflow_common.polygon_domain_tri_mesh_tagged", _fake_polygon_domain_tri_mesh_tagged)
    domain = initialize_mode1_domain("polygon", outer_boundary=outer, target_edge_size=0.25, progress=False)
    assert calls == ["gmsh", "native"]
    assert domain.topology.elements.shape == (2, 3)
    assert domain.metadata is not None
    assert domain.metadata.physical_names[(2, 1)] == "domain"


def test_mode1_extruded_workflow(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode1_domain("extruded", outer_boundary=outer, holes=[hole], target_edge_size=0.18, height=1.0, layers=3)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="extruded_workflow", steps=6, step_size=0.015, diagnostics_every=3)
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"


def test_mode1_box_volume_workflow(tmp_path: Path) -> None:
    domain = initialize_mode1_domain(
        "box-volume",
        bbox_min=jnp.asarray([-1.0, -0.5, 0.0]),
        bbox_max=jnp.asarray([1.0, 0.5, 2.0]),
        nx=7,
        ny=5,
        nz=6,
    )
    run = run_mode1_workflow(
        domain,
        output_dir=tmp_path,
        prefix="box_volume_workflow",
        steps=6,
        step_size=0.012,
        diagnostics_every=3,
        export_stl_surface=True,
    )
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"
    assert run.artifacts["stl"].exists()


def test_mode1_box_alias_workflow(tmp_path: Path) -> None:
    domain = initialize_mode1_domain("box", nx=5, ny=4, nz=3)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="box_alias_workflow", steps=4, step_size=0.012, diagnostics_every=2)
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"


def test_mode1_sphere_surface_workflow(tmp_path: Path) -> None:
    domain = initialize_mode1_domain(
        "sphere-surface",
        center=jnp.asarray([0.5, 0.5, 0.5]),
        radius=0.42,
        n_lat=7,
        n_lon=14,
    )
    run = run_mode1_workflow(
        domain,
        output_dir=tmp_path,
        prefix="sphere_surface_workflow",
        steps=6,
        step_size=0.01,
        diagnostics_every=3,
    )
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "triangle"
    assert imported.physical_names[(2, 33)] == "sphere_surface"


def test_mode1_sphere_volume_workflow(tmp_path: Path) -> None:
    domain = initialize_mode1_domain(
        "sphere-volume",
        center=jnp.asarray([0.5, 0.5, 0.5]),
        radius=0.42,
        nx=9,
        ny=9,
        nz=9,
    )
    run = run_mode1_workflow(
        domain,
        output_dir=tmp_path,
        prefix="sphere_volume_workflow",
        steps=6,
        step_size=0.012,
        diagnostics_every=3,
    )
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"
    assert imported.physical_names[(2, 320)] == "sphere_boundary"

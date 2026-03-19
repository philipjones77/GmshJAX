import json
import os
import subprocess
import sys
from pathlib import Path


def _run_probe(code: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_topojax_package_import_is_lazy() -> None:
    payload = _run_probe(
        (
            "import json, sys; "
            "import topojax; "
            "print(json.dumps({"
            "'topojax_visualization': 'topojax.visualization' in sys.modules, "
            "'topojax_mode1': 'topojax.ad.mode1' in sys.modules, "
            "'topojax_topology': 'topojax.mesh.topology' in sys.modules"
            "}))"
        )
    )
    assert payload == {
        "topojax_visualization": False,
        "topojax_mode1": False,
        "topojax_topology": False,
    }


def test_topojax_simple_tri_mesh_import_loads_only_mesh_module() -> None:
    payload = _run_probe(
        (
            "import json, sys; "
            "from topojax import unit_square_tri_mesh; "
            "print(json.dumps({"
            "'callable': callable(unit_square_tri_mesh), "
            "'topojax_visualization': 'topojax.visualization' in sys.modules, "
            "'topojax_mode1': 'topojax.ad.mode1' in sys.modules, "
            "'topojax_domains': 'topojax.mesh.domains' in sys.modules, "
            "'topojax_topology': 'topojax.mesh.topology' in sys.modules"
            "}))"
        )
    )
    assert payload == {
        "callable": True,
        "topojax_visualization": False,
        "topojax_mode1": False,
        "topojax_domains": False,
        "topojax_topology": True,
    }


def test_topojax_core_mode1_import_does_not_pull_domains_or_visualization() -> None:
    payload = _run_probe(
        (
            "import json, sys; "
            "from topojax import unit_square_tri_mesh, optimize_mode1_fixed_topology; "
            "print(json.dumps({"
            "'tri_mesh_callable': callable(unit_square_tri_mesh), "
            "'mode1_callable': callable(optimize_mode1_fixed_topology), "
            "'topojax_visualization': 'topojax.visualization' in sys.modules, "
            "'topojax_domains': 'topojax.mesh.domains' in sys.modules, "
            "'topojax_mode1': 'topojax.ad.mode1' in sys.modules, "
            "'topojax_topology': 'topojax.mesh.topology' in sys.modules"
            "}))"
        )
    )
    assert payload == {
        "tri_mesh_callable": True,
        "mode1_callable": True,
        "topojax_visualization": False,
        "topojax_domains": False,
        "topojax_mode1": True,
        "topojax_topology": True,
    }


def test_topojax_arbitrary_domain_builder_loads_domains_only_when_requested() -> None:
    payload = _run_probe(
        (
            "import json, sys; "
            "from topojax import polygon_domain_tri_mesh_tagged; "
            "print(json.dumps({"
            "'callable': callable(polygon_domain_tri_mesh_tagged), "
            "'topojax_visualization': 'topojax.visualization' in sys.modules, "
            "'topojax_mode1': 'topojax.ad.mode1' in sys.modules, "
            "'topojax_domains': 'topojax.mesh.domains' in sys.modules"
            "}))"
        )
    )
    assert payload == {
        "callable": True,
        "topojax_visualization": False,
        "topojax_mode1": False,
        "topojax_domains": True,
    }


def test_topojax_visualization_module_loads_only_when_requested() -> None:
    payload = _run_probe(
        (
            "import json, sys; "
            "from topojax import plot_mode1_matplotlib; "
            "print(json.dumps({"
            "'callable': callable(plot_mode1_matplotlib), "
            "'topojax_visualization': 'topojax.visualization' in sys.modules"
            "}))"
        )
    )
    assert payload == {
        "callable": True,
        "topojax_visualization": True,
    }


def test_topojax_visualization_backends_load_only_on_call() -> None:
    payload = _run_probe(
        (
            "import json, sys; "
            "from topojax import plot_mode1_matplotlib, plot_mode1_pyvista, plot_mode1_gmsh, plot_topo_viser; "
            "print(json.dumps({"
            "'matplotlib_callable': callable(plot_mode1_matplotlib), "
            "'pyvista_callable': callable(plot_mode1_pyvista), "
            "'gmsh_callable': callable(plot_mode1_gmsh), "
            "'viser_callable': callable(plot_topo_viser), "
            "'matplotlib_backend_loaded': 'topojax._visualization_backends.matplotlib_backend' in sys.modules, "
            "'pyvista_backend_loaded': 'topojax._visualization_backends.pyvista_backend' in sys.modules, "
            "'gmsh_backend_loaded': 'topojax._visualization_backends.gmsh_backend' in sys.modules, "
            "'viser_backend_loaded': 'topojax._visualization_backends.viser_backend' in sys.modules"
            "}))"
        )
    )
    assert payload == {
        "matplotlib_callable": True,
        "pyvista_callable": True,
        "gmsh_callable": True,
        "viser_callable": True,
        "matplotlib_backend_loaded": False,
        "pyvista_backend_loaded": False,
        "gmsh_backend_loaded": False,
        "viser_backend_loaded": False,
    }


def test_smpljax_package_import_is_lazy() -> None:
    payload = _run_probe(
        (
            "import json, sys; "
            "import smpljax; "
            "print(json.dumps({"
            "'smpljax_visualization': 'smpljax.visualization' in sys.modules, "
            "'smpljax_mode1': 'smpljax.mode1' in sys.modules"
            "}))"
        )
    )
    assert payload == {
        "smpljax_visualization": False,
        "smpljax_mode1": False,
    }

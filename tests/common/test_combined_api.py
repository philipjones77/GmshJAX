from __future__ import annotations

from types import SimpleNamespace

from topojax.ad.modes import MeshMovementMode
from common import get_backends, smpl as smpl_api, topo as topo_api


def test_common_backend_registry_reports_topo_mode5_as_implemented() -> None:
    specs = {spec.backend.value: spec for spec in get_backends()}

    assert specs["topo"].implemented_modes == (
        MeshMovementMode.FIXED_TOPOLOGY,
        MeshMovementMode.REMESH_RESTART,
        MeshMovementMode.SOFT_CONNECTIVITY,
        MeshMovementMode.STRAIGHT_THROUGH,
        MeshMovementMode.FULLY_DYNAMIC,
    )
    assert specs["topo"].review_stub_modes == ()
    assert specs["smpl"].implemented_modes == (
        MeshMovementMode.FIXED_TOPOLOGY,
        MeshMovementMode.REMESH_RESTART,
        MeshMovementMode.SOFT_CONNECTIVITY,
        MeshMovementMode.STRAIGHT_THROUGH,
        MeshMovementMode.FULLY_DYNAMIC,
    )
    assert specs["smpl"].review_stub_modes == ()


def test_common_topo_mode12_wrappers_delegate_to_topojax(monkeypatch) -> None:
    mode1_domain = SimpleNamespace(kind="mode1")
    mode2_domain = SimpleNamespace(kind="mode2")
    mode1_run = SimpleNamespace(result="mode1-run")
    mode2_run = SimpleNamespace(result="mode2-run")

    monkeypatch.setattr(
        topo_api.topo_workflow,
        "initialize_mode1_domain",
        lambda kind, *, progress=True, **kwargs: (kind, progress, kwargs, mode1_domain),
    )
    monkeypatch.setattr(
        topo_api.topo_workflow,
        "initialize_mode2_domain",
        lambda kind, *, progress=True, **kwargs: (kind, progress, kwargs, mode2_domain),
    )
    monkeypatch.setattr(
        topo_api.topo_workflow,
        "run_mode1_workflow",
        lambda domain, **kwargs: (domain, kwargs, mode1_run),
    )
    monkeypatch.setattr(
        topo_api.topo_workflow,
        "run_mode2_restart_workflow",
        lambda domain, **kwargs: (domain, kwargs, mode2_run),
    )

    initialized1 = topo_api.initialize_mode1_domain("line", progress=False, n=8)
    initialized2 = topo_api.initialize_mode2_domain("polygon", progress=False, target_edge_size=0.2)
    run1 = topo_api.run_mode1_workflow(mode1_domain, output_dir="out/m1", steps=4)
    run2 = topo_api.run_mode2_restart_workflow(mode2_domain, output_dir="out/m2", cycles=2)
    alias_run2 = topo_api.run_mode2_workflow(mode2_domain, output_dir="out/m2b", cycles=1)

    assert initialized1 == ("line", False, {"n": 8}, mode1_domain)
    assert initialized2 == ("polygon", False, {"target_edge_size": 0.2}, mode2_domain)
    assert run1 == (mode1_domain, {"output_dir": "out/m1", "steps": 4}, mode1_run)
    assert run2 == (mode2_domain, {"output_dir": "out/m2", "cycles": 2}, mode2_run)
    assert alias_run2 == (mode2_domain, {"output_dir": "out/m2b", "cycles": 1}, mode2_run)


def test_common_smpl_runtime_and_workflow_wrappers_delegate_to_smpljax(monkeypatch) -> None:
    monkeypatch.setattr(
        smpl_api.smpl_runtime_api,
        "create",
        lambda model_path, use_cache=True, max_entries=2: ("create", model_path, use_cache, max_entries),
    )
    monkeypatch.setattr(
        smpl_api.smpl_runtime_api,
        "create_uncached",
        lambda model_path: ("create_uncached", model_path),
    )
    monkeypatch.setattr(
        smpl_api.smpl_runtime_api,
        "create_optimized",
        lambda model_path, **kwargs: ("create_optimized", model_path, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_runtime_api,
        "create_runtime",
        lambda model_path, **kwargs: ("create_runtime", model_path, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_io_api,
        "load_model",
        lambda path, **kwargs: ("load_model", path, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_io_api,
        "load_model_cached",
        lambda path, **kwargs: ("load_model_cached", path, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_io_api,
        "load_model_uncached",
        lambda path: ("load_model_uncached", path),
    )
    monkeypatch.setattr(
        smpl_api.smpl_io_api,
        "describe_model",
        lambda path, **kwargs: ("describe_model", path, kwargs),
    )
    monkeypatch.setattr(smpl_api.smpl_io_api, "io_cache_diagnostics", lambda: {"entries": 1})
    clear_calls: list[str] = []
    monkeypatch.setattr(smpl_api.smpl_io_api, "clear_io_cache", lambda: clear_calls.append("cleared"))
    monkeypatch.setattr(
        smpl_api.smpl_mode1_api,
        "default_mode1_params",
        lambda model, **kwargs: {"model": model, "kwargs": kwargs},
    )
    monkeypatch.setattr(
        smpl_api.smpl_mode1_api,
        "initialize_mode1_model",
        lambda **kwargs: ("initialize_mode1_model", kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_mode1_api,
        "optimize_mode1",
        lambda model, params, **kwargs: ("optimize_mode1", model, params, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_mode1_api,
        "run_mode1_workflow",
        lambda **kwargs: ("run_mode1_workflow", kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_mode2_api,
        "optimize_mode2",
        lambda model, params, **kwargs: ("optimize_mode2", model, params, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_mode3_api,
        "optimize_mode3",
        lambda model, params, **kwargs: ("optimize_mode3", model, params, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_mode4_api,
        "optimize_mode4",
        lambda model, params, **kwargs: ("optimize_mode4", model, params, kwargs),
    )
    monkeypatch.setattr(
        smpl_api.smpl_mode5_api,
        "optimize_mode5",
        lambda model, params, **kwargs: ("optimize_mode5", model, params, kwargs),
    )

    assert smpl_api.create("model.npz") == ("create", "model.npz", True, 2)
    assert smpl_api.create_uncached("model.npz") == ("create_uncached", "model.npz")
    assert smpl_api.create_optimized("model.npz", dtype="float32") == ("create_optimized", "model.npz", {"dtype": "float32"})
    assert smpl_api.create_runtime("model.npz", mode="optimized") == ("create_runtime", "model.npz", {"mode": "optimized"})
    assert smpl_api.load_model("model.npz", use_cache=False) == ("load_model", "model.npz", {"use_cache": False})
    assert smpl_api.load_model_cached("model.npz", max_entries=4) == ("load_model_cached", "model.npz", {"max_entries": 4})
    assert smpl_api.load_model_uncached("model.npz") == ("load_model_uncached", "model.npz")
    assert smpl_api.describe_model("model.npz", use_cache=True) == ("describe_model", "model.npz", {"use_cache": True})
    assert smpl_api.io_cache_diagnostics() == {"entries": 1}
    smpl_api.clear_io_cache()
    assert clear_calls == ["cleared"]
    assert smpl_api.default_mode1_params("model", batch_size=2) == {"model": "model", "kwargs": {"batch_size": 2}}
    assert smpl_api.initialize_mode1_model(model_path="model.npz") == ("initialize_mode1_model", {"model_path": "model.npz"})
    assert smpl_api.optimize_mode1("model", {"transl": 0}, steps=4) == ("optimize_mode1", "model", {"transl": 0}, {"steps": 4})
    assert smpl_api.run_mode1_workflow(model_path="model.npz", output_dir="out") == ("run_mode1_workflow", {"model_path": "model.npz", "output_dir": "out"})
    assert smpl_api.optimize_mode2("model", {"transl": 0}, diagnostics_every=2) == ("optimize_mode2", "model", {"transl": 0}, {"diagnostics_every": 2})
    assert smpl_api.optimize_mode3("model", {"transl": 0}, diagnostics_every=2) == ("optimize_mode3", "model", {"transl": 0}, {"diagnostics_every": 2})
    assert smpl_api.optimize_mode4("model", {"transl": 0}, diagnostics_every=2) == ("optimize_mode4", "model", {"transl": 0}, {"diagnostics_every": 2})
    assert smpl_api.optimize_mode5("model", {"transl": 0}, cycles=2) == ("optimize_mode5", "model", {"transl": 0}, {"cycles": 2})

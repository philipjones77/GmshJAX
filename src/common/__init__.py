"""Common track namespace for shared TopoSmplJAX standards and adapters."""

from __future__ import annotations

from importlib import import_module


_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    ".backends": (
        "BackendName",
        "BackendSpec",
        "build_mode_bridge",
        "get_backend_mode_report",
        "get_backend_spec",
        "get_backends",
        "repair_print_mesh",
    ),
    ".diagnostics": (
        "DiagnosticsLogger",
        "diagnostics_payload",
        "to_jsonable",
        "write_runtime_diagnostics",
    ),
    ".io": (
        "atomic_copy2",
        "atomic_write_csv",
        "atomic_write_json",
        "atomic_write_npz",
        "atomic_write_text",
        "ensure_parent_dir",
    ),
    ".mesh_repair": (
        "MeshRepairResult",
        "RepairBackend",
        "export_repaired_stl",
        "repair_smpl_mesh_for_printing",
        "repair_topo_mesh_for_printing",
        "repair_triangle_mesh",
    ),
    ".movement": (
        "MeshMovementTransform",
        "apply_mesh_movement",
        "apply_mesh_movement_jax",
        "apply_mesh_movement_numpy",
        "default_mesh_movement_transform",
        "mesh_movement_vector_size",
        "pack_mesh_movement_transform",
        "unpack_mesh_movement_vector",
    ),
}

_SUBMODULE_EXPORTS: dict[str, str] = {
    "diagnostics": ".diagnostics",
    "io": ".io",
    "movement": ".movement",
    "numpy_mesh": ".numpy_mesh",
    "smpl": ".smpl",
    "topo": ".topo",
}

_EXPORT_MAP = {
    name: (module_name, name)
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = [
    "BackendName",
    "BackendSpec",
    "DiagnosticsLogger",
    "MeshMovementTransform",
    "MeshRepairResult",
    "RepairBackend",
    "atomic_copy2",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_npz",
    "atomic_write_text",
    "build_mode_bridge",
    "diagnostics",
    "diagnostics_payload",
    "ensure_parent_dir",
    "export_repaired_stl",
    "get_backend_mode_report",
    "get_backend_spec",
    "get_backends",
    "io",
    "movement",
    "numpy_mesh",
    "apply_mesh_movement",
    "apply_mesh_movement_jax",
    "apply_mesh_movement_numpy",
    "default_mesh_movement_transform",
    "mesh_movement_vector_size",
    "pack_mesh_movement_transform",
    "repair_print_mesh",
    "repair_smpl_mesh_for_printing",
    "repair_topo_mesh_for_printing",
    "repair_triangle_mesh",
    "smpl",
    "topo",
    "to_jsonable",
    "unpack_mesh_movement_vector",
    "write_runtime_diagnostics",
]


def __getattr__(name: str):
    if name in _SUBMODULE_EXPORTS:
        module = import_module(_SUBMODULE_EXPORTS[name], __name__)
        globals()[name] = module
        return module
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

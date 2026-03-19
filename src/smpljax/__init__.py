"""smpljax public API with lazy exports."""

from __future__ import annotations

from importlib import import_module


_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    ".api": ("RuntimeMode", "create", "create_optimized", "create_runtime", "create_uncached"),
    ".body_models": ("ModelOutput", "SMPLJAXModel"),
    ".diagnostics": ("DiagnosticsLogger", "diagnostics_payload", "write_runtime_diagnostics"),
    ".io": ("clear_io_cache", "describe_model", "io_cache_diagnostics", "load_model", "load_model_cached", "load_model_uncached"),
    ".mesh_export": (
        "export_ct_mesh_payload_pose",
        "export_ct_mesh_payload_template",
        "export_posed_mesh",
        "export_template_mesh",
        "to_randomfields77_dynamic_mesh_state",
        "to_randomfields77_static_domain_payload",
    ),
    ".mode1": (
        "SMPLMode1OptimizationResult",
        "SMPLMode1StepDiagnostics",
        "build_mode1_visualization_payload",
        "default_mode1_objective",
        "export_mode1_artifacts",
        "mode1_history_payload",
        "mode1_metrics_payload",
        "optimize_mode1",
        "summarize_mode1_result",
    ),
    ".mode2": (
        "SMPLMode2OptimizationResult",
        "build_mode2_visualization_payload",
        "export_mode2_artifacts",
        "mode2_history_payload",
        "mode2_metrics_payload",
        "optimize_mode2",
        "visualize_mode2_result",
    ),
    ".mode_snapshot": ("SMPL_MODE1_SNAPSHOT_SCHEMA", "SMPL_MODE1_SNAPSHOT_VERSION", "SMPLMode1Snapshot", "load_mode1_snapshot"),
    ".optimized": (
        "CachePolicy",
        "CompileEvent",
        "ForwardInputs",
        "OptimizedSMPLJAX",
        "RuntimeDiagnostics",
        "WarmupCoverage",
    ),
    ".validation": ("ModelSummary", "summarize_model_data", "validate_model_data"),
    ".visualization": ("plot_mode1_matplotlib", "plot_mode1_pyvista", "plot_mode1_viser", "visualize_mode1_result"),
}

_EXPORT_MAP = {
    name: (module_name, name)
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = [
    "create",
    "create_uncached",
    "create_optimized",
    "create_runtime",
    "RuntimeMode",
    "ModelOutput",
    "SMPLJAXModel",
    "DiagnosticsLogger",
    "ForwardInputs",
    "OptimizedSMPLJAX",
    "CachePolicy",
    "CompileEvent",
    "RuntimeDiagnostics",
    "WarmupCoverage",
    "ModelSummary",
    "SMPL_MODE1_SNAPSHOT_SCHEMA",
    "SMPL_MODE1_SNAPSHOT_VERSION",
    "SMPLMode1Snapshot",
    "load_model",
    "load_model_uncached",
    "load_model_cached",
    "describe_model",
    "diagnostics_payload",
    "export_template_mesh",
    "export_posed_mesh",
    "export_ct_mesh_payload_template",
    "export_ct_mesh_payload_pose",
    "SMPLMode1OptimizationResult",
    "SMPLMode1StepDiagnostics",
    "SMPLMode2OptimizationResult",
    "optimize_mode1",
    "optimize_mode2",
    "default_mode1_objective",
    "mode1_history_payload",
    "mode1_metrics_payload",
    "mode2_history_payload",
    "mode2_metrics_payload",
    "build_mode1_visualization_payload",
    "build_mode2_visualization_payload",
    "export_mode1_artifacts",
    "export_mode2_artifacts",
    "load_mode1_snapshot",
    "summarize_mode1_result",
    "plot_mode1_matplotlib",
    "plot_mode1_pyvista",
    "plot_mode1_viser",
    "visualize_mode1_result",
    "visualize_mode2_result",
    "to_randomfields77_static_domain_payload",
    "to_randomfields77_dynamic_mesh_state",
    "validate_model_data",
    "summarize_model_data",
    "write_runtime_diagnostics",
    "io_cache_diagnostics",
    "clear_io_cache",
]


def __getattr__(name: str):
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

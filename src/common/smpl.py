"""SMPL track adapters that obey the common TopoSmplJAX protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from smpljax import api as smpl_runtime_api
from smpljax import io as smpl_io_api
from smpljax import mode1 as smpl_mode1_api
from smpljax import mode2 as smpl_mode2_api
from smpljax import mode3 as smpl_mode3_api
from smpljax import mode4 as smpl_mode4_api
from smpljax import mode5 as smpl_mode5_api
from smpljax.body_models import SMPLJAXModel
from smpljax.mesh_export import export_posed_mesh, export_template_mesh
from smpljax.optimized import ForwardInputs, OptimizedSMPLJAX
from topojax.ad.modes import MeshMovementMode
from topojax.mesh.topology import mesh_topology_from_points_and_elements
from topojax.rf77 import (
    RandomFields77ModeBridge,
    build_mode1_randomfields77_bridge,
    build_mode2_randomfields77_bridge,
    build_mode3_randomfields77_bridge,
    build_mode4_randomfields77_bridge,
    build_mode5_randomfields77_bridge,
)

from .mesh_repair import MeshRepairResult, RepairBackend, repair_smpl_mesh_for_printing


ModelLike = SMPLJAXModel | OptimizedSMPLJAX
RuntimeMode = smpl_runtime_api.RuntimeMode
SMPLMode1Provision = smpl_mode1_api.SMPLMode1Provision
SMPLMode1WorkflowRun = smpl_mode1_api.SMPLMode1WorkflowRun
SMPLMode1OptimizationResult = smpl_mode1_api.SMPLMode1OptimizationResult
SMPLMode2StageConfig = smpl_mode2_api.SMPLMode2StageConfig
SMPLMode2PhaseSummary = smpl_mode2_api.SMPLMode2PhaseSummary
SMPLMode2OptimizationResult = smpl_mode2_api.SMPLMode2OptimizationResult
SMPLMode3GroupSpec = smpl_mode3_api.SMPLMode3GroupSpec
SMPLMode3GroupSummary = smpl_mode3_api.SMPLMode3GroupSummary
SMPLMode3OptimizationResult = smpl_mode3_api.SMPLMode3OptimizationResult
SMPLMode4OptimizationResult = smpl_mode4_api.SMPLMode4OptimizationResult
SMPLMode5ControllerDecision = smpl_mode5_api.SMPLMode5ControllerDecision
SMPLMode5PhaseSummary = smpl_mode5_api.SMPLMode5PhaseSummary
SMPLMode5TransferSummary = smpl_mode5_api.SMPLMode5TransferSummary
SMPLMode5OptimizationResult = smpl_mode5_api.SMPLMode5OptimizationResult

__all__ = [
    "ForwardInputs",
    "MeshRepairResult",
    "ModelLike",
    "OptimizedSMPLJAX",
    "RepairBackend",
    "RuntimeMode",
    "SMPLJAXModel",
    "SMPLMode1OptimizationResult",
    "SMPLMode1Provision",
    "SMPLMode1WorkflowRun",
    "SMPLMode2OptimizationResult",
    "SMPLMode2PhaseSummary",
    "SMPLMode2StageConfig",
    "SMPLMode3GroupSpec",
    "SMPLMode3GroupSummary",
    "SMPLMode3OptimizationResult",
    "SMPLMode4OptimizationResult",
    "SMPLMode5ControllerDecision",
    "SMPLMode5OptimizationResult",
    "SMPLMode5PhaseSummary",
    "SMPLMode5TransferSummary",
    "build_mode_bridge",
    "clear_io_cache",
    "create",
    "create_optimized",
    "create_runtime",
    "create_uncached",
    "default_mode1_params",
    "describe_model",
    "initialize_mode1_model",
    "io_cache_diagnostics",
    "load_model",
    "load_model_cached",
    "load_model_uncached",
    "optimize_mode1",
    "optimize_mode2",
    "optimize_mode3",
    "optimize_mode4",
    "optimize_mode5",
    "repair_print_mesh",
    "run_mode1_workflow",
]


def create(model_path, use_cache: bool = True, max_entries: int = 2) -> SMPLJAXModel:
    """Create a baseline SMPL runtime through the common track."""
    return smpl_runtime_api.create(model_path, use_cache=use_cache, max_entries=max_entries)


def create_uncached(model_path) -> SMPLJAXModel:
    """Create an uncached baseline SMPL runtime through the common track."""
    return smpl_runtime_api.create_uncached(model_path)


def create_optimized(model_path, **kwargs: Any) -> OptimizedSMPLJAX:
    """Create an optimized SMPL runtime with compile caching enabled."""
    return smpl_runtime_api.create_optimized(model_path, **kwargs)


def create_runtime(model_path, **kwargs: Any) -> ModelLike:
    """Create an SMPL runtime by mode through the common track."""
    return smpl_runtime_api.create_runtime(model_path, **kwargs)


def load_model(path, **kwargs: Any):
    """Load validated SMPL model data through the common track."""
    return smpl_io_api.load_model(path, **kwargs)


def load_model_cached(path, **kwargs: Any):
    """Load SMPL model data through the bounded IO cache."""
    return smpl_io_api.load_model_cached(path, **kwargs)


def load_model_uncached(path):
    """Load SMPL model data without using the IO cache."""
    return smpl_io_api.load_model_uncached(path)


def describe_model(path, **kwargs: Any):
    """Return SMPL model diagnostics and summary information."""
    return smpl_io_api.describe_model(path, **kwargs)


def io_cache_diagnostics():
    """Report current SMPL IO cache diagnostics."""
    return smpl_io_api.io_cache_diagnostics()


def clear_io_cache() -> None:
    """Clear the SMPL IO cache."""
    smpl_io_api.clear_io_cache()


def default_mode1_params(model: ModelLike, **kwargs: Any) -> dict[str, jnp.ndarray]:
    """Build default SMPL mode-1 parameter tensors."""
    return smpl_mode1_api.default_mode1_params(model, **kwargs)


def initialize_mode1_model(**kwargs: Any) -> SMPLMode1Provision:
    """Initialize the practical SMPL mode-1 workflow runtime."""
    return smpl_mode1_api.initialize_mode1_model(**kwargs)


def optimize_mode1(model: ModelLike, params, **kwargs: Any) -> SMPLMode1OptimizationResult:
    """Run the practical SMPL mode-1 optimization loop."""
    return smpl_mode1_api.optimize_mode1(model, params, **kwargs)


def run_mode1_workflow(**kwargs: Any) -> SMPLMode1WorkflowRun:
    """Run the practical SMPL mode-1 workflow."""
    return smpl_mode1_api.run_mode1_workflow(**kwargs)


def optimize_mode2(model: ModelLike, params, **kwargs: Any) -> SMPLMode2OptimizationResult:
    """Run the staged SMPL mode-2 optimization workflow."""
    return smpl_mode2_api.optimize_mode2(model, params, **kwargs)


def optimize_mode3(model: ModelLike, params, **kwargs: Any) -> SMPLMode3OptimizationResult:
    """Run the soft-routing SMPL mode-3 optimization workflow."""
    return smpl_mode3_api.optimize_mode3(model, params, **kwargs)


def optimize_mode4(model: ModelLike, params, **kwargs: Any) -> SMPLMode4OptimizationResult:
    """Run the straight-through SMPL mode-4 optimization workflow."""
    return smpl_mode4_api.optimize_mode4(model, params, **kwargs)


def optimize_mode5(model: ModelLike, params, **kwargs: Any) -> SMPLMode5OptimizationResult:
    """Run the dynamic-controller SMPL mode-5 optimization workflow."""
    return smpl_mode5_api.optimize_mode5(model, params, **kwargs)


def _template_points(model: ModelLike) -> jnp.ndarray:
    mesh = export_template_mesh(model)
    return jnp.asarray(mesh["nodes"])


def _topology(model: ModelLike):
    mesh = export_template_mesh(model)
    return mesh_topology_from_points_and_elements(jnp.asarray(mesh["nodes"]), jnp.asarray(mesh["faces"], dtype=jnp.int32))


def _posed_points_fn(model: ModelLike):
    def geometry_fn(params: Mapping[str, Any] | None = None) -> jnp.ndarray:
        if params is None:
            return _template_points(model)
        mesh = export_posed_mesh(model, params if isinstance(params, ForwardInputs) else dict(params))
        return jnp.asarray(mesh["nodes"])

    return geometry_fn


def _bridge_source(points: jnp.ndarray, topology):
    return type("SmplBridgeSource", (), {"points": points, "topology": topology, "metadata": None})()


def build_mode_bridge(
    model: ModelLike,
    mode: MeshMovementMode | str,
    *,
    params: Mapping[str, Any] | ForwardInputs | None = None,
    **kwargs: Any,
) -> RandomFields77ModeBridge:
    normalized = MeshMovementMode(mode)
    points = _template_points(model) if params is None else jnp.asarray(export_posed_mesh(model, params)["nodes"])
    topology = _topology(model)
    geometry_fn = _posed_points_fn(model)

    if normalized == MeshMovementMode.FIXED_TOPOLOGY:
        return build_mode1_randomfields77_bridge(
            _bridge_source(points, topology),
            geometry_fn=geometry_fn,
            builder_options={"backend": "smpl", **kwargs.pop("builder_options", {})},
            **kwargs,
        )
    if normalized == MeshMovementMode.REMESH_RESTART:
        return build_mode2_randomfields77_bridge(
            _bridge_source(points, topology),
            geometry_fn=geometry_fn,
            builder_options={"backend": "smpl", **kwargs.pop("builder_options", {})},
            **kwargs,
        )
    if normalized == MeshMovementMode.SOFT_CONNECTIVITY:
        return build_mode3_randomfields77_bridge(
            points,
            topology,
            geometry_fn=geometry_fn,
            builder_options={"backend": "smpl", "candidate_kind": "smpl-parameter-group-soft-routing", **kwargs.pop("builder_options", {})},
            **kwargs,
        )
    if normalized == MeshMovementMode.STRAIGHT_THROUGH:
        return build_mode4_randomfields77_bridge(
            points,
            topology,
            geometry_fn=geometry_fn,
            builder_options={"backend": "smpl", "candidate_kind": "smpl-parameter-group-straight-through", **kwargs.pop("builder_options", {})},
            **kwargs,
        )
    if normalized == MeshMovementMode.FULLY_DYNAMIC:
        return build_mode5_randomfields77_bridge(
            points,
            topology,
            geometry_fn=geometry_fn,
            builder_options={"backend": "smpl", "controller_kind": "parameter-group-dynamic-controller", **kwargs.pop("builder_options", {})},
            **kwargs,
        )
    raise KeyError(f"Unsupported smpl mode: {mode}")


def repair_print_mesh(
    model: ModelLike,
    *,
    params: Mapping[str, Any] | ForwardInputs | None = None,
    batch_index: int = 0,
    backend: RepairBackend = "auto",
    **kwargs: Any,
) -> MeshRepairResult:
    return repair_smpl_mesh_for_printing(
        model,
        params=params,
        batch_index=batch_index,
        backend=backend,
        **kwargs,
    )

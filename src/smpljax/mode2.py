from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from .disk import atomic_write_json


class SMPLMode2OptimizationResult(NamedTuple):
    implementation_status: str
    summary: str
    metadata: dict[str, Any]


def optimize_mode2(*args, **kwargs) -> SMPLMode2OptimizationResult:
    del args, kwargs
    raise NotImplementedError("SMPL Mode 2 is not implemented yet; use the payload/interface stubs for planning and tooling integration.")


def mode2_history_payload(result: SMPLMode2OptimizationResult) -> dict[str, Any]:
    return {"schema_name": "smpljax.mode2.history", "schema_version": "1.0", "implementation_status": result.implementation_status}


def mode2_metrics_payload(result: SMPLMode2OptimizationResult) -> dict[str, Any]:
    return {
        "schema_name": "smpljax.mode2.metrics",
        "schema_version": "1.0",
        "implementation_status": result.implementation_status,
        "summary": result.summary,
        "metadata": result.metadata,
    }


def build_mode2_visualization_payload(result: SMPLMode2OptimizationResult, *, title: str = "SMPL Mode 2 Result") -> dict[str, Any]:
    return {
        "schema_name": "smpljax.mode2.visualization",
        "schema_version": "1.0",
        "title": title,
        "implementation_status": result.implementation_status,
        "summary": result.summary,
        "metadata": result.metadata,
    }


def export_mode2_artifacts(output_dir: str | Path, result: SMPLMode2OptimizationResult, *, prefix: str = "smpl_mode2") -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    viewer_path = out_dir / f"{prefix}_viewer.json"
    atomic_write_json(metrics_path, mode2_metrics_payload(result))
    atomic_write_json(viewer_path, build_mode2_visualization_payload(result))
    return {"metrics": metrics_path, "viewer_payload": viewer_path}


def visualize_mode2_result(result: SMPLMode2OptimizationResult, *, backend: str = "matplotlib", **kwargs):
    del kwargs
    raise NotImplementedError(f"SMPL Mode 2 visualization backend '{backend}' is not implemented yet; payload/schema stubs are in place.")

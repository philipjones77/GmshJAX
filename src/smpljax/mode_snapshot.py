from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .disk import atomic_write_npz


SMPL_MODE1_SNAPSHOT_SCHEMA = "smpljax.mode1.snapshot"
SMPL_MODE1_SNAPSHOT_VERSION = "1.0"


class SMPLMode1Snapshot(NamedTuple):
    schema_name: str
    schema_version: str
    params: dict[str, np.ndarray]
    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray
    parents: np.ndarray
    objective_history: np.ndarray
    grad_norm_history: np.ndarray
    metrics: dict[str, Any]
    visualization_payload: dict[str, Any]


def export_mode1_snapshot(
    path: str | Path,
    *,
    params: dict[str, np.ndarray],
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    parents: np.ndarray,
    objective_history: np.ndarray,
    grad_norm_history: np.ndarray,
    metrics: dict[str, Any],
    visualization_payload: dict[str, Any],
) -> Path:
    target = Path(path)
    atomic_write_npz(
        target,
        schema_name=np.asarray(SMPL_MODE1_SNAPSHOT_SCHEMA),
        schema_version=np.asarray(SMPL_MODE1_SNAPSHOT_VERSION),
        vertices=np.asarray(vertices),
        joints=np.asarray(joints),
        faces=np.asarray(faces, dtype=np.int32),
        parents=np.asarray(parents, dtype=np.int32),
        objective_history=np.asarray(objective_history),
        grad_norm_history=np.asarray(grad_norm_history),
        metrics_json=np.asarray(json_dumps(metrics)),
        visualization_payload_json=np.asarray(json_dumps(visualization_payload)),
        **{f"param_{key}": np.asarray(value) for key, value in params.items()},
    )
    return target


def load_mode1_snapshot(path: str | Path) -> SMPLMode1Snapshot:
    with np.load(Path(path), allow_pickle=False) as data:
        params = {
            key.removeprefix("param_"): np.asarray(data[key])
            for key in data.files
            if key.startswith("param_")
        }
        return SMPLMode1Snapshot(
            schema_name=str(np.asarray(data["schema_name"]).item()),
            schema_version=str(np.asarray(data["schema_version"]).item()),
            params=params,
            vertices=np.asarray(data["vertices"]),
            joints=np.asarray(data["joints"]),
            faces=np.asarray(data["faces"], dtype=np.int32),
            parents=np.asarray(data["parents"], dtype=np.int32),
            objective_history=np.asarray(data["objective_history"]),
            grad_norm_history=np.asarray(data["grad_norm_history"]),
            metrics=json_loads(str(np.asarray(data["metrics_json"]).item())),
            visualization_payload=json_loads(str(np.asarray(data["visualization_payload_json"]).item())),
        )


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def json_loads(payload: str) -> dict[str, Any]:
    import json

    return json.loads(payload)

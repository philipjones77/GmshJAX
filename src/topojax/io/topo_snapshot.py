"""Versioned native Topo snapshot import/export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from topojax.io.exports import GmshElementBlock
from topojax.mesh.topology import MeshTopology


TOPO_SNAPSHOT_SCHEMA = "topojax.mode1.topo_snapshot"
TOPO_SNAPSHOT_VERSION = "1.0"


class TopoSnapshot(NamedTuple):
    schema_name: str
    schema_version: str
    points: jnp.ndarray
    topology: MeshTopology
    metrics: dict[str, Any]
    history: dict[str, np.ndarray]
    step_diagnostics: tuple[dict[str, Any], ...]
    element_fields: dict[str, np.ndarray]
    visualization_payload: dict[str, Any]
    physical_names: dict[tuple[int, int], str]
    extra_element_blocks: tuple[GmshElementBlock, ...]
    metadata: dict[str, Any]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(v) for v in value]
    return value


def _encode_physical_names(physical_names: Mapping[tuple[int, int], str] | None) -> list[dict[str, Any]]:
    if not physical_names:
        return []
    return [{"dim": int(dim), "tag": int(tag), "name": str(name)} for (dim, tag), name in sorted(physical_names.items())]


def _decode_physical_names(entries: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    return {(int(entry["dim"]), int(entry["tag"])): str(entry["name"]) for entry in entries}


def export_topo_snapshot(
    path: str | Path,
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    metrics: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
    step_diagnostics: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    element_fields: Mapping[str, Any] | None = None,
    visualization_payload: Mapping[str, Any] | None = None,
    physical_names: Mapping[tuple[int, int], str] | None = None,
    extra_element_blocks: tuple[GmshElementBlock, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a stable native Topo snapshot as a versioned `.npz` bundle."""
    payload: dict[str, np.ndarray] = {
        "schema_name": np.asarray(TOPO_SNAPSHOT_SCHEMA),
        "schema_version": np.asarray(TOPO_SNAPSHOT_VERSION),
        "points": np.asarray(points),
        "elements": np.asarray(topology.elements, dtype=np.int32),
        "edges": np.asarray(topology.edges, dtype=np.int32),
        "node_ids": np.asarray(topology.node_ids, dtype=np.int32),
        "element_ids": np.asarray(topology.element_ids, dtype=np.int32),
        "element_entity_tags": np.asarray(topology.element_entity_tags, dtype=np.int32),
        "n_nodes": np.asarray(int(topology.n_nodes), dtype=np.int32),
        "metrics_json": np.asarray(json.dumps(_json_scalar(dict(metrics or {})), indent=2, sort_keys=True)),
        "step_diagnostics_json": np.asarray(json.dumps(_json_scalar(list(step_diagnostics or ())), indent=2, sort_keys=True)),
        "visualization_payload_json": np.asarray(
            json.dumps(_json_scalar(dict(visualization_payload or {})), indent=2, sort_keys=True)
        ),
        "physical_names_json": np.asarray(json.dumps(_encode_physical_names(physical_names), indent=2, sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(_json_scalar(dict(metadata or {})), indent=2, sort_keys=True)),
        "history_keys_json": np.asarray(json.dumps(sorted((history or {}).keys()))),
        "element_field_keys_json": np.asarray(json.dumps(sorted((element_fields or {}).keys()))),
        "extra_block_count": np.asarray(len(extra_element_blocks or ()), dtype=np.int32),
    }

    for key, value in (history or {}).items():
        payload[f"history_{key}"] = np.asarray(value)
    for key, value in (element_fields or {}).items():
        payload[f"element_field_{key}"] = np.asarray(value)
    for idx, block in enumerate(extra_element_blocks or ()):
        payload[f"extra_block_{idx}_elements"] = np.asarray(block.elements, dtype=np.int32)
        payload[f"extra_block_{idx}_element_kind"] = np.asarray(str(block.element_kind))
        payload[f"extra_block_{idx}_physical_tags"] = (
            np.asarray(block.physical_tags, dtype=np.int32) if block.physical_tags is not None else np.asarray([], dtype=np.int32)
        )
        payload[f"extra_block_{idx}_geometrical_tags"] = (
            np.asarray(block.geometrical_tags, dtype=np.int32)
            if block.geometrical_tags is not None
            else np.asarray([], dtype=np.int32)
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


def load_topo_snapshot(path: str | Path) -> TopoSnapshot:
    """Load a native Topo snapshot bundle."""
    with np.load(Path(path), allow_pickle=False) as data:
        schema_name = str(data["schema_name"].item())
        schema_version = str(data["schema_version"].item())
        history_keys = json.loads(str(data["history_keys_json"].item()))
        element_field_keys = json.loads(str(data["element_field_keys_json"].item()))
        history = {str(key): np.asarray(data[f"history_{key}"]) for key in history_keys}
        element_fields = {str(key): np.asarray(data[f"element_field_{key}"]) for key in element_field_keys}
        extra_blocks: list[GmshElementBlock] = []
        for idx in range(int(data["extra_block_count"].item())):
            physical_tags = np.asarray(data[f"extra_block_{idx}_physical_tags"], dtype=np.int32)
            geometrical_tags = np.asarray(data[f"extra_block_{idx}_geometrical_tags"], dtype=np.int32)
            extra_blocks.append(
                GmshElementBlock(
                    elements=jnp.asarray(np.asarray(data[f"extra_block_{idx}_elements"], dtype=np.int32), dtype=jnp.int32),
                    element_kind=str(data[f"extra_block_{idx}_element_kind"].item()),
                    physical_tags=None if physical_tags.size == 0 else jnp.asarray(physical_tags, dtype=jnp.int32),
                    geometrical_tags=None if geometrical_tags.size == 0 else jnp.asarray(geometrical_tags, dtype=jnp.int32),
                )
            )

        topology = MeshTopology(
            elements=jnp.asarray(np.asarray(data["elements"], dtype=np.int32), dtype=jnp.int32),
            edges=jnp.asarray(np.asarray(data["edges"], dtype=np.int32), dtype=jnp.int32),
            node_ids=jnp.asarray(np.asarray(data["node_ids"], dtype=np.int32), dtype=jnp.int32),
            element_ids=jnp.asarray(np.asarray(data["element_ids"], dtype=np.int32), dtype=jnp.int32),
            element_entity_tags=jnp.asarray(np.asarray(data["element_entity_tags"], dtype=np.int32), dtype=jnp.int32),
            n_nodes=int(data["n_nodes"].item()),
        )

        return TopoSnapshot(
            schema_name=schema_name,
            schema_version=schema_version,
            points=jnp.asarray(np.asarray(data["points"])),
            topology=topology,
            metrics=dict(json.loads(str(data["metrics_json"].item()))),
            history=history,
            step_diagnostics=tuple(json.loads(str(data["step_diagnostics_json"].item()))),
            element_fields=element_fields,
            visualization_payload=dict(json.loads(str(data["visualization_payload_json"].item()))),
            physical_names=_decode_physical_names(json.loads(str(data["physical_names_json"].item()))),
            extra_element_blocks=tuple(extra_blocks),
            metadata=dict(json.loads(str(data["metadata_json"].item()))),
        )

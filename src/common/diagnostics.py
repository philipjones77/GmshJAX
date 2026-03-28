"""Shared diagnostics helpers for backend-neutral runtimes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, ensure_parent_dir


__all__ = [
    "DiagnosticsLogger",
    "diagnostics_payload",
    "to_jsonable",
    "write_runtime_diagnostics",
]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if hasattr(value, "_asdict"):
        return {str(key): to_jsonable(val) for key, val in value._asdict().items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def diagnostics_payload(
    *,
    runtime: Any | None = None,
    diagnostics: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime is None and diagnostics is None:
        raise ValueError("Provide either runtime or diagnostics")
    diag_value = diagnostics if diagnostics is not None else runtime.diagnostics()
    payload = {"runtime": to_jsonable(diag_value)}
    if extra:
        payload["extra"] = to_jsonable(extra)
    return payload


def write_runtime_diagnostics(
    path: str | Path,
    *,
    runtime: Any | None = None,
    diagnostics: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    atomic_write_json(path, diagnostics_payload(runtime=runtime, diagnostics=diagnostics, extra=extra))


class DiagnosticsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = ensure_parent_dir(path)

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(to_jsonable(event), sort_keys=True))
            f.write("\n")

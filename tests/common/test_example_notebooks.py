from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_notebook(name: str) -> dict:
    path = REPO_ROOT / "examples" / "common" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _all_source_text(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_common_topo_notebook_uses_combined_topo_namespace() -> None:
    notebook = _load_notebook("common_topo_mode12_demo.ipynb")
    source = _all_source_text(notebook)

    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
    assert "from common import topo" in source
    assert "run_mode1_workflow" in source
    assert "run_mode2_workflow" in source


def test_common_smpl_notebook_uses_combined_smpl_namespace() -> None:
    notebook = _load_notebook("common_smpl_optimized_runtime_demo.ipynb")
    source = _all_source_text(notebook)

    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
    assert "from common import smpl" in source
    assert "create_runtime" in source
    assert "optimize_mode2" in source

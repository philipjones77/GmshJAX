from __future__ import annotations

import json
from pathlib import Path
import pytest


pytestmark = pytest.mark.benchmark


def test_combined_topo_mode1_benchmark_emits_expected_fields(
    benchmark_output_dir: Path,
    benchmark_runner,
) -> None:
    out_json = benchmark_output_dir / "combined_topo_mode1.json"
    proc = benchmark_runner(
        "benchmarks/common/benchmark_combined_topo_mode12.py",
        "--mode",
        "1",
        "--steps",
        "4",
        "--out-dir",
        str(benchmark_output_dir / "artifacts_mode1"),
        "--out",
        str(out_json),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "combined_topo_mode1"
    assert payload["api_namespace"] == "common.topo"
    assert payload["mode"] == 1
    assert payload["n_steps"] == 4
    assert payload["runtime_s"] > 0.0


def test_combined_topo_mode2_benchmark_emits_expected_fields(
    benchmark_output_dir: Path,
    benchmark_runner,
) -> None:
    out_json = benchmark_output_dir / "combined_topo_mode2.json"
    proc = benchmark_runner(
        "benchmarks/common/benchmark_combined_topo_mode12.py",
        "--mode",
        "2",
        "--steps",
        "4",
        "--cycles",
        "1",
        "--out-dir",
        str(benchmark_output_dir / "artifacts_mode2"),
        "--out",
        str(out_json),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "combined_topo_mode2"
    assert payload["api_namespace"] == "common.topo"
    assert payload["mode"] == 2
    assert payload["n_phases"] == 1
    assert payload["runtime_s"] > 0.0


def test_combined_smpl_benchmark_emits_expected_fields(
    benchmark_output_dir: Path,
    benchmark_runner,
) -> None:
    out_json = benchmark_output_dir / "combined_smpl.json"
    proc = benchmark_runner(
        "benchmarks/common/benchmark_combined_smpl_api.py",
        "--iters",
        "2",
        "--output-json",
        str(out_json),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))[0]
    assert payload["benchmark"] == "combined_smpl_api"
    assert payload["api_namespace"] == "common.smpl"
    assert payload["runtime"] == "optimized"
    assert payload["io_cache_entries"] == 1
    assert payload["io_cache_hits"] >= 1
    assert payload["compile_count"] >= 1
    assert payload["mode2_status"] == "implemented-staged-workflow"


def test_common_numpy_mesh_benchmark_emits_expected_fields(
    benchmark_output_dir: Path,
    benchmark_runner,
) -> None:
    out_json = benchmark_output_dir / "common_numpy_mesh.json"
    proc = benchmark_runner(
        "benchmarks/common/benchmark_numpy_mesh_runtime.py",
        "--mode",
        "5",
        "--out-dir",
        str(benchmark_output_dir / "artifacts_numpy"),
        "--out",
        str(out_json),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "common_numpy_mesh_runtime"
    assert payload["api_namespace"] == "common.numpy_mesh"
    assert payload["mode"] == 5
    assert payload["n_nodes"] > 0
    assert payload["rf77_cells"] > 0
    assert payload["runtime_s"] > 0.0

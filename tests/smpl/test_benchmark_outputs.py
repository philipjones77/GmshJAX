import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.benchmark


def _run_script(script: str, output_root: Path, benchmark_runner, *extra_args: str) -> dict:
    out_json = output_root / f"{Path(script).stem}.json"
    proc = benchmark_runner(
        script,
        "--iters",
        "2",
        "--output-json",
        str(out_json),
        *extra_args,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload
    return payload[0]


def test_forward_benchmark_emits_json(benchmark_output_dir: Path, benchmark_runner) -> None:
    output_root = benchmark_output_dir / "forward"
    output_root.mkdir(parents=True, exist_ok=True)
    record = _run_script("benchmarks/smpl/benchmark_forward.py", output_root, benchmark_runner)
    assert record["benchmark"] == "forward"
    assert record["runtime"] == "baseline"
    assert "jax_backend" in record
    assert "device_kind" in record


def test_optimized_benchmark_emits_json(benchmark_output_dir: Path, benchmark_runner) -> None:
    output_root = benchmark_output_dir / "optimized"
    output_root.mkdir(parents=True, exist_ok=True)
    record = _run_script("benchmarks/smpl/benchmark_optimized_runtime.py", output_root, benchmark_runner)
    assert record["benchmark"] == "optimized_runtime"
    assert record["runtime"] == "optimized"
    assert "batch_buckets" in record
    assert "fixed_padded_batch_size" in record
    assert "forbid_new_compiles" in record


def test_optimized_benchmark_emits_fixed_batch_policy_json(benchmark_output_dir: Path, benchmark_runner) -> None:
    output_root = benchmark_output_dir / "optimized_fixed"
    output_root.mkdir(parents=True, exist_ok=True)
    record = _run_script(
        "benchmarks/smpl/benchmark_optimized_runtime.py",
        output_root,
        benchmark_runner,
        "--batch-size",
        "16",
        "--warmup-batch-size",
        "32",
        "--fixed-padded-batch-size",
        "32",
        "--forbid-new-compiles",
    )
    assert record["benchmark"] == "optimized_runtime"
    assert record["batch_buckets"]
    assert record["fixed_padded_batch_size"] == 32
    assert record["forbid_new_compiles"] is True
    assert record["warmup_batch_size"] == 32


def test_mode2_benchmark_emits_json(benchmark_output_dir: Path, benchmark_runner) -> None:
    output_root = benchmark_output_dir / "mode2"
    output_root.mkdir(parents=True, exist_ok=True)
    record = _run_script("benchmarks/smpl/benchmark_mode2.py", output_root, benchmark_runner)
    assert record["benchmark"] == "mode2"
    assert record["runtime"] == "staged"
    assert record["implementation_status"] == "implemented-staged-workflow"
    assert record["n_stages"] >= 1


def test_mode3_benchmark_emits_json(benchmark_output_dir: Path, benchmark_runner) -> None:
    output_root = benchmark_output_dir / "mode3"
    output_root.mkdir(parents=True, exist_ok=True)
    record = _run_script("benchmarks/smpl/benchmark_mode3.py", output_root, benchmark_runner)
    assert record["benchmark"] == "mode3"
    assert record["runtime"] == "soft-routing"
    assert record["implementation_status"] == "implemented-soft-routing"
    assert record["n_groups"] >= 1


def test_mode4_benchmark_emits_json(benchmark_output_dir: Path, benchmark_runner) -> None:
    output_root = benchmark_output_dir / "mode4"
    output_root.mkdir(parents=True, exist_ok=True)
    record = _run_script("benchmarks/smpl/benchmark_mode4.py", output_root, benchmark_runner)
    assert record["benchmark"] == "mode4"
    assert record["runtime"] == "straight-through-routing"
    assert record["implementation_status"] == "implemented-straight-through-routing"
    assert record["n_groups"] >= 1


def test_mode5_benchmark_emits_json(benchmark_output_dir: Path, benchmark_runner) -> None:
    output_root = benchmark_output_dir / "mode5"
    output_root.mkdir(parents=True, exist_ok=True)
    record = _run_script("benchmarks/smpl/benchmark_mode5.py", output_root, benchmark_runner)
    assert record["benchmark"] == "mode5"
    assert record["runtime"] == "dynamic-controller"
    assert record["implementation_status"] == "implemented-dynamic-controller"
    assert record["cycles"] >= 1

import json
from pathlib import Path
import pytest


pytestmark = pytest.mark.benchmark


def _run_benchmark(script: str, benchmark_output_dir: Path, benchmark_runner, *extra_args: str) -> dict:
    out_json = benchmark_output_dir / f"{Path(script).stem}.json"
    proc = benchmark_runner(
        script,
        *extra_args,
        "--steps",
        "8",
        "--out",
        str(out_json),
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out_json.read_text(encoding="utf-8"))


def test_mode3_benchmark_emits_expected_fields(benchmark_output_dir: Path, benchmark_runner) -> None:
    payload = _run_benchmark("benchmarks/topo/benchmark_mode3_surrogate.py", benchmark_output_dir, benchmark_runner, "--kind", "tri")
    assert payload["benchmark"] == "mode3_surrogate"
    assert payload["kind"] == "tri"
    assert payload["n_steps"] == 8
    assert payload["runtime_s"] > 0.0
    assert payload["final_objective"] <= payload["initial_objective"] + 1.0e-8


def test_mode4_benchmark_emits_expected_fields(benchmark_output_dir: Path, benchmark_runner) -> None:
    payload = _run_benchmark(
        "benchmarks/topo/benchmark_mode4_straight_through.py",
        benchmark_output_dir,
        benchmark_runner,
        "--kind",
        "tet",
    )
    assert payload["benchmark"] == "mode4_straight_through"
    assert payload["kind"] == "tet"
    assert payload["n_steps"] == 8
    assert payload["runtime_s"] > 0.0
    assert payload["final_objective"] <= payload["initial_objective"] + 1.0e-8

import json
from pathlib import Path
import pytest


pytestmark = pytest.mark.benchmark


def test_mode5_benchmark_emits_expected_fields(benchmark_output_dir: Path, benchmark_runner) -> None:
    out_json = benchmark_output_dir / "mode5_dynamic.json"
    proc = benchmark_runner(
        "benchmarks/topo/benchmark_mode5_dynamic.py",
        "--kind",
        "tri",
        "--cycles",
        "2",
        "--optimization-steps",
        "4",
        "--surrogate-steps",
        "4",
        "--out-dir",
        str(benchmark_output_dir / "artifacts"),
        "--out",
        str(out_json),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "mode5_dynamic"
    assert payload["kind"] == "tri"
    assert payload["runtime_s"] > 0.0
    assert payload["implementation_status"] == "implemented-relaxed-dynamic"

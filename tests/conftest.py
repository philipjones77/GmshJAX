from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("benchmarks")
    group.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run tests marked as benchmark",
    )
    group.addoption(
        "--benchmark-output-root",
        action="store",
        default=None,
        help="directory for benchmark test artifacts; defaults to pytest tmp paths",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark: benchmark harness tests controlled by --run-benchmarks",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-benchmarks"):
        return
    skip_benchmark = pytest.mark.skip(reason="benchmark tests require --run-benchmarks")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


@pytest.fixture
def benchmark_output_dir(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
) -> Path:
    root = pytestconfig.getoption("--benchmark-output-root")
    if root is None:
        return tmp_path
    safe_nodeid = re.sub(r"[^A-Za-z0-9._-]+", "_", request.node.nodeid)
    output_dir = Path(root) / safe_nodeid
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def benchmark_runner() -> Callable[..., subprocess.CompletedProcess[str]]:
    repo_root = Path(__file__).resolve().parents[1]

    def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"
        return subprocess.run(
            [sys.executable, script, *args],
            cwd=str(repo_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    return _run

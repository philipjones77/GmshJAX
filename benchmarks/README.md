# Benchmarks

This folder is the repository-wide benchmark namespace.

Governed structure:

- `common/`: cross-backend or shared benchmark harnesses
- `topo/`: TopoJAX benchmark harnesses
- `smpl/`: smplJAX benchmark harnesses

Benchmark code should live under the subtree that owns the measured system.

Pytest controls benchmark execution.

- `PYTHONPATH=src python -m pytest -m benchmark --run-benchmarks`
- `PYTHONPATH=src python -m pytest --run-benchmarks tests/smpl/test_benchmark_outputs.py`
- `PYTHONPATH=src python -m pytest --run-benchmarks --benchmark-output-root outputs/benchmarks -m benchmark`

Benchmark harness scripts still live under `benchmarks/`, but their test execution is gated through the pytest benchmark marker and options above.

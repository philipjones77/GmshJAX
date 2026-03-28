# Benchmark Validation Policy Standard

## Purpose

This standard defines how benchmark harnesses are validated in the repository.

## Rules

- benchmark harnesses should be exercised under pytest control
- benchmark tests should use the `benchmark` marker
- benchmark harness tests should be opt-in through `--run-benchmarks`
- retained benchmark artifacts should use pytest-managed temp directories or an explicit `--benchmark-output-root`
- benchmark tests should validate emitted schema and representative fields, not only exit status

## Current Repo Policy

The shared benchmark gate is implemented through `tests/conftest.py` and the pytest marker configuration in `pyproject.toml`.

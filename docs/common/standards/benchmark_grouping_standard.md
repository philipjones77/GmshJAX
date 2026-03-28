# Benchmark Grouping Standard

## Purpose

This standard defines how benchmark harnesses and benchmark docs should be grouped in this repo.

## Rules

- benchmark code should live under the owning subtree: `benchmarks/common/`, `benchmarks/topo/`, or `benchmarks/smpl/`
- benchmark tests should be grouped under the matching `tests/...` subtree
- common benchmarks should focus on shared wrappers, startup behavior, exports, or shared runtimes
- backend benchmarks should focus on backend algorithms and optimized runtime paths
- benchmark outputs should use stable JSON payloads when they are retained or inspected by tests

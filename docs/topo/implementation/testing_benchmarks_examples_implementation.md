# Topo Testing, Benchmarks, And Examples Implementation

Topo validation is organized across:

- `tests/topo/` for workflow, geometry, diagnostics, IO, and benchmark harness checks
- `benchmarks/topo/` for retained benchmark harness scripts
- `examples/topo/` for notebooks and runnable demos

Pytest benchmark control:

- benchmark harness checks are marked with `benchmark`
- benchmark validation is opt-in through `--run-benchmarks`
- benchmark harnesses are treated as testable outputs, not only standalone scripts

Current benchmark harnesses cover:

- mode 1 fixed-topology optimization
- mode 1 artifact export
- mode 2 restart workflows
- mode 3 surrogate connectivity
- mode 4 straight-through connectivity
- mode 5 dynamic workflows

Example coverage includes both notebooks and script-style demos for domain initialization, adaptive workflows, diagnostics, and export paths.

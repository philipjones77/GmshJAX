# SMPL Testing, Benchmarks, And Examples Implementation

SMPL validation is organized across:

- `tests/smpl/` for IO, validation, optimization, visualization, and benchmark harness checks
- `benchmarks/smpl/` for retained runtime and mode benchmark harnesses
- `examples/smpl/` for notebooks and visualization or minimal runtime demos

Pytest benchmark control:

- retained benchmark harness checks use the `benchmark` marker
- benchmark harness validation is opt-in through `--run-benchmarks`
- benchmark harness tests validate emitted JSON schema and representative fields

Current retained benchmark harnesses cover:

- baseline forward runtime
- optimized runtime
- mode 2 through mode 5 optimization paths

Current example coverage includes:

- `m1_mode1_workflow_demo.ipynb`
- `minimal_forward.py`
- visualization demos

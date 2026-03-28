# Common Testing, Benchmarks, And Examples Implementation

The shared track is exercised through three companion surfaces:

- tests under `tests/common/`
- benchmark harnesses under `benchmarks/common/`
- example notebooks under `examples/common/`

Pytest control:

- ordinary shared-surface validation runs under the default test suite
- benchmark harness validation runs under pytest control through the `benchmark` marker
- benchmark harness tests are opt-in through `--run-benchmarks`
- shared benchmark output location can be controlled with `--benchmark-output-root`

Current common benchmark harnesses:

- `benchmark_combined_topo_mode12.py`
- `benchmark_combined_smpl_api.py`
- `benchmark_numpy_mesh_runtime.py`

Current common example notebooks:

- `common_topo_mode12_demo.ipynb`
- `common_smpl_optimized_runtime_demo.ipynb`

The shared tests validate schema, bridge, artifact, and lazy-import behavior rather than re-testing every provider algorithm in full depth.

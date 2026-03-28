# Common API, Test, Benchmark, And Example Inventory

Shared API surfaces:

- `common`
- `common.topo`
- `common.smpl`
- `common.numpy_mesh`

Shared tests:

- `tests/common/test_combined_api.py`
- `tests/common/test_combined_workflows.py`
- `tests/common/test_combined_benchmarks.py`
- `tests/common/test_example_notebooks.py`
- `tests/common/test_movement.py`
- `tests/common/test_numpy_mesh.py`
- `tests/common/test_mesh_repair.py`
- `tests/test_lazy_imports.py`

Shared benchmark harnesses:

- `benchmarks/common/benchmark_combined_topo_mode12.py`
- `benchmarks/common/benchmark_combined_smpl_api.py`
- `benchmarks/common/benchmark_numpy_mesh_runtime.py`

Shared example notebooks:

- `examples/common/common_topo_mode12_demo.ipynb`
- `examples/common/common_smpl_optimized_runtime_demo.ipynb`

Pytest benchmark control is implemented in `tests/conftest.py` and configured through `pyproject.toml`.

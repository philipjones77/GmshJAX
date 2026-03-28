# Topo API, Test, Benchmark, And Example Inventory

Primary API surfaces:

- `topojax`
- `common.topo`

Representative Topo tests:

- `tests/topo/test_mode1_fixed_topology.py`
- `tests/topo/test_mode1_workflow.py`
- `tests/topo/test_mode2_workflow.py`
- `tests/topo/test_mode34_workflows.py`
- `tests/topo/test_mode5_workflow.py`
- `tests/topo/test_rf77_bridge.py`
- `tests/topo/test_model_pipeline.py`
- `tests/topo/test_numpy_jax_parity.py`

Topo benchmark harnesses:

- `benchmarks/topo/benchmark_mode1_fixed_topology.py`
- `benchmarks/topo/benchmark_mode1_artifacts.py`
- `benchmarks/topo/benchmark_mode2_restart.py`
- `benchmarks/topo/benchmark_mode3_surrogate.py`
- `benchmarks/topo/benchmark_mode4_straight_through.py`
- `benchmarks/topo/benchmark_mode5_dynamic.py`

Benchmark harness validation tests:

- `tests/topo/test_mode34_benchmarks.py`
- `tests/topo/test_mode5_benchmark.py`
- benchmark-marked checks in `tests/topo/test_mode1_completion.py`, `tests/topo/test_mode1_fixed_topology.py`, and `tests/topo/test_mode2_workflow.py`

Example entry points:

- `examples/topo/m1_mode1_domain_initializers_demo.ipynb`
- `examples/topo/m1_fixed_topology_mode_demo.py`
- `examples/topo/m1_polygon_volume_topo_demo.py`
- `examples/topo/m3_*` workflow and export demos

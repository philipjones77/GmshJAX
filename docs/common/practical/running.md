# Running Common Surfaces

Typical usage paths:

- import `common` for backend discovery and shared helper access
- import `common.topo` or `common.smpl` for backend adapters
- import `common.numpy_mesh` for array-native workflows

When startup cost matters:

- keep imports at the narrowest submodule needed
- separate cold-start measurements from warm-path benchmark measurements
- use `PYTHONPATH=src python -m pytest -m benchmark --run-benchmarks` for benchmark harness validation rather than running retained benchmark checks ad hoc

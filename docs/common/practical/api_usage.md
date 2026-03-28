# Common API Usage

Recommended shared import patterns:

- `import common` for backend discovery, diagnostics helpers, movement helpers, and repair helpers
- `from common import topo as topo_api` for Topo workflow access
- `from common import smpl as smpl_api` for SMPL runtime and optimization access
- `from common import numpy_mesh` for non-AD array-native mesh workflows

Typical use cases:

- downstream code that wants one shared mode vocabulary across Topo and SMPL
- RF77-style array workflows that need a NumPy-native mesh carrier
- export and diagnostics tooling that should not depend directly on provider-internal layout
- SMPL runtime creation through `common.smpl.create_optimized(...)` or `common.smpl.create_runtime(...)` with backend-aware compile-cache and padding defaults

Benchmark validation command:

- `PYTHONPATH=src python -m pytest -m benchmark --run-benchmarks`

Main repository validation command:

- `PYTHONPATH=src python -m pytest -q`

Canonical example notebooks:

- `examples/common/common_topo_mode12_demo.ipynb`
- `examples/common/common_smpl_optimized_runtime_demo.ipynb`

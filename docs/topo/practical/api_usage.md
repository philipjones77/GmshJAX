# Topo API Usage

Primary import choices:

- `import topojax` for direct Topo-native usage
- `from common import topo as topo_api` for the shared adapter surface

Typical direct workflow pattern:

- initialize a domain
- run a mode workflow
- inspect metrics, history, and artifacts

Typical benchmark validation command:

- `PYTHONPATH=src python -m pytest -m benchmark --run-benchmarks tests/topo`

Typical functional validation command:

- `PYTHONPATH=src python -m pytest -q tests/topo`

Canonical Topo notebook and demo entry points include:

- `examples/topo/m1_mode1_domain_initializers_demo.ipynb`
- `examples/topo/m1_fixed_topology_mode_demo.py`
- `examples/topo/m3_*` adaptive and diagnostics demos

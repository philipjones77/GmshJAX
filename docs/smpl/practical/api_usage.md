# SMPL API Usage

Primary import choices:

- `import smpljax` for direct runtime use
- `from common import smpl as smpl_api` for the shared adapter surface

Typical direct workflow pattern:

- load or create a runtime
- prepare inputs or default parameters
- run forward evaluation or a mode workflow
- inspect diagnostics, artifacts, and exported mesh payloads

Benchmark validation command:

- `PYTHONPATH=src python -m pytest -m benchmark --run-benchmarks tests/smpl`

Typical functional validation command:

- `PYTHONPATH=src python -m pytest -q tests/smpl`

Canonical example entry points:

- `examples/smpl/m1_mode1_workflow_demo.ipynb`
- `examples/common/common_smpl_optimized_runtime_demo.ipynb`
- `examples/smpl/minimal_forward.py`
- visualization demos under `examples/smpl/`

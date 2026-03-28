# TopoSmplJAX

`TopoSmplJAX` is organized as three tracks:

- `common`: shared core structures, mode standards, bridge protocols, and cross-backend utilities
- `topojax`: the Topo track implementation for differentiable mesh generation, mesh movement, and mesh operators
- `smpljax`: the SMPL track implementation for fixed-topology body-model meshes, staged optimization, and parameter-routing workflows

The common track defines the repository-wide interface, while the `topo` and `smpl` tracks keep their own optimized implementations and obey the shared standards and protocols.
It also contains a NumPy-native mesh runtime for RF77-style workflows that need repository-owned mesh building without depending on JAX autodiff.

## Layout

- `src/topojax/`: existing TopoJAX backend
- `src/smpljax/`: imported smplJAX backend
- `src/common/`: shared common track namespace, standards, and backend registry
- `tests/topo/`: TopoJAX and shared tests
- `tests/smpl/`: imported smplJAX tests
- `docs/common/`: shared repository docs
- `docs/topo/`: TopoJAX docs
- `docs/smpl/`: imported smplJAX docs
- `contracts/topo/`: TopoJAX contracts
- `contracts/smpl/`: imported smplJAX contracts
- `private_data/smpl/`: imported smplJAX model-data skeleton
- `examples/common/`: common-track examples and notebooks
- `examples/topo/`: TopoJAX examples
- `examples/smpl/`: imported smplJAX examples
- `tools/common/`, `tools/topo/`, `tools/smpl/`: shared and backend-owned tooling

## Documentation

- `docs/common/`: shared standards, protocols, diagnostics, benchmarks, notebooks, and status
- `docs/topo/`: Topo runtime, workflow, theory, and status documentation
- `docs/smpl/`: SMPL runtime, cache-policy, workflow, and status documentation
- `contracts/topo/` and `contracts/smpl/`: binding API and runtime guarantees

## Install

```bash
python -m pip install -e .[dev]
```

Visualization extras:

```bash
python -m pip install -e .[dev,viz]
```

CPU execution works with the default JAX install. GPU execution requires a CUDA-enabled `jaxlib` build that matches the local CUDA stack.

## Common API

Backend registry:

```python
from common import get_backends, build_mode_bridge

print(get_backends())
```

Topo backend mode bridge:

```python
from topojax import initialize_mode1_domain
from common import build_mode_bridge

domain = initialize_mode1_domain("line", n=8)
bridge = build_mode_bridge("topo", domain, "fixed-topology-ad")
payload = bridge.to_randomfields77_mesh_payload()
```

SMPL backend mode bridge:

```python
from smpljax import create
from common import build_mode_bridge

model = create("private_data/smpl/models/validated/smplx/MODEL_NAME.npz")
bridge = build_mode_bridge("smpl", model, "fixed-topology-ad")
payload = bridge.to_randomfields77_mesh_payload()
```

Topo workflow surface through the common track:

```python
from common import topo

domain = topo.initialize_mode1_domain("line", n=8)
run = topo.run_mode1_workflow(domain, output_dir="outputs/topo_demo", progress=False)
restart = topo.run_mode2_workflow(
    topo.initialize_mode2_domain("square", family="quad", nx=6, ny=5),
    output_dir="outputs/topo_mode2_demo",
    progress=False,
)
```

SMPL runtime and workflow surface through the common track:

```python
from common import smpl

runtime = smpl.create_runtime("private_data/smpl/models/validated/smplx/MODEL_NAME.npz", mode="optimized")
print(smpl.io_cache_diagnostics())
provision = smpl.initialize_mode1_model(model=runtime, progress=False)
mode3 = smpl.optimize_mode3(runtime, provision.params, diagnostics_every=2)
```

Common NumPy mesh runtime for RF77-style array workflows:

```python
from topojax.mesh.topology import unit_square_tri_mesh
from common import numpy_mesh

topology, points = unit_square_tri_mesh(6, 5)
runtime = numpy_mesh.create_mode1_runtime(points, topology.elements, metadata={"source": "unit-square"})
bridge = numpy_mesh.build_mode_bridge(runtime)
payload = bridge.to_randomfields77_mesh_payload()
```

Common movement transform for moving a mesh under any mode without changing topology:

```python
import numpy as np
from common import MeshMovementTransform, apply_mesh_movement_numpy

transform = MeshMovementTransform(
    translation=np.array([0.1, -0.05], dtype=np.float32),
    scale=np.array([1.0, 1.0], dtype=np.float32),
    shear=np.array([0.02, 0.0], dtype=np.float32),
    bend=np.array([0.0, 0.01], dtype=np.float32),
)
moved_points = apply_mesh_movement_numpy(points, transform)
```

## Compatibility

- Existing `topojax` imports remain valid.
- Existing `smpljax` imports remain valid.
- New shared entry points live under `common`.

## Licensing

- Repository-authored code, docs, tests, examples, tools, and contracts are MIT-licensed under `LICENSE`.
- Path-specific coverage is documented in `LICENSES/CODE_AND_CONTENT.md`.
- `private_data/smpl/` is a boundary folder: repository-authored metadata is MIT when present, but upstream SMPL-family model assets remain under their original terms and are not relicensed by this repository.

## Validation

Full non-benchmark suite:

```bash
PYTHONPATH=src python -m pytest -q
```

Benchmark suite under pytest control:

```bash
PYTHONPATH=src python -m pytest -q -m benchmark --run-benchmarks
```

Focused common and backend slices:

```bash
PYTHONPATH=src python -m pytest -q tests/common tests/topo tests/smpl
PYTHONPATH=src python -m pytest -q --run-benchmarks tests/smpl/test_benchmark_outputs.py tests/common/test_combined_benchmarks.py
```

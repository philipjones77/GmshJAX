# Project Overview

`TopoSmplJAX` provides JAX-compatible mesh functionality through three coordinated tracks.

- `common` is the shared protocol layer for backend discovery, mode vocabulary, shared IO and diagnostics, mesh repair, the canonical mesh-movement transform structure, and the NumPy-native mesh runtime used for non-AD array workflows.
- `topo` provides Topo-oriented mesh generation, remeshing, and differentiable mesh-movement implementations.
- `smpl` provides SMPL-oriented body-model runtime implementations and parameter-routing workflows.

The goal is not to merge the backends internally. The goal is to make them interoperable through a common standard so downstream programs can rely on one coherent interface.

The common-track document tree is organized by governance, implementation, notation, objects, practical usage, reports, specs, standards, status, and theory.

Operationally, repository validation is split into two pytest-controlled surfaces:

- the main functional suite under `PYTHONPATH=src python -m pytest -q`
- the benchmark-marked suite under `PYTHONPATH=src python -m pytest -m benchmark --run-benchmarks`

# Common Architecture

`common` is the shared track for the three-track repository layout.

Its job is to define one stable protocol surface that both backend tracks obey without forcing them into one internal implementation.

Core responsibilities:

- backend registry and mode reporting
- shared filesystem IO and diagnostics helpers
- mesh-repair utilities that can be reached through a common surface
- a canonical mesh-movement transform structure used across NumPy and JAX paths
- a NumPy-native mesh runtime for RF77-style array workflows
- adapter modules that expose Topo and SMPL through shared conventions

Track boundary:

- `common` defines the protocol and shared helper layer
- `topo` owns differentiable mesh-generation and remeshing implementation details
- `smpl` owns body-model loading, optimized runtime construction, and parameter-routing implementation details

The more detailed architecture narrative lives in `governance/architecture.md`.

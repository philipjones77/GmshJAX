# Common API And Adapter Runtime Implementation

The shared runtime surface is built around a small number of stable entry points:

- `common` for lightweight shared helpers and backend discovery
- `common.topo` for Topo workflow adapters
- `common.smpl` for SMPL runtime and optimization adapters
- `common.numpy_mesh` for the NumPy-native mesh runtime

Cold-path design:

- `src/common/__init__.py` uses lazy attribute and submodule resolution
- backend-specific adapters are imported only when their surfaces are requested
- shared helper modules such as `io`, `diagnostics`, and `movement` stay lightweight enough for direct use

Provider ownership:

- `common.topo` delegates to `topojax.ad.workflow` and RF77 bridge builders
- `common.smpl` delegates to `smpljax` runtime, mode, IO, and bridge surfaces
- `common.numpy_mesh` is implemented in the shared track but reuses Topo topology, visualization, and bridge machinery

This keeps the common layer practical for users who want one coherent import path without pretending that Topo and SMPL share one internal runtime.

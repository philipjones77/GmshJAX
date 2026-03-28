# Common API Surface Structure

The shared API surface is organized into four kinds of entry points:

- root exports from `common`
- adapter submodules: `common.topo` and `common.smpl`
- helper submodules: `common.io`, `common.diagnostics`, `common.movement`
- array-native runtime surface: `common.numpy_mesh`

Design intent:

- the root package is for discovery and light-touch helpers
- adapter submodules are for backend-specific entry points with shared naming
- helper submodules are for reusable backend-neutral infrastructure
- the NumPy runtime is the shared non-AD mesh carrier for RF77-style workflows

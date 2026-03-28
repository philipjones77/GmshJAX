# Common Track Architecture

The common track exists to standardize how the two backend tracks present mesh functionality to downstream code.

Shared architectural principles:

- one mode vocabulary across the repo
- one backend registry for capability discovery
- one common adapter layer for Topo and SMPL entry points
- one shared diagnostics and artifact-writing style
- one canonical transform structure for moving mesh coordinates
- one NumPy-native runtime for RF77 and other array-oriented consumers

Module map:

- `src/common/backends.py` defines backend discovery and mode reporting
- `src/common/topo.py` exposes Topo through the shared surface
- `src/common/smpl.py` exposes SMPL through the shared surface
- `src/common/io.py` and `src/common/diagnostics.py` hold backend-neutral helpers
- `src/common/movement.py` defines the canonical movement transform
- `src/common/numpy_mesh.py` provides the NumPy runtime
- `src/common/mesh_repair.py` exposes printing and repair utilities through a shared surface

Boundary rule:

- if a concern is backend-neutral and needed by both tracks or by external consumers, it belongs in `common`
- if a concern is an optimization or algorithm specific to one backend, it belongs in that backend tree

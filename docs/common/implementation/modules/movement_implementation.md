# Movement Module Implementation

`src/common/movement.py` defines the canonical low-dimensional transform used to move mesh coordinates without changing topology.

The shared structure includes:

- translation
- per-axis scale
- shear
- bend

The module also owns:

- JAX application
- NumPy application
- identity-transform construction
- vector packing and unpacking

This keeps mesh-motion parameterization consistent across common and backend-specific workflows.

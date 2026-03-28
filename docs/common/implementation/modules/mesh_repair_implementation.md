# Mesh Repair Module Implementation

`src/common/mesh_repair.py` gives the shared track a single repair entry point while delegating backend-specific repair details to the appropriate provider.

This module belongs in the common layer because:

- external consumers should not need to know whether repair is coming from Topo or SMPL
- print-prep workflows are part of the shared export story
- repair outputs need a common result shape for diagnostics and downstream automation

# RandomFields77 Bridge Implementation

The common layer does not implement RF77 logic itself. It standardizes access to backend bridge builders and provides a NumPy runtime that can emit RF77-friendly payloads.

Shared expectations:

- each mode can be reported through one mode-aware bridge surface
- bridge payloads should include mesh coordinates, element connectivity, mode identity, and mode-specific metadata
- NumPy and JAX-facing paths should produce comparable payload structure where possible

Implementation boundary:

- Topo bridge builders live under `topojax.rf77`
- the common layer dispatches to those builders through `src/common/backends.py`, `src/common/topo.py`, `src/common/smpl.py`, and `src/common/numpy_mesh.py`

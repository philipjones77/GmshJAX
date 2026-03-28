# NumPy Mesh Module Implementation

`src/common/numpy_mesh.py` provides the array-native shared runtime.

Capabilities:

- mode-tagged mesh runtime construction
- diagnostics and metadata reporting
- RF77 bridge payload generation
- artifact export and persistence
- visualization payload generation
- common movement-transform application

This runtime is the non-AD counterpart to the JAX-facing backend paths and is intended for RF77 and similar consumers that need mesh handling without differentiating through the mesh.

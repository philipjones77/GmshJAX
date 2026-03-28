# SMPL Adapter Implementation

`src/common/smpl.py` presents the SMPL track through the shared conventions.

It exposes:

- model creation and optimized runtime creation
- mode-aware optimization entry points
- IO cache and model-description helpers
- RF77 bridge dispatch
- print-mesh repair routing

The adapter should stay thin and avoid re-implementing SMPL internals in the common layer.

# Implicit Adjoint Operator Solve Standard

This filename is retained for taxonomy parity with the reference standards set. In this repo it governs the analogous indirect or nontrivial differentiation boundaries in mesh and body-model workflows.

## Purpose

This standard defines how indirect differentiation, approximate gradients, and solver-like AD boundaries should be documented.

## Rules

- do not imply exact AD through a workflow when the provider actually uses restart boundaries, surrogate gradients, or straight-through estimators
- make differentiability limits explicit in common docs and diagnostics
- keep indirect differentiation behavior attached to the owning backend implementation, with the common layer only reporting it honestly
- tests and examples should distinguish exact, approximate, and unavailable gradient paths

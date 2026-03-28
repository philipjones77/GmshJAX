# SMPL Mode 1 Status

Status: active
Date: 2026-03-23

## Scope Of This Status Note

This note closes out the current practical Mode 1 work for `smpljax`.

Mode 1 in this repository means:

- create or receive a loaded SMPL-family runtime
- initialize a concrete parameter tree for one batch shape
- optimize parameters with JAX-compatible objectives while model topology stays fixed
- export stable snapshots, histories, metrics, and visualization payloads

Mode 1 does not mean every possible research workflow built on top of SMPL. It means the repository-supported practical runtime and optimization surface is in place.

## Completed Practical Coverage

The current Mode 1 surface now includes:

- runtime factories for uncached, cached, and optimized execution paths
- practical forward execution for repository-supported SMPL-family model assets loaded from `.npz` and `.pkl`
- default parameter initialization for body, global orientation, translation, and optional expression, face, and hand pose paths when the model declares them
- a repository-supported default Mode 1 objective and a public optimization entry point for custom objectives
- end-to-end Mode 1 workflow helpers for initialize, optimize, export, summarize, and visualize flows
- stable metrics, history, and snapshot payloads for Mode 1 results
- artifact export including JSON, CSV, NPZ, and visualization payload outputs
- mesh export hooks for template and posed surfaces
- supported visualization paths for Matplotlib, PyVista, and Viser
- validation, model-summary, and IO diagnostics tooling that supports runtime setup and asset checking

## Validation State

The following validation points have been covered in the repository test surface:

- Mode 1 optimization reduces the default objective on toy fixtures
- parameter initialization and workflow provisioning preserve expected shapes
- artifact export and snapshot reload semantics are stable
- visualization payloads are schema-stable
- Matplotlib, PyVista, and Viser integration points are exercised through real or fake-backend tests
- parity, validation, diagnostics, caching, export, and benchmark smoke tests exist around the broader runtime surface

Representative focused commands:

```bash
PYTHONPATH=src pytest tests/smpl/test_mode1.py -q
PYTHONPATH=src pytest tests/smpl/test_parity_smplx.py tests/smpl/test_validation.py tests/smpl/test_mesh_export.py -q
```

## Remaining Known Boundaries

The practical Mode 1 surface is in good shape, but these boundaries remain intentional:

- later modes now exist, but Mode 1 remains the simplest stable baseline and should stay lightweight
- no claim of complete parity with every upstream `smplx` option or extension beyond the repository-supported asset/runtime surface
- visualization backends remain optional and therefore somewhat environment-sensitive
- benchmark publication and machine-specific baselines are still being built out operationally

## Next Implementation Plan

The next tranche should focus on moving beyond the single-phase Mode 1 workflow without weakening the stable Mode 1 baseline.

### Near-Term Priorities

- keep the current Mode 1 contract and artifact schemas stable
- continue parity and validation hardening for supported asset variants
- make benchmark and diagnostics outputs easier to compare across machines and runtime modes

### Mode 2 Through Mode 5 Priorities

- keep the staged, routing, and dynamic-controller modes coherent with the shared common-mode vocabulary
- continue documenting what makes SMPL Mode 2 through Mode 5 distinct from ad hoc user loops so the public API stays coherent
- expand benchmark and backend-guidance coverage for CPU and GPU execution

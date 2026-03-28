# Common Repo Standards

## Purpose

This document defines repo-root communication and placement rules as they apply to the shared track.

## Repo Communication Surfaces

The repo uses a three-track model:

- `src/common/`, `docs/common/`, `tests/common/`, `benchmarks/common/`, `examples/common/`
- `src/topojax/`, `docs/topo/`, `tests/topo/`, `benchmarks/topo/`, `examples/topo/`
- `src/smpljax/`, `docs/smpl/`, `tests/smpl/`, `benchmarks/smpl/`, `examples/smpl/`

The root `README.md` should stay high-level.
`docs/common/project_overview.md` and `docs/common/governance/architecture.md` should carry the shared structural detail.

## Placement Rules

- shared policy and structure belong in `docs/common/`
- backend-specific policy belongs in `docs/topo/` or `docs/smpl/`
- binding backend guarantees belong in `contracts/topo/` or `contracts/smpl/`
- shared helper code belongs in `src/common/`
- canonical retained artifacts should prefer `outputs/`
- tests and benchmark harnesses should stay under pytest control rather than ad hoc shell-only workflows

# Common Documentation

This subtree documents the shared `common` track for the repository.

It now uses the same taxonomy shape as `arbplusJAX/docs`, but the content here is specific to the shared layer in this repo: common protocols, backend adapters, shared IO and diagnostics, mesh-movement transforms, the NumPy runtime, and the RF77-facing bridge surface.

Base documents:

- `architecture.md`
- `documentation_governance.md`
- `overview.md`
- `project_overview.md`
- `licensing.md`
- `milestone_m2_static_generator.md`

Subfolders:

- `governance/`
- `implementation/`
- `notation/`
- `objects/`
- `practical/`
- `reports/`
- `specs/`
- `standards/`
- `status/`
- `theory/`

Placement rules:

- shared contracts, standards, and backend-neutral runtime surfaces belong here
- Topo-only implementation details belong in `docs/topo/`
- SMPL-only implementation details belong in `docs/smpl/`

The canonical common-track architecture lives in `governance/architecture.md`.

Useful entry points:

- `practical/api_usage.md` for shared import and runtime usage
- `practical/benchmarking.md` for pytest-controlled benchmark policy
- `reports/api_test_benchmark_inventory.md` for the shared test, benchmark, and notebook inventory
- `status/api_test_benchmark_status.md` for the current shared-layer validation posture

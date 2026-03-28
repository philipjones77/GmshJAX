# Common Documentation Governance

This file is the short entry point for common-track documentation governance.

Canonical governance documents:

- `governance/documentation_governance.md`
- `governance/architecture.md`
- `licensing.md`

Placement rules:

- backend-neutral design rules, specs, standards, reports, and shared-runtime notes belong under `docs/common/`
- Topo-only implementation notes belong under `docs/topo/`
- SMPL-only implementation notes belong under `docs/smpl/`
- runtime guarantees that should be enforced by code belong in `contracts/`

The objective is to keep `docs/common/` authoritative for the shared layer instead of letting shared behavior drift into the backend-specific trees.

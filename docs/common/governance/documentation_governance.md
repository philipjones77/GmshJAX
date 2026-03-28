# Common Track Documentation Governance

This subtree is the authoritative documentation location for shared repository behavior.

Section rules:

- `implementation/` explains how the shared layer is built
- `notation/` defines shared terminology
- `objects/` catalogs common-facing runtime objects and functions
- `practical/` covers usage guidance
- `reports/` captures inventories and audit-style summaries
- `specs/` defines protocol expectations
- `standards/` defines repository conventions that apply to the shared layer
- `status/` tracks current gaps and plans
- `theory/` explains the conceptual model behind the shared abstractions

Authoring rules:

- write shared behavior here before duplicating it under `docs/topo/` or `docs/smpl/`
- link to backend-specific docs when implementation diverges
- keep documents implementation-aware but not implementation-exclusive
- prefer stable section names so audits and inventories can be refreshed mechanically

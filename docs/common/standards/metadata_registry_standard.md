# Metadata Registry Standard

## Purpose

This standard defines how machine-readable registries and inventories should be maintained in the common track.

## Rules

- keep compact JSON registries in `docs/common/reports/` when they improve tooling or auditability
- keep a human-readable companion markdown summary nearby
- make registry scope explicit so it does not get confused with backend-internal inventories
- keep registry values JSON-safe and stable enough for downstream tooling
- if a registry is generated, state the source and refresh path clearly

## Shared-Layer Examples

Relevant registry-like surfaces in this repo include:

- function or capability inventories for `src/common/`
- backend mode inventories
- common artifact or notebook inventories when they are maintained as reports

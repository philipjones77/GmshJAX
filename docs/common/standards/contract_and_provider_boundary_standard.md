# Contract And Provider Boundary Standard

## Purpose

This standard defines:

- what belongs in shared specs versus backend contracts
- how the common track relates to the Topo and SMPL providers
- how external consumers should depend on shared capability surfaces rather than provider internals

## Authority Split

Use this order when obligations overlap:

1. `docs/common/specs/`
2. `contracts/topo/` or `contracts/smpl/`
3. `docs/common/objects/`
4. `docs/common/theory/`
5. `docs/common/implementation/`
6. `docs/common/practical/`
7. `docs/common/reports/`
8. `docs/common/status/`

## Boundary Rules

- `common` defines shared names, payload expectations, mode vocabulary, and adapter behavior
- `topo` and `smpl` own backend-specific algorithms and optimizations
- shared docs must say when a surface delegates immediately to a provider
- downstream users should depend on `common` or stable backend public APIs, not on provider-internal file layout
- if a binding guarantee is backend-specific today, place it in that backend contract tree rather than inventing an implied common guarantee

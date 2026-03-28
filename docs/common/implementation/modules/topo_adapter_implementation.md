# Topo Adapter Implementation

`src/common/topo.py` presents the Topo track through the shared conventions.

It exposes:

- mode-specific domain initialization
- mode-specific workflow runners
- RF77 bridge dispatch
- print-mesh repair routing

The adapter exists so downstream code can reach Topo through the common track without importing Topo implementation modules directly.

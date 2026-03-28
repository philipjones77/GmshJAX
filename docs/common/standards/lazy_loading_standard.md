# Lazy Loading Standard

## Purpose

This standard defines the lazy-import policy for the shared track.

## Rules

- `import common` should stay lightweight
- heavyweight backend modules should load only when their surface is requested
- public routing modules should avoid eager import of broad Topo or SMPL runtime stacks
- optional visualization or external-integration helpers should not be on the cold import path
- documentation should call out intentionally lazy submodules
- startup-oriented benchmarks should measure import cost separately from runtime execution

## Shared-Layer Interpretation

The common package may eagerly establish:

- export tables
- light helper modules
- module `__getattr__` dispatch

It should not eagerly load:

- large backend runtime modules
- benchmark harnesses
- notebook helpers
- optional visualization stacks

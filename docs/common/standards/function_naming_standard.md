# Function Naming Standard

## Purpose

This standard defines naming expectations for common-facing functions and modules.

## Rules

- use intent-bearing verbs such as `get_`, `build_`, `create_`, `initialize_`, `run_`, `optimize_`, `load_`, `save_`, `export_`, `write_`, and `repair_`
- mode-specific functions should name the mode explicitly when that is part of the contract
- adapter surfaces should preserve backend identity in module placement rather than in every function name
- avoid vague names for wrapper functions that only forward into providers

## Shared-Layer Interpretation

The shared layer should be predictable to downstream callers without obscuring whether a surface is registry, helper, runtime, or adapter oriented.

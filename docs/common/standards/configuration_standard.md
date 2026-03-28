# Configuration Standard

## Purpose

This standard defines how runtime configuration should be surfaced in the shared track.

## Rules

- prefer explicit function arguments, runtime objects, or documented environment variables over hidden global state
- document device, dtype, cache, and artifact-output settings when they materially affect behavior
- do not bury benchmark or example configuration in undocumented notebook-only cells
- if a dedicated repo-level config tree is introduced, use a clearly named top-level directory such as `configs/`

## Shared-Layer Interpretation

For the current repo, most common-track configuration is expressed through explicit API arguments, pytest options, and JAX runtime environment choices.

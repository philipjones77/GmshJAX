# SMPL Mode 2 Through Mode 5 Status

Status: active
Date: 2026-03-28

## Purpose

This document records the current implemented state for the higher-order SMPL workflow modes in `smpljax`.

Unlike the Topo track, SMPL does not meaningfully change connectivity during optimization. The useful backend-specific analogue is staged and routed optimization over one fixed mesh topology while preserving the shared mode vocabulary required by the common track.

## Mode Definitions

### Mode 2

Mode 2 is the first staged workflow above the current single-phase Mode 1 path.

The current implementation:

- accepts explicit stage configs
- supports parameter-block freezing and unfreezing across stages
- exports stable phase summaries, histories, metrics, and visualization payloads

### Mode 3

Mode 3 is the soft parameter-group routing workflow.

The current implementation:

- optimizes SMPL parameters jointly with soft activation weights over a fixed set of parameter groups
- exports stable routing weights and per-group activation summaries
- preserves standard metrics, histories, and viewer-neutral visualization payloads

### Mode 4

Mode 4 is the straight-through routing workflow.

The current implementation:

- uses hard forward parameter-group activations with soft-gradient backward behavior
- exports both hard and soft routing state
- reuses the Mode 1 viewer adapters for visualization dispatch

### Mode 5

Mode 5 is the dynamic-controller workflow.

The current implementation:

- runs repeated surrogate-routing plus refinement cycles
- records controller decisions and transfer summaries explicitly
- keeps the resulting payloads and artifacts compatible with the shared common-track bridge layer

## Boundaries

- these modes are SMPL-specific interpretations of the shared common mode vocabulary, not claims that SMPL performs Topo-style remeshing
- Mode 3 through Mode 5 are intentionally centered on parameter-group routing and controller logic over fixed triangle topology
- CPU and GPU performance guidance is still being refined operationally even though the public APIs and artifact formats are now implemented

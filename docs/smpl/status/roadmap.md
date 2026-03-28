# Roadmap

Status: active
Date: 2026-03-28

## Mode 1 Closeout

- treat the current Mode 1 runtime as the shipped practical baseline for SMPL workflows in this repository
- keep the stable path centered on runtime creation, forward execution, optimization, diagnostics, snapshot export, mesh export, and viewer-dispatched visualization
- preserve support for optional expression, face, and hand parameter paths when present in loaded SMPL-family assets

See `mode1_status.md` for the current-state summary.

## Mode 2 Through Mode 5

- keep Mode 2 as the explicit staged workflow above the single-phase Mode 1 baseline
- keep Mode 3 and Mode 4 centered on SMPL-specific parameter-group routing rather than pretending SMPL changes mesh connectivity
- keep Mode 5 centered on controller-driven active-group selection, transferred state, and explicit refinement cycles

See `mode2_roadmap.md` for the current plan.

## Supporting Work

- keep parity validation against `smplx` healthy on canonical samples
- continue benchmark collection for batched, optimized, and cached runtime paths
- keep conversion and validation tooling aligned with the runtime contract
- publish clearer migration and API-usage docs around the now-implemented Mode 2 through Mode 5 workflow surface

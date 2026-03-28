# Caching And Recompilation Standard

## Purpose

This standard defines how cache behavior and recompilation behavior should be discussed and surfaced through the common track.

## Rules

- distinguish IO caching, object reuse, lazy-import caching, and JIT compilation caching
- do not describe backend compile cost as if it were owned by `common`
- benchmarks must separate cold start, first compile, and steady-state behavior when those costs matter
- examples should show the intended reuse boundary for repeated calls when the public surface supports one
- NumPy runtime documents must not imply JAX compilation where none exists

## Shared-Layer Interpretation

Relevant cache boundaries in this repo include:

- SMPL model IO caching
- JAX compilation caching in Topo and SMPL runtime paths
- lazy-loaded module caching in `common`
- optional prepared/runtime-object reuse in optimized call paths

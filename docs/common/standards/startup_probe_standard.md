# Startup Probe Standard

## Purpose

This standard defines how startup probes and capability probes should behave in the shared track.

## Rules

- availability probes for optional dependencies or backends should be explicit and local to the feature that needs them
- probe code should not run on the package cold path unless the import surface is specifically a probe surface
- startup probe benchmarks should measure only the intended probe work, not unrelated runtime construction

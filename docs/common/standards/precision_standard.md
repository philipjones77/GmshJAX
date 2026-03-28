# Precision Standard

## Purpose

This standard defines how dtype and precision policy should be discussed in the common track.

## Rules

- JAX and NumPy surfaces should be explicit about dtype-sensitive behavior
- do not silently advertise `float64` semantics when the provider actually ran in `float32`
- examples, benchmarks, and diagnostics should report relevant dtype settings
- the common layer should preserve provider precision choices rather than overriding them implicitly

## Shared-Layer Interpretation

The NumPy runtime, Topo JAX paths, and SMPL optimized runtimes all participate in the shared documentation story and should report precision honestly.

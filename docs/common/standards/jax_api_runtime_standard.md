# JAX API Runtime Standard

## Purpose

This standard defines the public JAX-runtime expectations for common-facing surfaces.

## Rules

- JAX-facing runtime behavior should be explicit about device, dtype, and compile or cache expectations
- common wrappers should preserve provider runtime semantics rather than obscuring them
- repeated-call JAX surfaces should document their intended reuse pattern
- diagnostics should expose compile or cache state where the provider already supports it
- bridge or export helpers should not force unnecessary host transfers on the mandatory hot path

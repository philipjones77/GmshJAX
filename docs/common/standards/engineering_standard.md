# Engineering Standard

## Purpose

This standard defines the engineering expectations for common-facing runtime surfaces and adapters.

## Rules

- common APIs should expose real provider behavior honestly
- CPU is the minimum execution and validation slice; GPU portability should be preserved where the provider supports it
- JAX-facing hot paths should avoid unnecessary host transfers and Python-side control flow
- approximate AD, surrogate gradients, or straight-through behavior must be reported explicitly rather than implied as exact AD
- diagnostics should make execution strategy, dtype, and cache or compile state inspectable where practical

## Shared-Layer Interpretation

The common layer is not where backend algorithms are implemented, but it is where engineering truthfulness about those algorithms must be preserved.

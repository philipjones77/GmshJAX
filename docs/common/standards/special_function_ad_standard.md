# Special Function AD Standard

This filename is retained for taxonomy parity with the reference standards set. In this repo it governs the analogous nonstandard AD semantics for mesh and routing workflows.

## Purpose

This standard defines how approximate or specialized gradient behavior should be described.

## Rules

- say explicitly when a mode uses exact AD, surrogate gradients, straight-through estimation, restart boundaries, or no AD
- do not collapse these distinctions into a single vague “JAX-compatible” claim
- common docs should report the provider’s real gradient semantics instead of smoothing them over
- tests and benchmarks that depend on gradient semantics should name the mode and method clearly

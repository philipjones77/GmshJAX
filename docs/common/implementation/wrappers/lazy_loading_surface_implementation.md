# Lazy Loading Surface Implementation

The common package uses lazy imports for exported helpers and submodules.

This matters because:

- some downstream programs need only shared metadata or lightweight helpers
- eager import of heavy JAX or model-loading stacks would distort startup cost
- the common layer should remain usable for static analysis, documentation tools, and inventory generation

The shared docs should call out when a surface is intentionally lazy so benchmark and startup reports stay interpretable.

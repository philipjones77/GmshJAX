# Common Runtime Implementation

The shared runtime surface is intentionally thin and delegates backend-specific work to Topo or SMPL.

Runtime layers:

- backend registry: capability discovery and common bridge dispatch
- adapter modules: `common.topo` and `common.smpl`
- shared helpers: IO, diagnostics, movement, mesh repair
- NumPy runtime: an array-native mesh container with diagnostics, artifact export, bridge payloads, and visualization hooks

Design constraints:

- do not make backend-specific optimizations a requirement of the common layer
- keep import boundaries light so common imports do not eagerly pull heavy runtime stacks
- keep diagnostics and exported artifacts shaped similarly across JAX and NumPy paths

# Common Notation

Core terms used in the shared layer:

- backend: a provider track such as `topo` or `smpl`
- mode: one of the common mesh-movement modes used across the repository
- runtime: a concrete object that carries mesh state plus mode and metadata
- bridge: a mode-aware export surface for external consumers such as RF77
- transform: the canonical low-dimensional mesh-movement parameterization
- diagnostics payload: a JSON-safe summary of runtime state
- artifact: a file emitted by save, export, benchmark, or report flows

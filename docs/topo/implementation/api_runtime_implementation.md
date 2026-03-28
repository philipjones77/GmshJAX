# Topo API And Runtime Implementation

Topo exposes two practical public surfaces:

- the `topojax` package root with lazy exports for user-facing runtime helpers
- the `common.topo` adapter surface for shared-track access

Primary API groups:

- mesh generators and topology utilities
- mode 1 through mode 5 workflow entry points
- diagnostics, artifact export, and snapshot helpers
- RF77 bridge builders and visualization payload builders

Import policy:

- `src/topojax/__init__.py` is a lazy export surface
- heavy optional visualization or workflow modules are not all loaded on the cold path
- the common adapter remains a thin layer over the underlying workflow implementation

Artifact model:

- workflow runs write mesh, metrics, history, and viewer-oriented payloads
- benchmark harnesses emit JSON summaries
- snapshots and RF77 payloads preserve the practical export story for downstream tools

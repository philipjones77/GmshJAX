# SMPL API And Runtime Implementation

SMPL exposes two practical public surfaces:

- the `smpljax` package root with lazy exports for runtime, optimization, diagnostics, and visualization helpers
- the `common.smpl` adapter surface for shared-track usage

Primary runtime families:

- baseline runtime creation through `SMPLJAXModel`
- optimized runtime creation through `OptimizedSMPLJAX`
- mode 1 through mode 5 workflow and optimization surfaces
- artifact, visualization, diagnostics, and bridge export helpers

Import policy:

- `src/smpljax/__init__.py` resolves exports lazily
- runtime creation, IO, and mode modules load only when their surfaces are requested
- the shared adapter remains thin and preserves provider ownership

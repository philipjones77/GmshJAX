# Implementation Overview (Version 1.0.0)

## Modules
- `smpljax.io`: model parsing + normalization + bounded IO cache.
- `smpljax.lbs`: core linear blend skinning math.
- `smpljax.body_models`: baseline model interface.
- `smpljax.optimized`: JIT-focused runtime with cache policy + diagnostics.
- `smpljax.visualization`: optional `viser` and `pyvista` integration.

## Runtime Modes
- Baseline mode (`SMPLJAXModel`) for straightforward usage.
- Optimized mode (`OptimizedSMPLJAX`) for iterative optimization and repeated inference.

## Performance Controls
- `CachePolicy.max_compiled`: bounds compiled function count.
- `CachePolicy.batch_buckets`: limits shape variants and recompiles.
- `CachePolicy.fixed_padded_batch_size`: pins repeated smaller requests to one padded compile shape.
- `CachePolicy.forbid_new_compiles`: turns unexpected compile-key growth into a runtime error.
- `CachePolicy.dtype`: controls global numeric dtype.

## Shared Adapter Surface
- `common.smpl` exposes the practical shared runtime surface for creation, optimization, bridge building, and repair entry points.

## Validation Surface
- benchmark harnesses live under `benchmarks/smpl/` and are validated under pytest benchmark control
- examples live under `examples/smpl/`
- notebook and artifact structure is documented in `docs/smpl/practical/` and `docs/smpl/reports/`

## Cross-Platform Notes
- Python entry points and path handling are compatible with Windows/Linux.
- Optional visualization stack remains isolated from core runtime.

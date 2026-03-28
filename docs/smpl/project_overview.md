# smplJAX Project Overview (Version 1.0.0)

This project implements SMPL-family body model functionality in pure JAX with:
- JIT-friendly execution paths.
- Optional optimized runtime with bounded compile cache.
- Autodiff-compatible forward pipeline.
- Cross-platform support for Windows and Linux.
- Optional visualization with `viser` and `pyvista`.

The core product is a canonical JAX-native runtime and parameter/output object model. Asset files and optional external tools are ingestion boundaries that should terminate in cacheable, serializable JAX-facing objects instead of opaque external handles.

For full architecture and standards, see:
- `docs/smpl/architecture.md`
- `docs/smpl/documentation_governance.md`
- `docs/smpl/coding_practices.md`
- `docs/smpl/standards.md`
- `docs/smpl/jax_standards.md`
- `docs/smpl/references.md`

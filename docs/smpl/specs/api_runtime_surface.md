# SMPL API Runtime Surface Spec

The practical SMPL runtime surface includes:

- lazy package-root exports in `smpljax`
- baseline and optimized runtime creation
- mode 1 through mode 5 optimization or workflow surfaces
- diagnostics, artifact export, mesh export, and visualization helpers
- shared-track adapter access through `common.smpl`

Practical invariants:

- compile and cache policy should be inspectable through diagnostics
- batch padding behavior should be explicit rather than implicit
- benchmark harnesses should emit machine-readable JSON payloads
- benchmark harness validation belongs under pytest benchmark control

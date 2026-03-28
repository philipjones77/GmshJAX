# Topo Runtime Surface Spec

The practical Topo runtime surface includes:

- lazy package-root exports in `topojax`
- mode 1 through mode 5 workflow entry points
- diagnostics and artifact payloads
- mesh import/export and snapshot surfaces
- common-track adapter access through `common.topo`

Practical invariants:

- mode workflows return structured run objects rather than only raw arrays
- benchmark harnesses emit machine-readable JSON payloads when retained
- artifact export surfaces preserve mesh, metrics, and viewer-oriented outputs
- benchmark harness validation belongs under pytest rather than shell-only process discipline

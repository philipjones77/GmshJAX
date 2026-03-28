# Testing Harness Implementation

The common track is validated primarily through `tests/common/` and a small number of cross-track smoke tests.

Coverage areas:

- lazy-import behavior
- shared IO and diagnostics helpers
- common movement transform packing, unpacking, and application
- NumPy mesh runtime save/load/export/visualization payloads
- backend bridge dispatch and capability reporting

The common tests should assert protocol shape and adapter behavior, not re-test the full internal algorithms of Topo or SMPL.

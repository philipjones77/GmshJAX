# Common API, Test, Benchmark, And Example Status

Current shared-layer state:

- common adapter surfaces exist for Topo and SMPL
- the NumPy mesh runtime exists and is covered in `tests/common/`
- shared benchmark harnesses exist and are run under pytest control
- shared example notebooks exist for Topo-through-common and SMPL-through-common usage
- lazy-import policy is covered by dedicated tests
- repository-level functional validation and benchmark validation both run through pytest rather than ad hoc benchmark scripts

Current shared-layer gaps worth watching:

- keep report inventories synchronized with exported surfaces as the adapters evolve
- expand shared examples only when they demonstrate real cross-backend value rather than duplicating backend-native examples
- keep cache, startup, and benchmark docs aligned with the actual benchmark harness policy

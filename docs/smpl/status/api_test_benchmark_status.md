# SMPL API, Test, Benchmark, And Example Status

Current state:

- direct and common-adapter SMPL API surfaces exist
- optimized runtime cache and padding policy is implemented and covered by tests
- retained benchmark harnesses exist for forward, optimized, and higher-mode paths
- benchmark harness validation is under pytest control
- example coverage exists for mode 1, minimal runtime usage, and visualization

Current gaps to watch:

- keep practical docs aligned with evolving higher-mode coverage
- keep benchmark and report inventories synchronized as new retained harnesses are added
- expand example notebooks only where they exercise a real public runtime pattern

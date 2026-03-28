# Benchmark Process Implementation

Benchmarks for the common track exist to measure wrapper overhead, import behavior, runtime construction, export paths, and shared NumPy-runtime operations.

Benchmark expectations:

- benchmark documents should say whether they measure cold import, first-use compile, or steady-state execution
- benchmark outputs should be serializable through the common diagnostics/reporting helpers
- common benchmarks should avoid hiding backend costs inside ambiguous aggregate timings

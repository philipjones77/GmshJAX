# Common Benchmarking

Common benchmarks should answer one of these questions:

- what is the startup or import cost of the shared layer
- what is the wrapper overhead of common adapters
- what is the runtime cost of NumPy mesh operations
- what backend costs are exposed through the shared surface

Benchmark docs should separate cold import, first-use, and steady-state timings whenever possible.

Repository benchmark harnesses are run under pytest control through the `benchmark` marker and the `--run-benchmarks` option.

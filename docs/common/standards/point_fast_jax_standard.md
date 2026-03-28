# Point Fast JAX Standard

This filename is retained for taxonomy parity with the reference standards set. In this repo it governs the analogous hot-path JAX runtime surfaces for mesh and body-model workflows.

## Purpose

This standard defines the expectations for repeated-call, low-overhead JAX paths exposed through Topo, SMPL, or their common adapters.

## Rules

- hot-path JAX surfaces should minimize Python overhead and unnecessary host transfers
- the intended reuse boundary should be visible, such as prepared inputs, warmed runtimes, or cached compiled callables
- common wrappers over hot-path JAX surfaces should stay thin
- benchmark docs should separate first-call compile cost from steady-state execution

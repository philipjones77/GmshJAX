# Backend Realized Performance Standard

## Purpose

This standard defines how performance claims should be made for common-facing surfaces that ultimately execute on Topo, SMPL, NumPy, CPU, or GPU backends.

## Rules

- performance reports must say which backend actually did the work
- distinguish wrapper overhead from provider runtime cost
- distinguish CPU and GPU measurements explicitly
- separate cold import, first compile, and steady-state timings when relevant
- include dtype, device, and batch or mesh-size context for material measurements

## Shared-Layer Interpretation

`common` may report performance of its own helper logic, but it must not hide backend ownership of the heavy numeric path.

# Contracts

This folder contains binding runtime and API guarantees for `topojax`.

Top-level architectural rule:

- external meshing or viewer tools exist to produce or inspect canonical JAX-native runtime objects
- the repository center of gravity is the stable JAX object model and its cacheable snapshots, not long-lived external tool handles

Current contract focus:

- stable public API guarantees for the shipped Mode 1 fixed-topology workflow surface
- stable public API guarantees for the shipped Mode 2 remesh-restart workflow surface
- experimental runtime coverage for Mode 3 through Mode 5 on selected Topo families, without the same binding stability guarantees as Mode 1 and Mode 2

For current implementation status, see:

- `docs/topo/status/mode1_2d_3d_status.md`
- `docs/topo/status/mode2_roadmap.md`
- `docs/topo/status/mode5_roadmap.md`

Licensing:

- repository-authored contract text in this folder is MIT-licensed under the repository root license
- see [../../LICENSE](../../LICENSE) and [../../LICENSES/CODE_AND_CONTENT.md](../../LICENSES/CODE_AND_CONTENT.md)

# Mesh Movement Modes Implementation

This note maps the five mesh movement modes onto the current TopoJAX implementation.

## Mode 1: Fixed Topology AD

Primary code paths:

- `src/topojax/ad/compiled.py`
- `src/topojax/ad/mode1.py`
- `src/topojax/ad/pipeline.py`
- `src/topojax/mesh/operators.py`
- `src/topojax/visualization.py`

This mode is the production baseline. Node coordinates move, while topology is held fixed.

Current initialization scope includes first-class 1D lines, arbitrary 2D polygon domains, extruded polygonal 3D domains, implicit-volume tetrahedral backgrounds, and imported fixed meshes. The concrete initializers are cataloged in `docs/topo/objects/mode1_domain_initializers.md`.

The practical Mode 1 runtime surface now also includes:

- fixed-topology optimization through `optimize_mode1_fixed_topology`
- compact scalar summaries through `summarize_mode1_result`
- stable history and metrics payloads through `mode1_history_payload` and `mode1_metrics_payload`
- final artifact export through `export_mode1_artifacts`
- repository-supported visualization through native Gmsh, Matplotlib, PyVista, and Viser

## Mode 2: Remesh Restart

Primary code paths:

- `src/topojax/ad/restart.py`
- `src/topojax/mesh/adaptive.py`
- `src/topojax/mesh/adaptive_quad.py`
- `src/topojax/mesh/adaptive_tet.py`

This mode is implemented for triangle, quad, and tet workflows. It performs fixed-topology optimization, then a separate discrete remesh step, then restarts optimization on the new topology.

The practical runtime surface now also includes:

- a shared workflow-domain initializer through `initialize_mode2_domain`
- a high-level restart runner through `run_mode2_restart_workflow`
- final-mesh and phase-history export through `export_mode2_artifacts`
- compact restart summaries through `summarize_mode2_restart_result`
- shared visualization payload and viewer dispatch scaffolding through `build_mode2_visualization_payload` and `visualize_mode2_result`

The concrete current-state definition, boundaries, and TODO list for this mode are tracked in `docs/topo/status/mode2_roadmap.md`.

## Mode 3: Soft Connectivity Surrogate

Primary code paths:

- `src/topojax/ad/surrogate.py`

Current implementation scope now includes:

- 2D triangle edge-flip surrogate candidates
- 2D quad diagonal-choice surrogate candidates
- 3D tetra keep-vs-split surrogate candidates

The practical runtime surface is exposed through `optimize_soft_connectivity`, `run_mode3_workflow`, and the corresponding artifact export and visualization payload helpers.

## Mode 4: Straight-Through Connectivity

Primary code paths:

- `src/topojax/ad/straight_through.py`

Current implementation scope now includes:

- 2D triangle edge-flip straight-through candidates
- 2D quad diagonal-choice straight-through candidates
- 3D tetra keep-vs-split straight-through candidates

This mode keeps hard forward choices with surrogate backward gradients and is exposed through `optimize_straight_through_connectivity` and `run_mode4_workflow`.

## Mode 5: Fully Dynamic Remeshing

Current status: implemented relaxed prototype for 2D triangle and 3D tetra workflows.

The current implementation is a practical hybrid dynamic loop rather than exact reverse-mode through arbitrary remeshing. It combines:

- fixed-topology optimization inside each phase
- an in-loop surrogate phase using mode-3 or mode-4 style connectivity variables
- explicit controller decisions for remesh triggers
- explicit node-field and element-field transfer across remesh events

The practical runtime surface is exposed through `optimize_dynamic_topology`, `run_mode5_workflow`, and `export_mode5_artifacts`.

The concrete prototype plan, milestones, code targets, non-goals, and approximation strategy are tracked in `docs/topo/status/mode5_roadmap.md`.

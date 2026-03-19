# API Contract

Status: active
Version: v1.0
Date: 2026-03-14

## Scope

This document defines the binding runtime and API guarantees for the stable public entry surface of TopoJAX.

## Stable Public Surface

The preferred stable entry points are the symbols re-exported from `topojax.__init__`.

This contract currently covers these categories of public surface:

- mesh model and topology types such as `MeshModel`, `MeshState`, `MeshTopology`, and `GEntity`
- canonical topology builders such as `unit_interval_line_mesh`, `unit_square_tri_mesh`, `unit_square_quad_mesh`, `mapped_quad_mesh`, and `unit_cube_tet_mesh`
- arbitrary-domain topology builders such as `polyline_mesh`, `polygon_domain_tri_mesh`, `polygon_domain_tri_mesh_tagged`, `polygon_domain_quad_mesh`, `polygon_domain_quad_mesh_tagged`, `box_volume_tet_mesh`, `box_volume_tet_mesh_tagged`, `implicit_volume_tet_mesh`, `implicit_volume_tet_mesh_tagged`, `sphere_surface_tri_mesh`, `sphere_surface_tri_mesh_tagged`, `sphere_volume_tet_mesh`, `sphere_volume_tet_mesh_tagged`, and `extruded_polygon_tet_mesh`
- adaptive remeshing entry points such as `adaptive_remesh_tri`, `adaptive_remesh_quad`, and `adaptive_remesh_tet`
- AD pipeline builders such as `build_quality_value_and_grad`, `build_parametric_quality_value_and_grad`, and `build_model_parametric_quality_value_and_grad`
- Mode 1 runtime helpers such as `optimize_mode1_fixed_topology`, `benchmark_mode1_fixed_topology`, `summarize_mode1_result`, and `collect_mode1_jax_diagnostics`
- Mode 1 artifact and visualization helpers such as `mode1_history_payload`, `mode1_metrics_payload`, `export_mode1_artifacts`, `build_mode1_visualization_payload`, `export_mode1_visualization_payload`, `plot_mode1_matplotlib`, `plot_mode1_pyvista`, `plot_mode1_gmsh`, `visualize_mode1`, and `visualize_mode1_result`
- Mode 2 restart helpers such as `optimize_remesh_restart_tri`, `optimize_remesh_restart_quad`, `optimize_remesh_restart_tet`, `summarize_mode2_restart_result`, and `run_mode2_restart_workflow`
- Mode 2 visualization helpers and later-mode payload contracts such as `build_mode2_visualization_payload`, `build_mode3_visualization_payload`, `build_mode4_visualization_payload`, `build_mode5_visualization_payload`, `plot_topo_viser`, and `visualize_mode2_result`
- runtime precision helpers such as `set_runtime_precision` and `get_runtime_precision`
- interchange and workflow helpers such as `export_gmsh_msh`, `export_binary_stl`, `load_gmsh_msh`, `launch_gmsh_viewer`, `initialize_mode1_domain`, and `run_mode1_workflow`

Lower-level implementation modules may evolve without preserving the same stability level as long as the public API contract remains satisfied.

## Behavioral Guarantees

- Public API symbols covered by this contract must remain importable from `topojax` unless a documented breaking change is made.
- The package import boundary must preserve a core-path split:
  - simple topology builders and core Mode 1 AD helpers remain directly importable from `topojax`
  - arbitrary-domain builders remain directly importable from `topojax` but load their heavier domain-meshing module only when those symbols are requested
  - visualization and external-viewer helpers remain optional and must not load as part of simple topology or core Mode 1 AD imports
  - backend-specific visualization implementations remain isolated so `gmsh`, `matplotlib`, `pyvista`, and `viser` are loaded only when selected
- Static generator helpers must continue to return fixed-shape topology and coordinate structures compatible with the current tests and examples.
- Arbitrary-domain initializers and imported fixed meshes must return topologies compatible with fixed-topology Mode 1 optimization after initialization.
- Adaptive remeshing entry points must preserve their role as top-level workflow functions for triangle, quad, and tetrahedral adaptation.
- Runtime precision helpers must remain the supported way to change shared numeric precision policy.
- Public AD pipeline builders must continue to produce JAX-compatible callable outputs intended for value and gradient evaluation.
- Mode 1 optimization must remain a JAX-native fixed-topology coordinate-optimization path that keeps element connectivity unchanged while updating node coordinates.
- Mode 1 artifact export must continue to emit a stable final snapshot, stable scalar metrics, stable history payloads, a Gmsh `.msh` export, and a viewer-neutral visualization payload.
- Mode 1 viewer dispatch must continue to support the repository-supported backends: native Gmsh, Matplotlib, PyVista, and Viser.
- Native Gmsh remains the default visualization backend for Topo Mode 1 and generic Topo state dispatch.
- Tagged domain metadata exported through Gmsh blocks and physical names must remain stable enough to support the current tests and examples.

## Change Control

- Semantic mesh and object definitions belong in `docs/topo/specs/`.
- Explanatory implementation notes belong in `docs/topo/implementation/`.
- Any change that alters a contracted public symbol name, signature, return-shape convention, or top-level workflow role must update this file in the same change.

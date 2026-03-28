# API Contract

Status: active
Version: v1.0
Date: 2026-03-28

## Scope

This document defines the binding runtime and API guarantees for the stable public entry surface of `smpljax`.

## Stable Public Surface

The preferred stable entry points are the symbols re-exported from `smpljax.__init__`.

This contract currently covers these categories of public surface:

- runtime and factory APIs such as `create`, `create_uncached`, `create_optimized`, `create_runtime`, `RuntimeMode`, `SMPLJAXModel`, and `OptimizedSMPLJAX`
- runtime diagnostics and IO APIs such as `describe_model`, `load_model`, `load_model_cached`, `load_model_uncached`, `io_cache_diagnostics`, `clear_io_cache`, `diagnostics_payload`, and `write_runtime_diagnostics`
- validation and summary APIs such as `validate_model_data`, `summarize_model_data`, and `ModelSummary`
- mesh-export helpers such as `export_template_mesh`, `export_posed_mesh`, `export_ct_mesh_payload_template`, `export_ct_mesh_payload_pose`, `to_randomfields77_static_domain_payload`, and `to_randomfields77_dynamic_mesh_state`
- Mode 1 workflow APIs such as `default_mode1_params`, `default_mode1_objective`, `initialize_mode1_model`, `optimize_mode1`, `run_mode1_workflow`, `summarize_mode1_result`, `mode1_history_payload`, `mode1_metrics_payload`, `build_mode1_visualization_payload`, `export_mode1_artifacts`, `SMPLMode1Provision`, `SMPLMode1OptimizationResult`, and `SMPLMode1WorkflowRun`
- Mode 1 visualization helpers such as `plot_mode1_matplotlib`, `plot_mode1_pyvista`, `plot_mode1_viser`, and `visualize_mode1_result`
- Mode 1 snapshot helpers such as `SMPLMode1Snapshot`, `SMPL_MODE1_SNAPSHOT_SCHEMA`, `SMPL_MODE1_SNAPSHOT_VERSION`, and `load_mode1_snapshot`
- Mode 2 staged-workflow APIs such as `SMPLMode2StageConfig`, `SMPLMode2PhaseSummary`, `SMPLMode2OptimizationResult`, `mode2_history_payload`, `mode2_metrics_payload`, `build_mode2_visualization_payload`, `export_mode2_artifacts`, `optimize_mode2`, and `visualize_mode2_result`
- Mode 3 soft-routing APIs such as `SMPLMode3GroupSpec`, `SMPLMode3GroupSummary`, `SMPLMode3OptimizationResult`, `mode3_history_payload`, `mode3_metrics_payload`, `build_mode3_visualization_payload`, `export_mode3_artifacts`, `optimize_mode3`, and `visualize_mode3_result`
- Mode 4 straight-through routing APIs such as `SMPLMode4OptimizationResult`, `mode4_history_payload`, `mode4_metrics_payload`, `build_mode4_visualization_payload`, `export_mode4_artifacts`, `optimize_mode4`, and `visualize_mode4_result`
- Mode 5 dynamic-controller APIs such as `SMPLMode5PhaseSummary`, `SMPLMode5ControllerDecision`, `SMPLMode5TransferSummary`, `SMPLMode5OptimizationResult`, `mode5_history_payload`, `mode5_metrics_payload`, `build_mode5_visualization_payload`, `export_mode5_artifacts`, `optimize_mode5`, and `visualize_mode5_result`

Lower-level implementation modules may evolve without preserving the same stability level as long as the public API contract remains satisfied.

## Behavioral Guarantees

- Core forward paths must remain autodiff-compatible.
- Runtime caching must remain bounded and observable through diagnostics.
- IO model cache must remain capped to 2 models in memory by default.
- Non-cached model loading must remain available for workflows that need strict reload semantics.
- Optimized runtime must continue to support controlled recompiles through fixed-shape execution and optional batch bucketing.
- Formal model IO diagnostics must remain available through `describe_model(...)` and validator tooling.
- Mode 1 must remain the supported end-to-end SMPL optimization workflow in this repository, including initialization, optimization, artifact export, snapshot export, and supported viewer-neutral or backend-dispatched visualization payloads.
- Mode 1 parameter initialization must continue to expose the optional expression, face, and hand pose paths whenever the loaded model declares those degrees of freedom.
- Mode 2 must remain the repository-supported staged optimization workflow above Mode 1, with explicit phase summaries, stable history and metrics payloads, artifact export, and visualization dispatch through the Mode 1 viewer adapters.
- Mode 3 must remain the repository-supported soft parameter-group routing workflow, including stable group summaries and exported routing weights.
- Mode 4 must remain the repository-supported straight-through parameter-group routing workflow, with hard forward routing state and soft-gradient backward behavior exposed in stable payloads.
- Mode 5 must remain the repository-supported dynamic-controller workflow built around explicit controller decisions, state-transfer summaries, and stable exported histories.

## Compatibility

- Supported model containers: `.npz`, `.pkl`.
- Supported OS targets: Windows, Linux.

## Change Control

- Semantic SMPL object and runtime definitions belong in `docs/smpl/specs/`.
- Explanatory implementation notes belong in `docs/smpl/implementation/`.
- Any change that alters a contracted public symbol name, signature, return-shape convention, or top-level workflow role must update this file in the same change.

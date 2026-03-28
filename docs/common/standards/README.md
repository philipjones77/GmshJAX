# Common Standards

This section holds the common-track standards for this repository.

It mirrors the standard-set shape used in `arbplusJAX/docs/standards`, but the content here is adapted to the shared mesh/runtime repo with three tracks:

- `common`
- `topo`
- `smpl`

Reading model:

- owner standard: the main policy for a concept
- companion standard: a narrower rule for one aspect of that concept
- repo adaptation: the way this repo applies the rule to common/topo/smpl and the NumPy runtime

Some filenames are inherited from the reference taxonomy even when the meaning is slightly reinterpreted for this repo. In those cases, the file states the analogous mesh/runtime concept it governs here.

## Concept Groups

### 1. Runtime, JAX Surface, Startup, And Performance

Primary owners:

- `jax_api_runtime_standard.md`
- `engineering_standard.md`

Companion documents:

- `api_surface_kinds_standard.md`
- `backend_realized_performance_standard.md`
- `caching_recompilation_standard.md`
- `configuration_standard.md`
- `core_scalar_service_calling_standard.md`
- `implicit_adjoint_operator_solve_standard.md`
- `jax_surface_policy_standard.md`
- `lazy_loading_standard.md`
- `point_fast_jax_standard.md`
- `point_surface_standard.md`
- `precision_standard.md`
- `startup_compile_playbook_standard.md`
- `startup_compile_standard.md`
- `startup_import_boundary_standard.md`
- `startup_probe_standard.md`

### 2. Validation, Benchmarks, And Examples

Primary owners:

- `benchmark_validation_policy_standard.md`
- `example_notebook_standard.md`

Companion documents:

- `api_usability_standard.md`
- `benchmark_grouping_standard.md`
- `pytest_test_naming_standard.md`

### 3. Portability, Layout, And Release Discipline

Primary owners:

- `environment_portability_standard.md`
- `experiment_layout_standard.md`
- `release_execution_checklist_standard.md`

### 4. Contracts, Providers, And Metadata

Primary owners:

- `contract_and_provider_boundary_standard.md`
- `metadata_registry_standard.md`

### 5. Documentation Outputs

Primary owners:

- `generated_documentation_standard.md`
- `report_standard.md`
- `status_standard.md`
- `repo_standards.md`
- `update_standard.md`

### 6. Theory, Naming, And AD Semantics

Primary owners:

- `function_naming_standard.md`
- `theory_notation_standard.md`

Companion documents:

- `special_function_ad_standard.md`

## Detailed Standards

- `api_surface_kinds_standard.md`
- `api_usability_standard.md`
- `backend_realized_performance_standard.md`
- `benchmark_grouping_standard.md`
- `benchmark_validation_policy_standard.md`
- `caching_recompilation_standard.md`
- `configuration_standard.md`
- `contract_and_provider_boundary_standard.md`
- `core_scalar_service_calling_standard.md`
- `engineering_standard.md`
- `environment_portability_standard.md`
- `example_notebook_standard.md`
- `experiment_layout_standard.md`
- `function_naming_standard.md`
- `generated_documentation_standard.md`
- `implicit_adjoint_operator_solve_standard.md`
- `jax_api_runtime_standard.md`
- `jax_surface_policy_standard.md`
- `lazy_loading_standard.md`
- `metadata_registry_standard.md`
- `point_fast_jax_standard.md`
- `point_surface_standard.md`
- `precision_standard.md`
- `pytest_test_naming_standard.md`
- `release_execution_checklist_standard.md`
- `repo_standards.md`
- `report_standard.md`
- `special_function_ad_standard.md`
- `startup_compile_playbook_standard.md`
- `startup_compile_standard.md`
- `startup_import_boundary_standard.md`
- `startup_probe_standard.md`
- `status_standard.md`
- `theory_notation_standard.md`
- `update_standard.md`

Current inventories belong in `docs/common/reports/`.
Current implementation progress belongs in `docs/common/status/`.

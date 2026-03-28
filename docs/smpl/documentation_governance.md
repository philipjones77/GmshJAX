# smplJAX Documentation Governance

---
*Status*: ACTIVE
*Version*: v1.0
*Date*: 2026-03-09

**Base Documents**

* `docs/smpl/architecture.md`
* `docs/smpl/documentation_governance.md`
* `docs/smpl/project_overview.md`

---

## 0. Scope

This file governs repository structure and documentation placement for `smplJAX`.

It defines:
- the repo-root folder layout
- the conceptual role of each `docs/` subfolder
- the authority boundary between `docs/` and top-level `contracts/`
- where new documents should be placed

It does not define mathematical semantics. Those belong in `docs/smpl/specs/`.

## 1. Repository Layout

Top-level structure:
- `docs/common/`: shared repository documentation
- `docs/topo/`: TopoJAX documentation authority
- `docs/smpl/`: smplJAX documentation authority
- `contracts/smpl/`: binding smplJAX runtime and API guarantees
- `src/`: executable implementation
- `tests/`: automated test namespace
  Expected subtree:
  - `tests/common/`
  - `tests/topo/`
  - `tests/smpl/`
- `examples/smpl/`: runnable demos and user-facing templates
- `experiments/`: exploratory namespace
  Expected subtree:
  - `experiments/common/`
  - `experiments/topo/`
  - `experiments/smpl/`
- `configs/`: reusable configuration artifacts
- `private_data/`: private or upstream-governed data assets not intended for source bundle packaging
- `outputs/`: retained exported artifacts needed later
  Expected subtree:
  - `outputs/common/`
  - `outputs/topo/`
  - `outputs/smpl/`
- `stuff/`: scratch/archive overflow
- `tools/`: repository tooling namespace
  Expected subtree:
  - `tools/common/`
  - `tools/topo/`
  - `tools/smpl/`
  Root-level convenience scripts are also allowed in `tools/` when direct access is preferable for repository operators.
- `benchmarks/`: benchmark namespace
  Expected subtree:
  - `benchmarks/common/`
  - `benchmarks/topo/`
  - `benchmarks/smpl/`
- `papers/`: writing and publication workspace
- `_bundles/`: generated source zip bundles and similar packaging artifacts

## 2. Docs Layout

The `docs/smpl/` tree is organized as:
- `governance/`
- `notation/`
- `standards/`
- `specs/`
- `objects/`
- `theory/`
- `implementation/`
- `status/`
- `reports/`
- `practical/`

Top-level files inside `docs/smpl/` are reserved for high-level entry documents such as:
- `architecture.md`
- `documentation_governance.md`
- `project_overview.md`

## 3. Authority Split

Use this order:
1. `docs/smpl/specs/`
2. `contracts/smpl/`
3. `docs/smpl/objects/`
4. `docs/smpl/theory/`
5. `docs/smpl/implementation/`
6. `docs/smpl/status/`
7. `docs/smpl/reports/`
8. `docs/smpl/practical/`

Operational guarantees belong in `contracts/smpl/`, not under `docs/smpl/`.

## 4. Placement Rules

- semantic definitions and invariants go in `docs/smpl/specs/`
- runtime/API obligations go in `contracts/smpl/`
- named runtime catalogs go in `docs/smpl/objects/`
- derivations and explanations go in `docs/smpl/theory/`
- workflows, benchmarks, implementation mapping, and methodology notes go in `docs/smpl/implementation/`
- roadmaps, current-state summaries, and active TODOs go in `docs/smpl/status/`
- produced functionality reports, coverage reports, parity reports, benchmark result packages, and official validation reports go in `docs/smpl/reports/`
- user-facing practical usage guides, runtime recipes, operator checklists, and how-to material go in `docs/smpl/practical/`
- structural/process rules go in `docs/smpl/governance/` or this file

## 5. Status Layer

The `docs/smpl/status/` folder is the lightweight planning/status layer.

It should answer:
- what is done now
- what is next in the short term
- what remains in the long term

## 6. Current Repo Mapping

Current implementation-oriented docs:
- `docs/smpl/implementation/README.md`
- `docs/smpl/implementation/api_runtime_implementation.md`
- `docs/smpl/implementation/api_usage.md`
- `docs/smpl/implementation/benchmark_results.md`
- `docs/smpl/implementation/cache_padding_policy_implementation.md`
- `docs/smpl/implementation/dependency_registry.md`
- `docs/smpl/implementation/implementation_overview.md`
- `docs/smpl/implementation/parity_workflow.md`
- `docs/smpl/implementation/testing_benchmarks_examples_implementation.md`

Current notation docs:
- `docs/smpl/notation/README.md`
- `docs/smpl/notation/glossary.md`
- `docs/smpl/notation/mode_glossary.md`

Current object docs:
- `docs/smpl/objects/README.md`
- `docs/smpl/objects/runtime_objects.md`

Current theory docs:
- `docs/smpl/theory/README.md`
- `docs/smpl/theory/lbs_math_notes.md`
- `docs/smpl/theory/runtime_mode_methodology.md`

Current status docs:
- `docs/smpl/status/README.md`
- `docs/smpl/status/api_test_benchmark_status.md`
- `docs/smpl/status/mode1_status.md`
- `docs/smpl/status/mode2_roadmap.md`
- `docs/smpl/status/roadmap.md`

Current reports docs:
- `docs/smpl/reports/README.md`
- `docs/smpl/reports/api_test_benchmark_inventory.md`

Current practical docs:
- `docs/smpl/practical/README.md`
- `docs/smpl/practical/api_usage.md`

Current specs docs:
- `docs/smpl/specs/README.md`
- `docs/smpl/specs/api_runtime_surface.md`
- `docs/smpl/specs/runtime_specs.md`

Current contract docs:
- `contracts/smpl/api_contract.md`

Operational note:
- retained benchmark harnesses live under `benchmarks/smpl/` and their validation is controlled through pytest benchmark-marked tests
- user-facing notebooks and demos live under `examples/smpl/`

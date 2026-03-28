# smplJAX Status Overview

## Done

- Core JAX runtime, optimized runtime, and factory APIs are in place.
- Validation, conversion tooling, and diagnostics are implemented.
- Cross-platform CI, gated parity workflow, and benchmark workflow are defined.
- Optional `viser` and `pyvista` visualization workflows are integrated.
- Mode 1 is implemented as the supported end-to-end optimization path, including initialization, optimization, snapshot and artifact export, and visualization payloads.
- The current Mode 1 surface covers practical full-runtime use for SMPL-family models supported by the repository loaders, including optional expression, face, and hand parameter paths when present in model data.
- Mode 2 is implemented as the staged optimization path with explicit phase summaries and exported artifacts.
- Mode 3 is implemented as a soft parameter-group routing workflow over a fixed set of SMPL parameter blocks.
- Mode 4 is implemented as a straight-through parameter-group routing workflow with hard forward activations and soft backward gradients.
- Mode 5 is implemented as a controller-driven multi-cycle workflow that combines routing surrogates with explicit refinement and state-transfer summaries.

## Active

- Remote workflow validation on GitHub Actions.
- Benchmark artifact publication and baseline result collection.
- Ongoing documentation cleanup into the new governance structure.
- Shared bridge and workflow docs are being aligned with the now-implemented Mode 2 through Mode 5 SMPL surface.

## Next

- Keep aligning remaining docs with the repo-architecture template.
- Continue reducing local-environment-specific test friction.
- Publish benchmark artifacts for concrete machines/backends.
- Harden CPU versus GPU benchmark baselines and runtime-policy recommendations for the staged and routing workflows.

# Example Notebook Standard

## Purpose

This standard defines the required structure for notebooks under `examples/common/`.

## Required Content

Common notebooks should:

- demonstrate shared entry points rather than backend internals
- say whether the notebook is JAX-oriented, NumPy-oriented, cold-start-oriented, or warm-path-oriented
- show the owning public import path explicitly
- emit artifacts and diagnostics in the same style used elsewhere in the common track
- avoid hidden setup that changes benchmark interpretation
- show how the common layer relates to the backend provider beneath it

## Required Sections

Each canonical common notebook should include:

- scope
- environment and backend notes
- object or runtime construction
- direct usage
- production or repeated-call pattern
- diagnostics or artifact summary
- benchmark or performance summary when relevant

## Source Tree Rule

Common notebooks should run against the repo source tree and use repo-relative paths for assets and outputs.

# Experiment Layout Standard

## Purpose

This standard defines where exploratory and retained shared-track artifacts should live.

## Rules

- canonical examples live under `examples/common/`
- exploratory work lives under `experiments/common/`
- retained promoted outputs should prefer `outputs/common/`
- pytest-managed benchmark and test artifacts should use temporary directories unless explicitly promoted
- docs and notebooks should not depend on stray files in scratch output directories

# Pytest Test Naming Standard

## Purpose

This standard defines naming and placement expectations for pytest coverage relevant to the shared track.

## Rules

- pytest files should use `test_*.py`
- benchmark harness tests should use the `benchmark` marker
- tests that validate a benchmark script should say so clearly in the test name
- common-surface tests belong in `tests/common/`
- backend-specific benchmark harness validation belongs under the owning backend test tree

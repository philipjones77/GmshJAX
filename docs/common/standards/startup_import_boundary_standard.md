# Startup Import Boundary Standard

## Purpose

This standard defines import-boundary rules for startup-sensitive shared modules.

## Rules

- package roots and routing modules must stay import-light
- benchmark, docs, visualization, or optional dependency helpers should not be pulled onto the default import path
- startup-sensitive tests should target the public import surfaces, not only internal helpers
- import-boundary regressions should be treated as real shared-track regressions

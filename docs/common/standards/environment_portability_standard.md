# Environment Portability Standard

## Purpose

This standard defines portability expectations for scripts, docs, examples, and tests in the shared track.

## Rules

- use repo-relative paths rather than host-specific absolute paths in committed docs and examples
- scripts and notebooks should work from the repo source tree
- CPU and GPU assumptions should be stated explicitly when relevant
- Linux and WSL should remain first-class execution environments
- optional dependencies and visualization stacks should fail clearly rather than implicitly at import time

# Core Scalar Service Calling Standard

This filename is retained for taxonomy parity with the reference standards set. In this repo it governs the analogous low-level shared service surfaces rather than scalar math functions.

## Purpose

This standard defines calling expectations for small, reusable common services such as IO, diagnostics, movement application, bridge building, and mesh repair dispatch.

## Rules

- service helpers should be narrowly scoped and explicit about inputs and outputs
- hot-path helpers should avoid unnecessary Python-side materialization
- JSON-producing helpers should stay deterministic and serializable
- service helpers should not silently pull in heavyweight backend modules unless that is the purpose of the call

# API Surface Kinds Standard

## Purpose

This standard defines the public surface taxonomy for the common track.

## Surface Kinds

Shared public surfaces should be described as one of:

- registry surface: backend or capability discovery
- adapter surface: `common.topo` and `common.smpl` wrappers over provider implementations
- helper surface: shared IO, diagnostics, movement, mesh repair, and similar support utilities
- runtime surface: stateful or structured runtime objects such as the NumPy mesh runtime
- bridge surface: payload or export builders for RF77 and related external consumers
- artifact surface: save, export, report, or diagnostics emission helpers

## Rules

- every common-facing document should make clear which surface kind it is describing
- adapter surfaces should remain thin and preserve provider ownership
- helper surfaces should be backend-neutral unless a backend-specific adapter is the point of the API
- runtime surfaces should expose inspectable diagnostics and stable metadata
- bridge and artifact surfaces should describe output shape and persistence expectations explicitly

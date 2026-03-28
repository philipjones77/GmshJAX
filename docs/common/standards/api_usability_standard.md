# API Usability Standard

## Purpose

This standard defines the usability expectations for common-facing APIs and the way examples and practical docs should teach them.

## Rules

- common-facing imports should be easy to discover
- the owning backend should remain visible when it matters
- examples should show the intended calling pattern rather than only toy one-offs
- repeated-call surfaces should explain reuse, caching, and diagnostics expectations
- user-facing docs should prefer the shared import path when the goal is cross-backend interoperability

## Shared-Layer Interpretation

The common track should feel coherent without pretending that Topo and SMPL are internally the same system.

# JAX Surface Policy Standard

## Purpose

This standard defines how JAX-oriented and non-JAX-oriented surfaces coexist in the repository.

## Rules

- JAX-oriented public surfaces should remain primarily in `topojax`, `smpljax`, and their common adapters
- NumPy-native shared surfaces should be clearly labeled as non-AD array workflows
- the common layer should not silently convert every path into JAX or every path into NumPy
- docs should state whether a surface is JAX-first, NumPy-first, or bridge-oriented

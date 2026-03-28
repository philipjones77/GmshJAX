# IO Module Implementation

`src/common/io.py` provides the shared filesystem write path.

Current contract:

- create parent directories automatically
- write text, JSON, CSV, and NPZ atomically
- support atomic file copy for artifact promotion

The common layer uses these helpers to keep exported artifacts consistent across benchmarks, diagnostics, and NumPy runtime saves.

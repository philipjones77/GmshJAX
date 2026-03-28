# Diagnostics Module Implementation

`src/common/diagnostics.py` provides backend-neutral serialization helpers and a simple append-only logger.

Key behaviors:

- converts dataclasses, named tuples, NumPy arrays, and scalar types to JSON-safe values
- writes stable runtime diagnostics payloads
- provides a JSONL logger for long-running benchmark or workflow events

The module is kept generic so both the JAX-facing and NumPy-facing paths can reuse it.

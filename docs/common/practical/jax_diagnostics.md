# Common JAX Diagnostics

The shared layer does not replace backend-specific diagnostics, but it should preserve comparable reporting structure.

Practical guidance:

- use backend-native diagnostics for deep algorithm debugging
- use common diagnostics payloads for cross-backend reporting and artifact export
- keep JSON-safe summaries stable so benchmark and example outputs remain comparable

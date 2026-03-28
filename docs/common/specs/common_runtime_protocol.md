# Common Runtime Protocol

The shared runtime protocol defines the minimum expectations for common-facing mesh and backend surfaces.

Protocol points:

- a backend can be discovered and reported through the backend registry
- a mode-aware bridge can be requested through the common surface
- diagnostics should be representable as JSON-safe payloads
- artifact export should use stable file-writing helpers
- mesh movement should be representable through the canonical transform structure
- NumPy-native workflows should preserve mode, metadata, and visualization compatibility even without AD

Backend-specific implementations may expose richer behavior, but they should not violate this shared contract.

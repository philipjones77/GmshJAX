# Backends Module Implementation

`src/common/backends.py` defines the registry for the shared track.

Responsibilities:

- enumerate backends
- report implemented modes per backend
- dispatch bridge building
- dispatch print-mesh repair helpers

It is intentionally a small routing layer and should stay free of heavy backend imports until a backend-specific action is requested.

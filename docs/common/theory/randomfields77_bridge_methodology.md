# RandomFields77 Bridge Methodology

The RF77 methodology in the common track is to expose one stable, mode-aware payload shape while preserving backend ownership of the actual mesh-generation logic.

Conceptual rules:

- the bridge surface is mode-aware, not backend-agnostic to the point of losing meaning
- payloads should keep enough metadata to explain how the mesh was produced
- NumPy runtimes should be able to participate in the same export story for non-AD workflows

# NumPy Runtime Wrappers Implementation

The NumPy runtime acts as a wrapper over existing mesh topology, visualization, and RF77-bridge capabilities while storing its state as NumPy arrays.

This wrapper layer should preserve:

- mode identity
- element-kind information
- metadata and mode payloads
- compatible diagnostics and export behavior

It should not attempt to emulate backend-specific AD behavior.

# Common Runtime Objects

Primary shared objects and structured surfaces:

- `BackendSpec` and `BackendName` from `common.backends`
- `MeshMovementTransform` from `common.movement`
- `NumpyMeshRuntime` and `NumpyMeshDiagnostics` from `common.numpy_mesh`
- `MeshRepairResult` from `common.mesh_repair`
- `DiagnosticsLogger` payload helpers from `common.diagnostics`

Adapter-oriented structured objects exposed through the common layer include:

- Topo workflow domain and run objects re-exported from `common.topo`
- SMPL provision, workflow, and optimization result objects re-exported from `common.smpl`

These objects define the shared runtime vocabulary for backend discovery, movement, diagnostics, export, and bridge behavior.

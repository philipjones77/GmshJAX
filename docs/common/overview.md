# Overview

The `common` track is the shared contract layer for the repository.

It defines the standard surface that `topo` and `smpl` must obey, while each backend keeps its own optimized implementation strategy.
It also owns backend-neutral utilities such as shared IO, diagnostics, the canonical mesh-movement transform structure, mesh repair, the backend registry, and the NumPy-native mesh runtime used for RF77-style array workflows.

The detailed content is split by category under `docs/common/` so the shared layer has the same document shape as the reference docs tree in `arbplusJAX`.

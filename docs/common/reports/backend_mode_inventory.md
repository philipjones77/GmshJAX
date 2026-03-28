# Backend Mode Inventory

Current shared backend inventory:

| Backend | Package | Modes reported through `common` |
| --- | --- | --- |
| `topo` | `topojax` | 1 through 5 |
| `smpl` | `smpljax` | 1 through 5 |

Inventory notes:

- the common layer reports capability through `src/common/backends.py`
- backend-specific implementation details remain documented under `docs/topo/` and `docs/smpl/`
- the NumPy runtime mirrors the same mode vocabulary for shared array workflows

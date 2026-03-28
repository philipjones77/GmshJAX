# Repository Licensing

This document defines the licensing structure for the three-track `TopoSmplJAX` repository.

## Root Rule

Repository-authored code and documentation are MIT-licensed under the root [LICENSE](../../LICENSE) unless a narrower notice says otherwise.

## Domain Mapping

Topo slice:
- `src/topojax/`
- `docs/topo/`
- `contracts/topo/`
- `examples/topo/`
- `tools/topo/`
- `tests/topo/`

SMPL slice:
- `src/smpljax/`
- `docs/smpl/`
- `contracts/smpl/`
- `examples/smpl/`
- `tools/smpl/`
- `tests/smpl/`

Shared slice:
- `src/common/`
- `benchmarks/common/`
- `examples/common/`
- `docs/common/`
- `tools/common/`
- `tests/common/`

The authoritative path matrix lives in [../../LICENSES/CODE_AND_CONTENT.md](../../LICENSES/CODE_AND_CONTENT.md).

## Asset Boundary

`private_data/smpl/` is not a blanket MIT asset bucket. Repository-authored metadata and small synthetic fixtures may be MIT, but third-party SMPL-family model files remain governed by their upstream terms.

See [../../LICENSES/ASSET_BOUNDARY.md](../../LICENSES/ASSET_BOUNDARY.md).

## Third-Party Notices

Some repository areas reference or adapt upstream work from:

- Gmsh
- `smplx`
- SMPL-X-associated tooling such as the Meshcapade SMPL Blender Add-on

Those acknowledgments and upstream-license boundaries are tracked in [../../LICENSES/THIRD_PARTY_NOTICES.md](../../LICENSES/THIRD_PARTY_NOTICES.md).

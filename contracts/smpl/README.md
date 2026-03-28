# Contracts

This folder contains binding runtime and API guarantees for `smplJAX`.

Contracts refine `docs/smpl/specs/` operationally and should not invent semantics that are absent from the specs layer.

Current contract focus:

- stable runtime and API guarantees for the implemented Mode 1 through Mode 5 SMPL workflow surface
- shared expectations for staged optimization, parameter-routing surrogates, and dynamic-controller workflows

For current status, see:

- `docs/smpl/status/mode1_status.md`
- `docs/smpl/status/mode2_roadmap.md`

Licensing:

- repository-authored contract text in this folder is MIT-licensed under the repository root license
- upstream SMPL-family model assets are outside the scope of this contract-folder license notice

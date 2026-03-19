# Third-Party Notices

This repository contains repository-authored code under the root MIT license, but some parts of the project are derived from or designed around upstream third-party software and assets. Those upstream rights remain with their respective authors and licensors.

## Gmsh

The Topo slice includes Gmsh-inspired APIs and, in some cases, code adapted from Gmsh sources or algorithms.

Upstream project:
- Gmsh, by Christophe Geuzaine and Jean-Francois Remacle
- Official site: https://gmsh.info/

Upstream licensing:
- Gmsh is copyrighted by its authors and distributed under the GNU General Public License, version 2 or later, with an exception for easier linking with external libraries.
- If a repository file includes direct code copied or adapted from Gmsh, that file should carry an adjacent notice identifying the upstream source.

Maintainer rule:
- Do not represent Gmsh-derived code as wholly original repository-authored code.
- Preserve upstream attribution and comply with the applicable Gmsh license terms for copied or adapted portions.

## SMPL / SMPL-X

This repository acknowledges the upstream SMPL and SMPL-X projects, including the `smplx` reference implementation maintained by the Max Planck group.

Upstream projects:
- SMPL: https://smpl.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- `smplx` reference code: https://github.com/vchoutas/smplx

Upstream licensing:
- SMPL-family model files and associated software are not relicensed by this repository.
- The official SMPL and SMPL-X sites require registration and agreement to their upstream terms for access to model downloads.
- The `smplx` repository is distributed under the Max Planck non-commercial scientific research license.

Maintainer rule:
- Keep SMPL-family model assets under `private_data/smpl/` unless redistribution is clearly permitted.
- Preserve upstream attribution when this repository references or adapts behavior from `smplx`.

## SMPL-X Toolbox / Associated Tooling

This repository also acknowledges SMPL-X-associated tooling used as implementation reference material, including the Meshcapade SMPL Blender Add-on.

Reference project:
- Meshcapade SMPL Blender Add-on: https://github.com/Meshcapade/SMPL_blender_addon

Upstream licensing:
- The Blender add-on code is published under GPL-3.0 according to the upstream repository.
- SMPL-family model files used with that tooling remain under their separate upstream model licenses.

Maintainer rule:
- When code or behavior is derived from SMPL-X-associated tooling, add a local note naming the upstream project and keep the upstream license boundary explicit.

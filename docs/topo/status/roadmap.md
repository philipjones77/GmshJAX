# Roadmap

Status: active
Date: 2026-03-15

## Documentation Migration

- adopt the governed `docs/common/`, `docs/topo/`, and `docs/smpl/` layout with domain-specific `contracts/` folders
- keep TopoJAX documents under `docs/topo/`
- move authored workflow and milestone notes into `docs/topo/implementation/` over time

## Near-Term Priorities

- add formal mesh and object semantics under `docs/topo/specs/`
- keep overview and milestone notes under the `docs/topo/` subtree
- extend `contracts/topo/` only when additional surfaces are intentionally treated as stable

## Mode 5 Dynamic Work

- keep refining the relaxed `fully-dynamic-remeshing` implementation for triangle and tetra workflows
- strengthen transfer quality, controller behavior, and runtime coverage beyond the first released prototype
- keep the mode-5 documents explicit about non-goals and approximation strategy

See `mode5_roadmap.md` for the concrete milestone plan and code targets.

## Mode 1 Closeout

- treat the current Mode 1 1D, 2D, and 3D fixed-topology surface as the practical baseline for create or import, optimize, export, and view workflows
- keep extending documentation and validation around the now-public domain initializers and workflow entry points
- prefer future investment in transfer, surrogate, and controller infrastructure over chasing broad parity inside Mode 1 itself

See `mode1_2d_3d_status.md` for the current-state summary and next-step implementation plan.

## Deferred TODO: Product Domains And Lazy Mode 1 Initialization

- keep the first practical Mode 1 path focused on discrete mesh creation followed by fixed-topology coordinate motion, AD, export, and optional visualization
- treat meshing as a pre-AD initialization stage; do not differentiate through triangulation, tensorization, clipping, or other topology-building steps in Mode 1
- keep native Gmsh as the default visualization backend, while preserving backend-dispatched support for Matplotlib, PyVista, Viser, and later viewers
- add a future lazy domain-to-mesh realization layer so bounded-domain or product-domain specs can be instantiated only when a mesh is actually needed
- revisit a unified domain-spec API for practical 2D and 3D base domains, including squares, cubes, mapped blocks, polygonal domains, spheres, and implicit bounded domains
- defer general Cartesian-product domain composition for now; if RandomFields already owns domain algebra and Cartesian products, prefer using that layer instead of duplicating it inside TopoJAX
- if TopoJAX later needs direct product-domain support, build it from separate factor domains or factor meshes and then realize the combined mesh combinatorially
- keep any future product-domain implementation explicit about the distinction between intrinsic domain dimension and ambient embedding dimension
- support Gmsh-backed viewing only for realized Euclidean embeddings that fit the viewer assumptions; higher-dimensional product embeddings would need projection or a different visualization path

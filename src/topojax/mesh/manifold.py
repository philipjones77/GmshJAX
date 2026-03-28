"""Manifold parameterization and deformation maps.

This module keeps the historical TopoJAX names, but the canonical transform
structure now lives in `common.movement`.
"""

from __future__ import annotations

from common.movement import MeshMovementTransform as DeformationParams
from common.movement import apply_mesh_movement_jax as apply_deformation

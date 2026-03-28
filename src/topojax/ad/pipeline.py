"""Compiled AD pipelines for fixed-topology moving meshes."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from common.movement import (
    default_mesh_movement_transform,
    pack_mesh_movement_transform,
    unpack_mesh_movement_vector,
)
from topojax.mesh.manifold import DeformationParams, apply_deformation
from topojax.mesh.operators import mesh_quality_energy
from topojax.mesh.topology import MeshTopology
from topojax.model import MeshModel
from topojax.runtime import jax_float_dtype


def _pack_params(params: DeformationParams) -> jnp.ndarray:
    return jnp.asarray(pack_mesh_movement_transform(params))


def _unpack_params(vec: jnp.ndarray) -> DeformationParams:
    return DeformationParams(*unpack_mesh_movement_vector(vec, point_dim=2, backend="jax"))


def build_parametric_quality_value_and_grad(topology: MeshTopology, reference_points: jnp.ndarray):
    """Compile value/grad wrt deformation params with static topology and point count."""

    def energy_from_vec(theta: jnp.ndarray) -> jnp.ndarray:
        params = _unpack_params(theta)
        points = apply_deformation(reference_points, params)
        return mesh_quality_energy(points, topology)

    return jax.jit(jax.value_and_grad(energy_from_vec))


def build_model_parametric_quality_value_and_grad(model: MeshModel):
    """Compile value/grad using model topology and current points as reference."""
    return build_parametric_quality_value_and_grad(model.topology, model.state.points)


def default_params() -> DeformationParams:
    """Reasonable initialization for smooth deformations."""
    return DeformationParams(*default_mesh_movement_transform(point_dim=2, dtype=jax_float_dtype(), backend="jax"))


def default_param_vector() -> jnp.ndarray:
    """Packed default parameter vector."""
    return _pack_params(default_params())

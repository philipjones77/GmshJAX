"""Shared mesh-movement transform structures and apply helpers."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "MeshMovementTransform",
    "apply_mesh_movement",
    "apply_mesh_movement_jax",
    "apply_mesh_movement_numpy",
    "default_mesh_movement_transform",
    "mesh_movement_vector_size",
    "pack_mesh_movement_transform",
    "unpack_mesh_movement_vector",
]


class MeshMovementTransform(NamedTuple):
    """Low-dimensional transform for moving a mesh without changing topology.

    `shear` supports two shapes:
    - `(2,)` for the legacy 2D `[xy, yx]` shear parameterization
    - `(D, D)` for a general additive shear matrix in `D` dimensions

    `bend` is per-axis sinusoidal bending. In 2D this preserves the historical
    semantics used by TopoJAX:
    - `bend[0]` bends x as a function of y
    - `bend[1]` bends y as a function of x
    """

    translation: Any
    scale: Any
    shear: Any
    bend: Any


def _coerce_vector(xp, value: Any, *, size: int, dtype, name: str):
    arr = xp.asarray(value, dtype=dtype)
    if arr.ndim != 1 or int(arr.shape[0]) != size:
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr


def _as_shear_matrix(xp, shear: Any, *, point_dim: int, dtype):
    arr = xp.asarray(shear, dtype=dtype)
    if arr.ndim == 1:
        if point_dim != 2 or int(arr.shape[0]) != 2:
            raise ValueError("1D shear is only supported for 2D transforms with shape (2,)")
        zero = xp.asarray(0.0, dtype=dtype)
        return xp.asarray([[zero, arr[0]], [arr[1], zero]], dtype=dtype)
    if arr.ndim == 2 and arr.shape == (point_dim, point_dim):
        return arr
    raise ValueError(f"shear must have shape (2,) or ({point_dim}, {point_dim}), got {arr.shape}")


def _bend_term(xp, points, bend):
    point_dim = int(points.shape[1])
    if point_dim == 0:
        return xp.zeros_like(points)
    terms = []
    for axis in range(point_dim):
        source_axis = (axis + 1) % point_dim
        terms.append(bend[axis] * xp.sin(2.0 * xp.pi * points[:, source_axis]))
    return xp.stack(terms, axis=1)


def _apply_mesh_movement_impl(xp, reference_points, transform: MeshMovementTransform):
    pts = xp.asarray(reference_points)
    if pts.ndim != 2:
        raise ValueError("reference_points must be rank-2")
    point_dim = int(pts.shape[1])
    dtype = pts.dtype

    translation = _coerce_vector(xp, transform.translation, size=point_dim, dtype=dtype, name="translation")
    scale = _coerce_vector(xp, transform.scale, size=point_dim, dtype=dtype, name="scale")
    bend = _coerce_vector(xp, transform.bend, size=point_dim, dtype=dtype, name="bend")
    shear_matrix = _as_shear_matrix(xp, transform.shear, point_dim=point_dim, dtype=dtype)

    scaled = pts * scale[None, :]
    shear_term = pts @ xp.swapaxes(shear_matrix, 0, 1)
    bend_term = _bend_term(xp, pts, bend)
    return scaled + shear_term + bend_term + translation[None, :]


def apply_mesh_movement_jax(reference_points: jax.Array, transform: MeshMovementTransform) -> jax.Array:
    """Apply a mesh movement transform using JAX arrays and primitives."""
    return _apply_mesh_movement_impl(jnp, reference_points, transform)


def apply_mesh_movement_numpy(reference_points: np.ndarray, transform: MeshMovementTransform) -> np.ndarray:
    """Apply a mesh movement transform using NumPy arrays."""
    return _apply_mesh_movement_impl(np, reference_points, transform)


def apply_mesh_movement(reference_points, transform: MeshMovementTransform):
    """Dispatch mesh movement based on the input array backend."""
    if isinstance(reference_points, jax.Array):
        return apply_mesh_movement_jax(reference_points, transform)
    return apply_mesh_movement_numpy(np.asarray(reference_points), transform)


def default_mesh_movement_transform(
    *,
    point_dim: int = 2,
    dtype=None,
    backend: str = "numpy",
) -> MeshMovementTransform:
    """Return an identity movement transform for the requested dimension/backend."""
    if point_dim <= 0:
        raise ValueError("point_dim must be positive")
    xp = jnp if backend == "jax" else np
    if dtype is None:
        dtype = jnp.float32 if xp is jnp else np.float32
    translation = xp.zeros((point_dim,), dtype=dtype)
    scale = xp.ones((point_dim,), dtype=dtype)
    bend = xp.zeros((point_dim,), dtype=dtype)
    shear = xp.zeros((2,), dtype=dtype) if point_dim == 2 else xp.zeros((point_dim, point_dim), dtype=dtype)
    return MeshMovementTransform(translation=translation, scale=scale, shear=shear, bend=bend)


def mesh_movement_vector_size(point_dim: int) -> int:
    if point_dim <= 0:
        raise ValueError("point_dim must be positive")
    shear_size = 2 if point_dim == 2 else point_dim * point_dim
    return point_dim + point_dim + shear_size + point_dim


def pack_mesh_movement_transform(transform: MeshMovementTransform):
    """Pack a transform to a flat vector.

    For 2D transforms, a `(2,)` shear vector is preserved to keep parity with the
    historical TopoJAX parameterization.
    """
    if isinstance(transform.translation, jax.Array):
        translation = jnp.asarray(transform.translation)
        point_dim = int(translation.shape[0])
        scale = jnp.asarray(transform.scale)
        bend = jnp.asarray(transform.bend)
        shear = jnp.asarray(transform.shear)
        shear_flat = shear if point_dim == 2 and shear.shape == (2,) else jnp.reshape(shear, (-1,))
        return jnp.concatenate(
            [translation, scale, shear_flat, bend],
            axis=0,
        )
    translation = np.asarray(transform.translation)
    point_dim = int(translation.shape[0])
    scale = np.asarray(transform.scale)
    bend = np.asarray(transform.bend)
    shear = np.asarray(transform.shear)
    shear_flat = shear if point_dim == 2 and shear.shape == (2,) else shear.reshape(-1)
    return np.concatenate([translation, scale, shear_flat, bend], axis=0)


def unpack_mesh_movement_vector(vec, *, point_dim: int = 2, backend: str = "jax") -> MeshMovementTransform:
    """Unpack a flat vector into a mesh movement transform."""
    xp = jnp if backend == "jax" else np
    flat = xp.asarray(vec)
    expected = mesh_movement_vector_size(point_dim)
    if flat.ndim != 1 or int(flat.shape[0]) != expected:
        raise ValueError(f"Expected vector of shape ({expected},), got {flat.shape}")

    offset = 0
    translation = flat[offset : offset + point_dim]
    offset += point_dim
    scale = flat[offset : offset + point_dim]
    offset += point_dim
    shear_size = 2 if point_dim == 2 else point_dim * point_dim
    shear_flat = flat[offset : offset + shear_size]
    offset += shear_size
    bend = flat[offset : offset + point_dim]
    shear = shear_flat if point_dim == 2 else xp.reshape(shear_flat, (point_dim, point_dim))
    return MeshMovementTransform(translation=translation, scale=scale, shear=shear, bend=bend)

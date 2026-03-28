from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from common import (
    MeshMovementTransform,
    apply_mesh_movement_jax,
    apply_mesh_movement_numpy,
    default_mesh_movement_transform,
    mesh_movement_vector_size,
    pack_mesh_movement_transform,
    unpack_mesh_movement_vector,
)
from common import numpy_mesh
from topojax.mesh.topology import unit_square_tri_mesh


def test_common_mesh_movement_transform_preserves_legacy_2d_semantics() -> None:
    points_np = np.asarray([[0.0, 0.0], [1.0, 0.5], [0.25, 1.0]], dtype=np.float32)
    transform = MeshMovementTransform(
        translation=np.array([0.1, -0.2], dtype=np.float32),
        scale=np.array([1.05, 0.95], dtype=np.float32),
        shear=np.array([0.04, -0.01], dtype=np.float32),
        bend=np.array([0.02, 0.03], dtype=np.float32),
    )

    moved_np = apply_mesh_movement_numpy(points_np, transform)
    moved_jax = np.asarray(apply_mesh_movement_jax(jnp.asarray(points_np), transform))

    x = points_np[:, 0]
    y = points_np[:, 1]
    expected = np.stack(
        [
            transform.scale[0] * x + transform.shear[0] * y + transform.bend[0] * np.sin(2.0 * np.pi * y) + transform.translation[0],
            transform.scale[1] * y + transform.shear[1] * x + transform.bend[1] * np.sin(2.0 * np.pi * x) + transform.translation[1],
        ],
        axis=-1,
    )

    assert np.allclose(moved_np, expected)
    assert np.allclose(moved_jax, expected)


def test_common_mesh_movement_transform_supports_pack_unpack_and_3d() -> None:
    transform2 = default_mesh_movement_transform(point_dim=2, dtype=np.float32, backend="numpy")
    packed = pack_mesh_movement_transform(transform2)
    unpacked = unpack_mesh_movement_vector(packed, point_dim=2, backend="numpy")
    assert packed.shape == (mesh_movement_vector_size(2),)
    assert np.allclose(np.asarray(unpacked.translation), np.asarray(transform2.translation))
    assert np.allclose(np.asarray(unpacked.scale), np.asarray(transform2.scale))

    points3 = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]], dtype=np.float32)
    transform3 = MeshMovementTransform(
        translation=np.array([0.1, -0.1, 0.2], dtype=np.float32),
        scale=np.array([1.0, 0.9, 1.1], dtype=np.float32),
        shear=np.array(
            [
                [0.0, 0.02, 0.0],
                [0.0, 0.0, -0.03],
                [0.01, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        bend=np.array([0.02, 0.01, -0.02], dtype=np.float32),
    )
    moved3 = apply_mesh_movement_numpy(points3, transform3)
    assert moved3.shape == points3.shape
    assert np.all(np.isfinite(moved3))
    assert mesh_movement_vector_size(3) == 18


def test_numpy_mesh_runtime_apply_transform_preserves_mode_and_connectivity() -> None:
    topology, points = unit_square_tri_mesh(4, 3)
    runtime = numpy_mesh.create_mode5_runtime(
        points,
        topology.elements,
        controller_history=[{"cycle": 0, "reason": "seed"}],
        transfer_history=[{"cycle": 0, "transferred": True}],
    )
    transform = MeshMovementTransform(
        translation=np.array([0.2, -0.1], dtype=np.float32),
        scale=np.array([1.0, 1.0], dtype=np.float32),
        shear=np.array([0.0, 0.0], dtype=np.float32),
        bend=np.array([0.0, 0.0], dtype=np.float32),
    )
    moved = runtime.apply_transform(transform)

    assert moved.mode == runtime.mode
    assert np.array_equal(moved.elements, runtime.elements)
    assert np.allclose(moved.points, runtime.points + np.array([0.2, -0.1], dtype=np.float32))

import pytest

import jax.numpy as jnp

from topojax.mesh.generators import unit_square_points
from topojax.mesh.topology import mapped_quad_mesh, unit_cube_tet_mesh, unit_square_quad_mesh


def test_unit_square_points_shape() -> None:
    points = unit_square_points(4, 3)
    assert points.shape == (12, 2)


def test_unit_square_quad_mesh_shape() -> None:
    topo, points = unit_square_quad_mesh(5, 4)
    assert points.shape == (20, 2)
    assert topo.elements.shape == (12, 4)


def test_unit_cube_tet_mesh_shape() -> None:
    topo, points = unit_cube_tet_mesh(4, 3, 2)
    assert points.shape == (24, 3)
    assert topo.elements.shape[1] == 4


def test_mapped_quad_mesh_maps_reference_grid() -> None:
    def bilinear_map(xi_eta: jnp.ndarray) -> jnp.ndarray:
        x = xi_eta[:, 0]
        y = xi_eta[:, 1]
        return jnp.stack([x + 0.2 * y, y + 0.1 * x * (1.0 - x)], axis=1)

    topo, points = mapped_quad_mesh(bilinear_map, 5, 4)

    assert points.shape == (20, 2)
    assert topo.elements.shape == (12, 4)
    assert jnp.allclose(points[0], jnp.array([0.0, 0.0], dtype=points.dtype))
    assert jnp.allclose(points[-1], jnp.array([1.2, 1.0], dtype=points.dtype))


def test_mapped_quad_mesh_rejects_wrong_shape() -> None:
    def bad_map(xi_eta: jnp.ndarray) -> jnp.ndarray:
        return xi_eta[:, 0]

    with pytest.raises(ValueError, match="shape"):
        mapped_quad_mesh(bad_map, 4, 3)


def test_mapped_quad_mesh_rejects_inverted_quads() -> None:
    def inverted_map(xi_eta: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([1.0 - xi_eta[:, 0], xi_eta[:, 1]], axis=1)

    with pytest.raises(ValueError, match="non-positive-area"):
        mapped_quad_mesh(inverted_map, 4, 3)

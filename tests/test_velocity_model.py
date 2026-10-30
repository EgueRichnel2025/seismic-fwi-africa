import numpy as np
import pytest

from seismic.model.velocity_model import (
    build_synthetic_velocity_model,
)


def _index_nearest(axis: np.ndarray, value: float) -> int:
    """Retourne l'indice de la coordonnée la plus proche."""
    return int(np.argmin(np.abs(axis - value)))


def test_model_has_expected_physical_dimensions():
    model = build_synthetic_velocity_model()

    assert model.x_m.size == 301
    assert model.z_m.size == 151
    assert model.velocity_m_s.shape == (151, 301)
    assert model.anomaly_mask.shape == (151, 301)

    np.testing.assert_allclose(model.x_m[[0, -1]], [0.0, 600.0])
    np.testing.assert_allclose(model.z_m[[0, -1]], [0.0, 300.0])


def test_grid_spacing_is_two_metres():
    model = build_synthetic_velocity_model()

    np.testing.assert_allclose(np.diff(model.x_m), 2.0)
    np.testing.assert_allclose(np.diff(model.z_m), 2.0)


def test_background_layers_have_expected_velocities():
    model = build_synthetic_velocity_model()

    x_index = _index_nearest(model.x_m, 100.0)

    z_surface = _index_nearest(model.z_m, 20.0)
    z_middle = _index_nearest(model.z_m, 100.0)
    z_substratum = _index_nearest(model.z_m, 250.0)

    assert model.velocity_m_s[z_surface, x_index] == 1200.0
    assert model.velocity_m_s[z_middle, x_index] == 2000.0
    assert model.velocity_m_s[z_substratum, x_index] == 2800.0


def test_anomaly_has_expected_velocity_and_location():
    model = build_synthetic_velocity_model()

    x_index = _index_nearest(model.x_m, 360.0)
    z_index = _index_nearest(model.z_m, 125.0)

    assert model.anomaly_mask[z_index, x_index]
    assert model.velocity_m_s[z_index, x_index] == 1500.0

    # Le point situé hors de l'anomalie conserve la vitesse de la couche.
    outside_x_index = _index_nearest(model.x_m, 200.0)

    assert not model.anomaly_mask[z_index, outside_x_index]
    assert model.velocity_m_s[z_index, outside_x_index] == 2000.0

    assert np.count_nonzero(model.anomaly_mask) > 100


def test_all_velocities_are_finite_and_positive():
    model = build_synthetic_velocity_model()

    assert np.all(np.isfinite(model.velocity_m_s))
    assert np.all(model.velocity_m_s > 0.0)


def test_zero_spacing_is_rejected():
    with pytest.raises(ValueError, match="strictement positif"):
        build_synthetic_velocity_model(dx_m=0.0)


def test_spacing_must_divide_the_physical_extent():
    with pytest.raises(ValueError, match="diviser exactement"):
        build_synthetic_velocity_model(dx_m=7.0)

"""Tests de la fonction d'ondelette de Ricker."""

import numpy as np
import pytest

from seismic.wave.ricker import ricker_wavelet


def test_ricker_has_unit_amplitude_at_center():
    center = 0.06

    values = ricker_wavelet(
        time_s=np.array([center]),
        frequency_hz=25.0,
        center_time_s=center,
    )

    np.testing.assert_allclose(
        values,
        [1.0],
        rtol=0.0,
        atol=1e-14,
    )


def test_ricker_is_symmetric_around_center():
    center = 0.06
    offsets = np.array([0.002, 0.005, 0.010, 0.020])

    before = ricker_wavelet(
        center - offsets,
        frequency_hz=25.0,
        center_time_s=center,
    )
    after = ricker_wavelet(
        center + offsets,
        frequency_hz=25.0,
        center_time_s=center,
    )

    np.testing.assert_allclose(
        before,
        after,
        rtol=1e-12,
        atol=1e-14,
    )


def test_default_center_time_is_one_point_five_over_frequency():
    frequency = 25.0
    center = 1.5 / frequency

    values = ricker_wavelet(
        np.array([center]),
        frequency_hz=frequency,
    )

    np.testing.assert_allclose(
        values,
        [1.0],
        rtol=0.0,
        atol=1e-14,
    )


def test_ricker_returns_finite_values():
    times = np.linspace(0.0, 0.12, 301)

    values = ricker_wavelet(
        times,
        frequency_hz=25.0,
        center_time_s=0.06,
    )

    assert values.shape == times.shape
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize(
    "frequency",
    [0.0, -25.0, np.inf, np.nan],
)
def test_invalid_frequency_is_rejected(frequency):
    with pytest.raises(ValueError, match="fréquence"):
        ricker_wavelet([0.0, 0.01], frequency)


def test_non_finite_time_is_rejected():
    with pytest.raises(ValueError, match="instants"):
        ricker_wavelet(
            [0.0, np.nan],
            frequency_hz=25.0,
        )


def test_non_finite_center_time_is_rejected():
    with pytest.raises(ValueError, match="temps central"):
        ricker_wavelet(
            [0.0, 0.01],
            frequency_hz=25.0,
            center_time_s=np.inf,
        )

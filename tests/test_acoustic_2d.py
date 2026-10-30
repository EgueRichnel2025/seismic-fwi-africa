import numpy as np
import pytest

from seismic.model import build_synthetic_velocity_model
from seismic.wave import simulate_acoustic_wave


def test_wave_propagation_is_finite_and_nonzero():
    model = build_synthetic_velocity_model()

    result = simulate_acoustic_wave(
        model,
        duration_s=0.10,
        dt_s=4.0e-4,
        snapshot_stride=25,
    )

    assert result.cfl_number < 1.0
    assert np.all(np.isfinite(result.snapshots))
    assert np.all(np.isfinite(result.max_abs_per_step))

    # La source doit effectivement produire un champ d'onde.
    assert np.max(np.abs(result.snapshots)) > 0.0


def test_snapshot_dimensions_match_the_velocity_model():
    model = build_synthetic_velocity_model()

    result = simulate_acoustic_wave(
        model,
        duration_s=0.04,
        dt_s=4.0e-4,
        snapshot_stride=10,
    )

    assert result.snapshots.ndim == 3

    assert result.snapshots.shape[1:] == (
        model.z_m.size,
        model.x_m.size,
    )

    assert result.times_s.size == result.snapshots.shape[0]


def test_simulation_records_initial_and_final_times():
    model = build_synthetic_velocity_model()

    result = simulate_acoustic_wave(
        model,
        duration_s=0.04,
        dt_s=4.0e-4,
        snapshot_stride=10,
    )

    assert result.times_s[0] == 0.0
    assert result.times_s[-1] >= 0.04
    assert result.max_abs_per_step[0] == 0.0


def test_unstable_time_step_is_rejected():
    model = build_synthetic_velocity_model()

    with pytest.raises(ValueError, match="Pas temporel instable"):
        simulate_acoustic_wave(
            model,
            duration_s=0.02,
            dt_s=1.0e-3,
        )


def test_source_outside_the_model_is_rejected():
    model = build_synthetic_velocity_model()

    with pytest.raises(ValueError, match="hors du modèle"):
        simulate_acoustic_wave(
            model,
            source_x_m=700.0,
        )


def test_invalid_source_frequency_is_rejected():
    model = build_synthetic_velocity_model()

    with pytest.raises(ValueError, match="fréquence"):
        simulate_acoustic_wave(
            model,
            source_frequency_hz=0.0,
        )


def test_receiver_signals_are_recorded_at_every_time_step():
    model = build_synthetic_velocity_model()

    receiver_positions = np.array([80.0, 100.0, 120.0])

    result = simulate_acoustic_wave(
        model,
        duration_s=0.10,
        dt_s=4.0e-4,
        receiver_x_m=receiver_positions,
        receiver_z_m=4.0,
        snapshot_stride=10,
    )

    assert result.receiver_signals.shape == (
        result.max_abs_per_step.size,
        receiver_positions.size,
    )

    assert result.sample_times_s.size == result.max_abs_per_step.size

    assert result.sample_times_s[0] == 0.0

    assert np.all(np.isfinite(result.receiver_signals))

    # La source doit finir par produire un signal non nul aux récepteurs.
    assert np.max(np.abs(result.receiver_signals)) > 0.0

    np.testing.assert_allclose(
        result.receiver_x_m,
        receiver_positions,
    )
    assert result.receiver_z_m == 4.0


def test_nonpositive_source_sigma_is_rejected():
    model = build_synthetic_velocity_model()

    with pytest.raises(ValueError, match="largeur spatiale"):
        simulate_acoustic_wave(
            model,
            source_sigma_m=0.0,
        )


"""Tests du propagateur d'inversion et du gradient adjoint.

Le test central est celui de la dérivée directionnelle : il compare le
gradient adjoint à une différence finie centrée de la fonction objectif.
C'est la seule vérification qui garantit qu'un optimiseur peut faire
confiance au gradient.
"""

import numpy as np
import pytest

from seismic.model.velocity_model import build_synthetic_velocity_model
from seismic.wave.acoustic_2d import simulate_acoustic_wave
from seismic.inversion.fwi import (
    build_acquisition,
    forward_data,
    gradient_with_respect_to_velocity,
    objective_and_gradient,
    objective_and_gradient_multishot,
    objective_only,
)

# Grille volontairement grossière : les tests doivent rester rapides.
COARSE_DX_M = 20.0
COARSE_DZ_M = 20.0
COARSE_DT_S = 4.0e-3
COARSE_DURATION_S = 0.42
COARSE_DEPTH_M = 20.0


@pytest.fixture(scope="module")
def coarse_setup():
    """Construit un modèle, une acquisition et des données observées."""

    model = build_synthetic_velocity_model(
        dx_m=COARSE_DX_M, dz_m=COARSE_DZ_M
    )
    acquisition = build_acquisition(
        model.x_m,
        model.z_m,
        duration_s=COARSE_DURATION_S,
        dt_s=COARSE_DT_S,
        source_z_m=COARSE_DEPTH_M,
        receiver_z_m=COARSE_DEPTH_M,
    )

    true_squared = model.velocity_m_s**2
    observed = forward_data(true_squared, acquisition).receiver_signals

    start_velocity = model.velocity_m_s.copy()
    start_velocity[model.anomaly_mask] = 2000.0

    return model, acquisition, true_squared, observed, start_velocity**2


def test_propagator_matches_reference_solver(coarse_setup):
    """Le propagateur d'inversion reproduit le solveur de référence.

    Les deux implémentations doivent coïncider à la précision machine,
    sinon l'inversion n'inverse pas la même physique que le modèle direct
    déjà validé du projet.
    """

    model, _, true_squared, _, _ = coarse_setup

    reference = simulate_acoustic_wave(
        model,
        duration_s=COARSE_DURATION_S,
        dt_s=COARSE_DT_S,
        source_z_m=COARSE_DEPTH_M,
        receiver_z_m=COARSE_DEPTH_M,
        snapshot_stride=10_000,
    )

    acquisition = build_acquisition(
        model.x_m,
        model.z_m,
        duration_s=COARSE_DURATION_S,
        dt_s=COARSE_DT_S,
        source_z_m=COARSE_DEPTH_M,
        receiver_z_m=COARSE_DEPTH_M,
        receiver_x_m=reference.receiver_x_m,
    )

    produced = forward_data(true_squared, acquisition).receiver_signals

    amplitude = float(np.max(np.abs(reference.receiver_signals)))

    assert amplitude > 0.0

    relative_difference = (
        float(np.max(np.abs(produced - reference.receiver_signals)))
        / amplitude
    )

    assert relative_difference < 1.0e-12


def test_directional_derivative_matches_adjoint_gradient(coarse_setup):
    """Le gradient adjoint reproduit la dérivée directionnelle.

    La différence finie centrée est d'ordre deux : diviser epsilon par dix
    doit diviser l'erreur par cent, jusqu'au plancher d'arrondi. On teste
    la valeur et cette décroissance.
    """

    _, acquisition, _, observed, initial_squared = coarse_setup

    objective_value, gradient = objective_and_gradient(
        initial_squared, observed, acquisition
    )

    assert np.isfinite(objective_value)
    assert np.all(np.isfinite(gradient))

    generator = np.random.default_rng(20261030)
    direction = np.zeros_like(initial_squared)
    direction[1:-1, 1:-1] = generator.standard_normal(
        (initial_squared.shape[0] - 2, initial_squared.shape[1] - 2)
    )
    direction *= float(np.mean(initial_squared))

    analytic = float(np.sum(gradient * direction))

    assert abs(analytic) > 0.0

    errors = []

    for epsilon in (1.0e-5, 1.0e-6):
        plus = objective_only(
            initial_squared + epsilon * direction, observed, acquisition
        )
        minus = objective_only(
            initial_squared - epsilon * direction, observed, acquisition
        )
        numeric = (plus - minus) / (2.0 * epsilon)
        errors.append(abs(numeric - analytic) / abs(analytic))

    # Précision absolue au pas le plus fin.
    assert errors[-1] < 1.0e-6

    # Décroissance d'ordre deux : au moins un facteur dix sur un facteur
    # dix d'epsilon, ce qui exclut un gradient faux mais proche.
    assert errors[1] < errors[0] / 10.0


def test_gradient_is_zero_at_the_true_model(coarse_setup):
    """Au modèle exact, le résidu et donc le gradient s'annulent."""

    _, acquisition, true_squared, observed, _ = coarse_setup

    objective_value, gradient = objective_and_gradient(
        true_squared, observed, acquisition
    )

    assert objective_value == pytest.approx(0.0, abs=1.0e-20)
    assert float(np.max(np.abs(gradient))) == pytest.approx(
        0.0, abs=1.0e-18
    )


def test_pseudo_hessian_is_non_negative(coarse_setup):
    """Le pseudo-hessien est une somme de carrés, donc positif."""

    _, acquisition, _, observed, initial_squared = coarse_setup

    _, _, pseudo_hessian = objective_and_gradient(
        initial_squared,
        observed,
        acquisition,
        return_pseudo_hessian=True,
    )

    assert pseudo_hessian is not None
    assert np.all(pseudo_hessian >= 0.0)
    assert float(np.max(pseudo_hessian)) > 0.0


def test_multishot_sums_single_shot_contributions(coarse_setup):
    """La version multi-tirs additionne bien les contributions."""

    model, acquisition, _, observed, initial_squared = coarse_setup

    second_acquisition = build_acquisition(
        model.x_m,
        model.z_m,
        duration_s=COARSE_DURATION_S,
        dt_s=COARSE_DT_S,
        source_x_m=300.0,
        source_z_m=COARSE_DEPTH_M,
        receiver_z_m=COARSE_DEPTH_M,
    )
    second_observed = forward_data(
        model.velocity_m_s**2, second_acquisition
    ).receiver_signals

    first_value, first_gradient = objective_and_gradient(
        initial_squared, observed, acquisition
    )
    second_value, second_gradient = objective_and_gradient(
        initial_squared, second_observed, second_acquisition
    )

    total_value, total_gradient = objective_and_gradient_multishot(
        initial_squared,
        [observed, second_observed],
        [acquisition, second_acquisition],
    )

    assert total_value == pytest.approx(first_value + second_value)
    np.testing.assert_allclose(
        total_gradient, first_gradient + second_gradient, rtol=1.0e-12
    )


def test_velocity_gradient_applies_chain_rule():
    """La conversion du gradient en m vers un gradient en v est exacte."""

    velocity = np.array([[1000.0, 2000.0], [1500.0, 2500.0]])
    gradient_m = np.array([[1.0, -2.0], [0.5, 3.0]])

    produced = gradient_with_respect_to_velocity(velocity, gradient_m)

    np.testing.assert_allclose(produced, 2.0 * velocity * gradient_m)


def test_forward_rejects_unstable_time_step(coarse_setup):
    """Un pas temporel violant la condition CFL est refusé."""

    model, _, true_squared, _, _ = coarse_setup

    acquisition = build_acquisition(
        model.x_m,
        model.z_m,
        duration_s=0.05,
        dt_s=5.0e-2,
        source_z_m=COARSE_DEPTH_M,
        receiver_z_m=COARSE_DEPTH_M,
    )

    with pytest.raises(ValueError, match="CFL"):
        forward_data(true_squared, acquisition)


def test_objective_rejects_mismatched_observed_shape(coarse_setup):
    """Des données observées de mauvaise forme sont refusées."""

    _, acquisition, _, observed, initial_squared = coarse_setup

    with pytest.raises(ValueError):
        objective_and_gradient(
            initial_squared, observed[:-5, :], acquisition
        )

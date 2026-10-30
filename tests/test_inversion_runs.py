"""Tests de non-régression des deux formulations d'inversion.

Ces tests vérifient le comportement attendu d'une inversion : la fonction
objectif diminue, le modèle se rapproche de la référence, et les garde-fous
numériques fonctionnent. Ils tournent sur une grille très grossière pour
rester rapides.
"""

import numpy as np
import pytest

from seismic.model.velocity_model import build_synthetic_velocity_model
from seismic.inversion.fwi import build_acquisition, forward_data
from seismic.inversion.optimize import (
    build_preconditioner,
    lbfgs_inversion,
    smooth_field,
)
from seismic.inversion.parametric import (
    build_gaussian_basis,
    gauss_newton_inversion,
)

GRID_SPACING_M = 20.0
TIME_STEP_S = 4.0e-3
DURATION_S = 0.42
INSTRUMENT_DEPTH_M = 20.0


@pytest.fixture(scope="module")
def inversion_setup():
    """Modèle grossier, deux tirs et données observées synthétiques."""

    model = build_synthetic_velocity_model(
        dx_m=GRID_SPACING_M, dz_m=GRID_SPACING_M
    )

    acquisitions = [
        build_acquisition(
            model.x_m,
            model.z_m,
            duration_s=DURATION_S,
            dt_s=TIME_STEP_S,
            source_x_m=source_x_m,
            source_z_m=INSTRUMENT_DEPTH_M,
            receiver_z_m=INSTRUMENT_DEPTH_M,
        )
        for source_x_m in (180.0, 420.0)
    ]

    true_squared = model.velocity_m_s**2
    observed = [
        forward_data(true_squared, acquisition).receiver_signals
        for acquisition in acquisitions
    ]

    start_velocity = model.velocity_m_s.copy()
    start_velocity[model.anomaly_mask] = 2000.0

    return model, acquisitions, observed, true_squared, start_velocity**2


def test_lbfgs_reduces_objective_and_model_error(inversion_setup):
    """L'inversion FWI diminue Phi et rapproche le modèle de la référence."""

    model, acquisitions, observed, true_squared, initial = inversion_setup

    final, history = lbfgs_inversion(
        initial,
        observed,
        acquisitions,
        max_iterations=6,
        true_squared_velocity=true_squared,
        target_mask=model.anomaly_mask,
        gradient_smoothing_points=1.0,
        verbose=False,
    )

    assert len(history.objectives) >= 2

    # La fonction objectif doit strictement diminuer.
    assert history.objectives[-1] < history.objectives[0]

    # Et la decroissance doit etre monotone sur les pas acceptes.
    differences = np.diff(history.objectives)
    assert np.all(differences <= 1.0e-18)

    # L'erreur sur la zone cible doit s'ameliorer.
    assert history.target_errors[-1] < history.target_errors[0]

    # Le modele reste dans des bornes physiques.
    velocity = np.sqrt(final)
    assert float(np.min(velocity)) >= 800.0
    assert float(np.max(velocity)) <= 3500.0
    assert np.all(np.isfinite(velocity))


def test_lbfgs_stays_put_when_data_already_match(inversion_setup):
    """Partir du modèle exact ne dégrade pas la solution."""

    model, acquisitions, observed, true_squared, _ = inversion_setup

    final, history = lbfgs_inversion(
        true_squared,
        observed,
        acquisitions,
        max_iterations=2,
        true_squared_velocity=true_squared,
        target_mask=model.anomaly_mask,
        verbose=False,
    )

    assert history.objectives[0] == pytest.approx(0.0, abs=1.0e-18)
    np.testing.assert_allclose(final, true_squared, rtol=1.0e-10)


def test_lbfgs_rejects_inconsistent_inputs(inversion_setup):
    """Un nombre de jeux de données incohérent est refusé."""

    _, acquisitions, observed, _, initial = inversion_setup

    with pytest.raises(ValueError):
        lbfgs_inversion(
            initial, observed[:1], acquisitions, verbose=False
        )

    with pytest.raises(ValueError):
        lbfgs_inversion(
            initial,
            observed,
            acquisitions,
            velocity_bounds_m_s=(3000.0, 1000.0),
            verbose=False,
        )


def test_preconditioner_is_bounded_and_finite():
    """Le préconditionneur est fini, positif et normalisé."""

    pseudo_hessian = np.abs(
        np.random.default_rng(5).standard_normal((31, 41))
    )

    # Une zone non eclairee, pour verifier l'amortissement.
    pseudo_hessian[:5, :] = 0.0

    preconditioner = build_preconditioner(pseudo_hessian)

    assert np.all(np.isfinite(preconditioner))
    assert np.all(preconditioner > 0.0)
    assert float(np.max(preconditioner)) == pytest.approx(1.0)


def test_smoothing_preserves_a_constant_field():
    """Le lissage gaussien laisse un champ constant inchangé."""

    constant = np.full((25, 30), 3.5)

    np.testing.assert_allclose(
        smooth_field(constant, 2.0), constant, rtol=1.0e-12
    )

    # Un lissage desactive renvoie exactement l'entree.
    np.testing.assert_array_equal(smooth_field(constant, 0.0), constant)


def test_gaussian_basis_shape_and_positivity():
    """La base réduite a la forme attendue et reste positive."""

    model = build_synthetic_velocity_model(dx_m=20.0, dz_m=20.0)

    basis = build_gaussian_basis(
        model.x_m,
        model.z_m,
        x_range_m=(240.0, 480.0),
        z_range_m=(80.0, 180.0),
        number_x=4,
        number_z=3,
    )

    assert basis.shape == (12, model.z_m.size, model.x_m.size)
    assert np.all(basis >= 0.0)
    assert float(np.max(basis)) == pytest.approx(1.0, rel=1.0e-6)

    with pytest.raises(ValueError):
        build_gaussian_basis(
            model.x_m,
            model.z_m,
            x_range_m=(240.0, 480.0),
            z_range_m=(80.0, 180.0),
            number_x=0,
        )


def test_parametric_inversion_reduces_objective(inversion_setup):
    """L'inversion paramétrique réduite diminue Phi via LAPACK."""

    model, acquisitions, observed, _, initial = inversion_setup

    basis = build_gaussian_basis(
        model.x_m,
        model.z_m,
        x_range_m=(280.0, 440.0),
        z_range_m=(100.0, 160.0),
        number_x=3,
        number_z=2,
    )

    coefficients, final, history = gauss_newton_inversion(
        initial,
        basis,
        observed,
        acquisitions,
        max_iterations=2,
        verbose=False,
    )

    assert coefficients.size == basis.shape[0]
    assert history.objectives[-1] < history.objectives[0]
    assert np.all(final > 0.0)

    # La routine employee doit etre une routine LAPACK identifiee.
    assert set(history.solver_used) <= {"DPOSV", "DGELSD"}
    assert all(
        np.isfinite(value) for value in history.condition_numbers
    )

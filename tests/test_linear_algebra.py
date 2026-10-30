"""Tests des noyaux BLAS/LAPACK et du diagnostic de backend.

Chaque routine est comparée à une référence NumPy indépendante. C'est
l'exigence de validation croisée du projet : une implémentation optimisée
n'est utilisable que si elle produit le même résultat numérique que la
référence, dans une tolérance déclarée.
"""

import numpy as np
import pytest

from seismic.hpc.backend import diagnose_backend, format_report
from seismic.linear_algebra.kernels import (
    axpy,
    dot,
    gemm,
    gemv,
    nrm2,
    solve_least_squares,
    solve_symmetric_positive_definite,
)

TOLERANCE = 1.0e-12


@pytest.fixture(scope="module")
def sample_data():
    """Jeu de vecteurs et matrices reproductible."""

    generator = np.random.default_rng(4242)

    return {
        "x": generator.standard_normal(97),
        "y": generator.standard_normal(97),
        "matrix": generator.standard_normal((140, 11)),
        "vector": generator.standard_normal(140),
    }


def test_dot_matches_numpy(sample_data):
    """DDOT reproduit le produit scalaire de NumPy."""

    expected = float(sample_data["x"] @ sample_data["y"])

    assert dot(sample_data["x"], sample_data["y"]) == pytest.approx(
        expected, rel=TOLERANCE
    )


def test_axpy_matches_numpy(sample_data):
    """DAXPY reproduit ``alpha * x + y``."""

    expected = 2.5 * sample_data["x"] + sample_data["y"]
    produced = axpy(sample_data["x"], sample_data["y"].copy(), 2.5)

    np.testing.assert_allclose(produced, expected, rtol=TOLERANCE)


def test_nrm2_matches_numpy(sample_data):
    """DNRM2 reproduit la norme euclidienne."""

    expected = float(np.linalg.norm(sample_data["x"]))

    assert nrm2(sample_data["x"]) == pytest.approx(expected, rel=TOLERANCE)


def test_nrm2_avoids_overflow():
    """DNRM2 gère des amplitudes qui feraient déborder un calcul naïf."""

    large = np.full(4, 1.0e200, dtype=np.float64)

    # Le calcul naïf deborde, DNRM2 non. Le depassement est attendu ici.
    with np.errstate(over="ignore"):
        assert np.isinf(float(np.sum(large**2)))
    assert np.isfinite(nrm2(large))
    assert nrm2(large) == pytest.approx(2.0e200, rel=1.0e-12)


def test_gemv_matches_numpy(sample_data):
    """DGEMV reproduit le produit matrice-vecteur, transposé ou non."""

    matrix = sample_data["matrix"]
    generator = np.random.default_rng(7)
    small_vector = generator.standard_normal(matrix.shape[1])

    np.testing.assert_allclose(
        gemv(matrix, small_vector), matrix @ small_vector, rtol=TOLERANCE
    )
    np.testing.assert_allclose(
        gemv(matrix, sample_data["vector"], transpose=True),
        matrix.T @ sample_data["vector"],
        rtol=TOLERANCE,
    )


def test_gemv_rejects_wrong_dimension(sample_data):
    """Une taille de vecteur incompatible est refusée."""

    with pytest.raises(ValueError):
        gemv(sample_data["matrix"], np.ones(3))


def test_gemm_builds_normal_matrix(sample_data):
    """DGEMM reproduit la matrice normale ``J' J``."""

    matrix = sample_data["matrix"]

    np.testing.assert_allclose(
        gemm(matrix, matrix, transpose_left=True),
        matrix.T @ matrix,
        rtol=TOLERANCE,
    )


def test_posv_solves_symmetric_system(sample_data):
    """DPOSV résout le système normal régularisé."""

    matrix = sample_data["matrix"]
    normal = matrix.T @ matrix + 0.5 * np.eye(matrix.shape[1])
    right_hand_side = matrix.T @ sample_data["vector"]

    produced = solve_symmetric_positive_definite(normal, right_hand_side)
    expected = np.linalg.solve(normal, right_hand_side)

    np.testing.assert_allclose(produced, expected, rtol=1.0e-10)


def test_posv_rejects_non_positive_definite():
    """Une matrice non définie positive déclenche une erreur explicite."""

    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(np.linalg.LinAlgError):
        solve_symmetric_positive_definite(indefinite, np.ones(2))


def test_gelsd_matches_numpy_least_squares(sample_data):
    """DGELSD reproduit la solution des moindres carrés de NumPy."""

    matrix = sample_data["matrix"]
    solution, rank, singular_values = solve_least_squares(
        matrix, sample_data["vector"]
    )

    expected = np.linalg.lstsq(
        matrix, sample_data["vector"], rcond=None
    )[0]

    np.testing.assert_allclose(solution, expected, rtol=1.0e-10)
    assert rank == matrix.shape[1]
    assert singular_values.size == matrix.shape[1]


def test_gelsd_handles_rank_deficiency():
    """DGELSD traite une matrice de rang déficient sans échouer."""

    base = np.random.default_rng(11).standard_normal((40, 3))

    # La quatrieme colonne duplique la premiere : rang 3, pas 4.
    matrix = np.column_stack([base, base[:, 0]])
    target = np.random.default_rng(12).standard_normal(40)

    solution, rank, _ = solve_least_squares(matrix, target)

    assert rank == 3
    assert np.all(np.isfinite(solution))


def test_backend_report_is_consistent():
    """Le diagnostic de backend est cohérent et sérialisable."""

    report = diagnose_backend()

    assert isinstance(report.provider, str)
    assert report.provider

    # L'affirmation oneMKL doit decouler du fournisseur detecte.
    assert report.uses_onemkl == (report.provider == "intel_onemkl")

    payload = report.to_dict()

    assert payload["provider"] == report.provider
    assert "environment" in payload

    text = format_report(report)

    assert "BACKEND" in text
    assert report.provider in text

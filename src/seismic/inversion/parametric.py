"""Inversion paramétrique réduite par Gauss-Newton et LAPACK.

Ce module complète la FWI par état adjoint avec une seconde formulation,
qui répond à un besoin différent et fait apparaître LAPACK de façon
légitime.

**Pourquoi une seconde formulation ?** La FWI par état adjoint inverse
les 45 451 cellules de la grille. Former sa Jacobienne demanderait autant
de simulations directes, et la matrice occuperait environ dix gigaoctets :
c'est hors de portée, et c'est précisément pourquoi la méthode adjointe
existe. Mais cela signifie aussi qu'aucune matrice dense n'apparaît, donc
aucune factorisation.

En réduisant le modèle à une combinaison de quelques dizaines de fonctions
de base spatiales, la Jacobienne redevient constructible — quelques
dizaines de simulations — et de taille raisonnable. Les équations
normales régularisées

    (J' J + lambda I) dc = J' r

sont alors un vrai système symétrique défini positif, résolu par
factorisation de Cholesky (LAPACK DPOSV), avec repli sur une résolution
par SVD (LAPACK DGELSD) si le conditionnement l'exige. Les produits
``J' J`` et ``J' r`` sont des appels BLAS DGEMM et DGEMV sur des matrices
de plusieurs dizaines de milliers de lignes.

Les deux formulations sont complémentaires et doivent être nommées
distinctement : la FWI adjointe reconstruit un champ de vitesse, celle-ci
estime les coefficients d'une base réduite.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from seismic.linear_algebra.kernels import (
    gemm,
    gemv,
    nrm2,
    solve_least_squares,
    solve_symmetric_positive_definite,
)

from .fwi import Acquisition, forward_data

FloatArray = NDArray[np.float64]


@dataclass
class ParametricHistory:
    """Historique d'une inversion paramétrique réduite.

    Attributes:
        objectives: Valeur de Phi à chaque itération.
        step_lengths: Pas retenu par la recherche linéaire.
        condition_numbers: Conditionnement de la matrice normale.
        solver_used: Routine LAPACK employée à chaque itération.
        rank_estimates: Rang estimé lorsque DGELSD est utilisé.
    """

    objectives: list[float] = field(default_factory=list)
    step_lengths: list[float] = field(default_factory=list)
    condition_numbers: list[float] = field(default_factory=list)
    solver_used: list[str] = field(default_factory=list)
    rank_estimates: list[int] = field(default_factory=list)


def build_gaussian_basis(
    x_m: FloatArray,
    z_m: FloatArray,
    *,
    x_range_m: tuple[float, float],
    z_range_m: tuple[float, float],
    number_x: int = 8,
    number_z: int = 5,
    overlap: float = 1.0,
) -> FloatArray:
    """Construit une base de fonctions gaussiennes sur une zone d'intérêt.

    Chaque fonction est une gaussienne centrée sur un nœud d'une grille
    régulière couvrant la zone. Les perturbations du modèle sont
    recherchées dans l'espace engendré par ces fonctions, ce qui réduit
    le nombre d'inconnues de plusieurs dizaines de milliers à quelques
    dizaines.

    Args:
        x_m: Axe horizontal de la grille.
        z_m: Axe vertical de la grille.
        x_range_m: Bornes horizontales de la zone d'intérêt.
        z_range_m: Bornes verticales de la zone d'intérêt.
        number_x: Nombre de fonctions selon x.
        number_z: Nombre de fonctions selon z.
        overlap: Recouvrement des gaussiennes, en multiples de
            l'espacement entre centres.

    Returns:
        La base, de forme (number_z * number_x, nz, nx).

    Raises:
        ValueError: Si la configuration de la base est invalide.
    """

    if number_x < 1 or number_z < 1:
        raise ValueError("La base doit contenir au moins une fonction.")

    if overlap <= 0.0:
        raise ValueError("Le recouvrement doit être strictement positif.")

    centers_x = np.linspace(x_range_m[0], x_range_m[1], number_x)
    centers_z = np.linspace(z_range_m[0], z_range_m[1], number_z)

    spacing_x = (
        float(centers_x[1] - centers_x[0])
        if number_x > 1
        else float(x_range_m[1] - x_range_m[0]) or 1.0
    )
    spacing_z = (
        float(centers_z[1] - centers_z[0])
        if number_z > 1
        else float(z_range_m[1] - z_range_m[0]) or 1.0
    )

    sigma_x = overlap * abs(spacing_x)
    sigma_z = overlap * abs(spacing_z)

    x_grid, z_grid = np.meshgrid(x_m, z_m)

    functions = []

    for center_z in centers_z:
        for center_x in centers_x:
            functions.append(
                np.exp(
                    -0.5
                    * (
                        ((x_grid - center_x) / sigma_x) ** 2
                        + ((z_grid - center_z) / sigma_z) ** 2
                    )
                )
            )

    return np.stack(functions, axis=0)


def _model_from_coefficients(
    reference_squared_velocity: FloatArray,
    basis: FloatArray,
    coefficients: FloatArray,
) -> FloatArray:
    """Reconstruit le modèle complet depuis les coefficients réduits."""

    perturbation = np.tensordot(coefficients, basis, axes=(0, 0))

    return reference_squared_velocity + perturbation


def _stack_residual(
    predicted: list[FloatArray],
    observed: list[FloatArray],
) -> FloatArray:
    """Empile les résidus de tous les tirs en un seul vecteur."""

    return np.concatenate(
        [
            (p - o).ravel()
            for p, o in zip(predicted, observed)
        ]
    )


def build_reduced_jacobian(
    coefficients: FloatArray,
    reference_squared_velocity: FloatArray,
    basis: FloatArray,
    observed_datasets: list[FloatArray],
    acquisitions: list[Acquisition],
    *,
    perturbation: float = 1.0e3,
) -> tuple[FloatArray, FloatArray]:
    """Construit la Jacobienne réduite par différences finies.

    Le coût est d'une simulation directe par coefficient et par tir. Cela
    reste praticable parce que la base compte quelques dizaines de
    fonctions, et non les 45 451 cellules de la grille.

    Args:
        coefficients: Coefficients courants.
        reference_squared_velocity: Modèle de référence de la base.
        basis: Base de fonctions, forme (nk, nz, nx).
        observed_datasets: Données observées, une matrice par tir.
        acquisitions: Acquisitions, une par tir.
        perturbation: Pas de différence finie sur les coefficients, en
            unités de ``m = v**2``.

    Returns:
        Le couple ``(jacobienne, residu)``. La Jacobienne a la forme
        (nd, nk) et le résidu la taille nd.
    """

    model = _model_from_coefficients(
        reference_squared_velocity, basis, coefficients
    )

    predicted = [
        forward_data(model, acquisition).receiver_signals
        for acquisition in acquisitions
    ]
    residual = _stack_residual(predicted, observed_datasets)

    number_of_coefficients = coefficients.size

    jacobian = np.zeros(
        (residual.size, number_of_coefficients),
        dtype=np.float64,
    )

    for index in range(number_of_coefficients):
        perturbed_coefficients = coefficients.copy()
        perturbed_coefficients[index] += perturbation

        perturbed_model = _model_from_coefficients(
            reference_squared_velocity, basis, perturbed_coefficients
        )

        perturbed_predicted = [
            forward_data(perturbed_model, acquisition).receiver_signals
            for acquisition in acquisitions
        ]

        perturbed_residual = _stack_residual(
            perturbed_predicted, observed_datasets
        )

        jacobian[:, index] = (
            perturbed_residual - residual
        ) / perturbation

    return jacobian, residual


def gauss_newton_inversion(
    reference_squared_velocity: FloatArray,
    basis: FloatArray,
    observed_datasets: list[FloatArray],
    acquisitions: list[Acquisition],
    *,
    max_iterations: int = 4,
    regularization: float = 1.0e-2,
    perturbation: float = 1.0e3,
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
    condition_threshold: float = 1.0e10,
    verbose: bool = True,
) -> tuple[FloatArray, FloatArray, ParametricHistory]:
    """Inverse les coefficients d'une base réduite par Gauss-Newton.

    À chaque itération, la Jacobienne réduite est construite, la matrice
    normale régularisée est formée par DGEMM, le second membre par DGEMV,
    et le système est résolu par DPOSV ou, si le conditionnement est
    mauvais, par DGELSD.

    Args:
        reference_squared_velocity: Modèle de référence ``m_ref``.
        basis: Base de fonctions, forme (nk, nz, nx).
        observed_datasets: Données observées, une matrice par tir.
        acquisitions: Acquisitions, une par tir.
        max_iterations: Nombre d'itérations de Gauss-Newton.
        regularization: Coefficient ``lambda`` relatif à la trace de la
            matrice normale.
        perturbation: Pas de différence finie sur les coefficients.
        line_search_alphas: Pas essayés par la recherche linéaire.
        condition_threshold: Au-delà, DGELSD remplace DPOSV.
        verbose: Affiche la progression.

    Returns:
        Le triplet ``(coefficients, modèle_final, historique)``.
    """

    number_of_coefficients = basis.shape[0]
    coefficients = np.zeros(number_of_coefficients, dtype=np.float64)

    history = ParametricHistory()

    if verbose:
        print(
            f"{'iter':>4} | {'Phi':>14} | {'cond(N)':>11} | "
            f"{'routine':>8} | {'pas':>6}"
        )
        print("-" * 56)

    for iteration in range(max_iterations):
        jacobian, residual = build_reduced_jacobian(
            coefficients,
            reference_squared_velocity,
            basis,
            observed_datasets,
            acquisitions,
            perturbation=perturbation,
        )

        objective_value = 0.5 * float(nrm2(residual) ** 2)

        # Equations normales : N = J'J (DGEMM), b = -J'r (DGEMV).
        normal_matrix = gemm(jacobian, jacobian, transpose_left=True)
        right_hand_side = gemv(jacobian, residual, transpose=True, alpha=-1.0)

        trace = float(np.trace(normal_matrix))
        damping = regularization * trace / max(number_of_coefficients, 1)

        regularised = normal_matrix + damping * np.eye(
            number_of_coefficients
        )

        eigenvalues = np.linalg.eigvalsh(regularised)
        condition_number = float(
            abs(eigenvalues[-1]) / max(abs(eigenvalues[0]), 1.0e-300)
        )

        if condition_number < condition_threshold:
            try:
                update = solve_symmetric_positive_definite(
                    regularised, right_hand_side
                )
                routine = "DPOSV"
                rank = number_of_coefficients
            except np.linalg.LinAlgError:
                update, rank, _ = solve_least_squares(
                    jacobian, -residual
                )
                routine = "DGELSD"
        else:
            update, rank, _ = solve_least_squares(jacobian, -residual)
            routine = "DGELSD"

        accepted_alpha = 0.0
        accepted_objective = objective_value
        accepted_coefficients = coefficients.copy()

        for alpha in line_search_alphas:
            candidate = coefficients + alpha * update
            candidate_model = _model_from_coefficients(
                reference_squared_velocity, basis, candidate
            )

            if np.any(candidate_model <= 0.0):
                continue

            candidate_residual = _stack_residual(
                [
                    forward_data(
                        candidate_model, acquisition
                    ).receiver_signals
                    for acquisition in acquisitions
                ],
                observed_datasets,
            )

            candidate_objective = 0.5 * float(
                nrm2(candidate_residual) ** 2
            )

            if candidate_objective < accepted_objective:
                accepted_alpha = alpha
                accepted_objective = candidate_objective
                accepted_coefficients = candidate
                break

        history.objectives.append(objective_value)
        history.condition_numbers.append(condition_number)
        history.solver_used.append(routine)
        history.rank_estimates.append(int(rank))
        history.step_lengths.append(accepted_alpha)

        if verbose:
            print(
                f"{iteration:4d} | {objective_value:14.6e} | "
                f"{condition_number:11.3e} | {routine:>8} | "
                f"{accepted_alpha:6.2f}"
            )

        if accepted_alpha == 0.0:
            if verbose:
                print("Arret : aucun pas n'ameliore la fonction objectif.")
            break

        coefficients = accepted_coefficients

    final_model = _model_from_coefficients(
        reference_squared_velocity, basis, coefficients
    )

    final_residual = _stack_residual(
        [
            forward_data(final_model, acquisition).receiver_signals
            for acquisition in acquisitions
        ],
        observed_datasets,
    )
    history.objectives.append(0.5 * float(nrm2(final_residual) ** 2))

    return coefficients, final_model, history

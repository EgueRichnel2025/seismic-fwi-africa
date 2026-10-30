"""Optimisation L-BFGS préconditionnée de la fonction objectif FWI.

La récursion à deux boucles de L-BFGS est implémentée explicitement à
partir des noyaux BLAS du projet (DDOT, DAXPY, DNRM2). Ce choix n'est pas
cosmétique : la récursion est entièrement composée de produits scalaires
et de mises à jour vectorielles sur des vecteurs de taille ``Nz * Nx``,
soit 45 451 composantes pour la grille de référence. Ce sont les seules
opérations d'algèbre linéaire denses que contient une FWI par état
adjoint, puisque la Jacobienne n'est jamais formée.

Deux points méritent d'être explicités.

**Préconditionnement.** Le gradient brut d'une FWI est dominé par les
zones de forte amplitude du champ, c'est-à-dire le voisinage immédiat de
la source. Sans correction, l'optimiseur dépense ses corrections en
surface au lieu d'atteindre les cibles profondes. La diagonale du
pseudo-hessien, ``somme_n (L u^n)^2``, mesure précisément l'illumination
de chaque point du modèle ; son inverse régularisé sert de matrice
initiale ``H0`` dans la récursion. Le préconditionneur entre donc par
``H0`` et non par une modification du gradient : les paires de courbure
restent cohérentes et la dérivée directionnelle utilisée par la condition
d'Armijo reste celle du vrai gradient.

**Multi-tirs.** Un tir unique n'éclaire qu'une ouverture angulaire
limitée. Les contributions de plusieurs tirs s'additionnent dans la
fonction objectif comme dans le gradient.

Référence de l'algorithme : Nocedal et Wright, *Numerical Optimization*,
chapitre 7. Référence du pseudo-hessien : Shin, Jang et Min, *Improved
amplitude preservation for prestack depth migration by inverse scattering
theory*, Geophysical Prospecting, 2001.
"""

from dataclasses import dataclass, field
from collections import deque
import math

import numpy as np
from numpy.typing import NDArray

from seismic.linear_algebra.kernels import axpy, dot, nrm2

from .fwi import (
    Acquisition,
    objective_and_gradient_multishot,
    objective_only_multishot,
)

FloatArray = NDArray[np.float64]


@dataclass
class InversionHistory:
    """Historique quantitatif d'une inversion.

    Attributes:
        objectives: Valeur de Phi à chaque itération acceptée.
        gradient_norms: Norme du gradient à chaque itération.
        step_lengths: Pas retenu par la recherche linéaire.
        model_errors: Erreur relative de modèle sur toute la grille.
        target_errors: Erreur relative restreinte à la zone cible.
        data_errors: Erreur relative sur les données.
        rejected_steps: Itérations sans pas admissible.
        simulation_count: Nombre total de simulations directes effectuées.
    """

    objectives: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    step_lengths: list[float] = field(default_factory=list)
    model_errors: list[float] = field(default_factory=list)
    target_errors: list[float] = field(default_factory=list)
    data_errors: list[float] = field(default_factory=list)
    rejected_steps: list[int] = field(default_factory=list)
    simulation_count: int = 0


def smooth_field(field_array: FloatArray, sigma_points: float) -> FloatArray:
    """Lisse un champ par convolution gaussienne séparable.

    Args:
        field_array: Champ à lisser, forme (nz, nx).
        sigma_points: Écart-type du noyau en points de grille. Une valeur
            nulle ou négative désactive le lissage.

    Returns:
        Le champ lissé.
    """

    if sigma_points <= 0.0:
        return field_array

    radius = max(1, int(np.ceil(3.0 * sigma_points)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
    kernel /= float(np.sum(kernel))

    def convolve_with_edge_padding(line: FloatArray) -> FloatArray:
        """Convolue en répliquant les bords.

        Sans ce rembourrage, ``mode="same"`` suppose un champ nul hors des
        limites et atténue artificiellement le gradient sur les bords du
        modèle, ce qui biaiserait l'inversion près de la surface.
        """

        padded = np.pad(line, radius, mode="edge")

        return np.convolve(padded, kernel, mode="valid")

    smoothed = np.apply_along_axis(
        convolve_with_edge_padding, axis=1, arr=field_array
    )

    return np.apply_along_axis(
        convolve_with_edge_padding, axis=0, arr=smoothed
    )


def build_preconditioner(
    pseudo_hessian: FloatArray,
    *,
    damping: float = 1.0e-3,
    smoothing_points: float = 2.0,
) -> FloatArray:
    """Construit le préconditionneur diagonal d'illumination.

    Args:
        pseudo_hessian: Diagonale du pseudo-hessien, forme (nz, nx).
        damping: Amortissement relatif au maximum, qui évite d'amplifier
            démesurément les zones non éclairées.
        smoothing_points: Lissage appliqué au pseudo-hessien.

    Returns:
        Le préconditionneur, normalisé à un maximum unitaire.
    """

    illumination = smooth_field(
        np.abs(pseudo_hessian), smoothing_points
    )

    maximum = float(np.max(illumination))

    if not math.isfinite(maximum) or maximum <= 0.0:
        return np.ones_like(pseudo_hessian)

    preconditioner = 1.0 / (illumination + damping * maximum)

    return preconditioner / float(np.max(preconditioner))


def _two_loop_recursion(
    gradient: FloatArray,
    memory_s: deque,
    memory_y: deque,
    memory_rho: deque,
    preconditioner: FloatArray,
) -> FloatArray:
    """Calcule la direction L-BFGS ``-H.g`` par récursion à deux boucles.

    L'intégralité de la récursion repose sur DDOT et DAXPY. Le
    préconditionneur intervient comme matrice initiale ``H0``, mise à
    l'échelle par le facteur de Nocedal généralisé ``(s'y) / (y' P y)``.

    Args:
        gradient: Gradient courant aplati.
        memory_s: Différences de modèle successives.
        memory_y: Différences de gradient successives.
        memory_rho: Coefficients ``1 / (y'.s)``.
        preconditioner: Diagonale de ``H0``, aplatie.

    Returns:
        La direction de descente, aplatie.
    """

    q = gradient.copy()
    alphas: list[float] = []

    for s_vector, y_vector, rho in zip(
        reversed(memory_s), reversed(memory_y), reversed(memory_rho)
    ):
        alpha = rho * dot(s_vector, q)
        alphas.append(alpha)
        q = axpy(y_vector, q, -alpha)

    q = q * preconditioner

    if memory_s:
        s_last = memory_s[-1]
        y_last = memory_y[-1]
        denominator = dot(y_last, y_last * preconditioner)

        if denominator > 0.0:
            q = q * (dot(s_last, y_last) / denominator)

    for s_vector, y_vector, rho, alpha in zip(
        memory_s, memory_y, memory_rho, reversed(alphas)
    ):
        beta = rho * dot(y_vector, q)
        q = axpy(s_vector, q, alpha - beta)

    return -q


def lbfgs_inversion(
    initial_squared_velocity: FloatArray,
    observed_datasets: list[FloatArray],
    acquisitions: list[Acquisition],
    *,
    max_iterations: int = 20,
    memory_size: int = 8,
    velocity_bounds_m_s: tuple[float, float] = (800.0, 3500.0),
    gradient_smoothing_points: float = 2.0,
    use_preconditioner: bool = True,
    preconditioner_damping: float = 1.0e-3,
    armijo_coefficient: float = 1.0e-4,
    max_line_search_trials: int = 12,
    gradient_tolerance: float = 1.0e-10,
    true_squared_velocity: FloatArray | None = None,
    target_mask: NDArray[np.bool_] | None = None,
    verbose: bool = True,
) -> tuple[FloatArray, InversionHistory]:
    """Minimise Phi(m) par L-BFGS préconditionné avec gradient adjoint.

    Args:
        initial_squared_velocity: Modèle initial ``m0 = v0**2``.
        observed_datasets: Données observées, une matrice par tir.
        acquisitions: Acquisitions, une par tir.
        max_iterations: Nombre maximal d'itérations.
        memory_size: Nombre de paires (s, y) conservées.
        velocity_bounds_m_s: Bornes physiques imposées à la vitesse.
        gradient_smoothing_points: Lissage appliqué au gradient.
        use_preconditioner: Active le préconditionnement d'illumination.
        preconditioner_damping: Amortissement du préconditionneur.
        armijo_coefficient: Coefficient de décroissance suffisante.
        max_line_search_trials: Nombre maximal de rebroussements.
        gradient_tolerance: Seuil d'arrêt relatif sur la norme du gradient.
        true_squared_velocity: Modèle de référence, utilisé uniquement
            pour mesurer l'erreur. Il n'intervient jamais dans
            l'optimisation.
        target_mask: Zone d'intérêt pour l'erreur localisée.
        verbose: Affiche la progression.

    Returns:
        Le couple ``(modèle_final, historique)``.

    Raises:
        ValueError: Si les bornes ou les entrées sont incohérentes.
    """

    lower_velocity, upper_velocity = velocity_bounds_m_s

    if not 0.0 < lower_velocity < upper_velocity:
        raise ValueError("Les bornes de vitesse sont incohérentes.")

    if len(observed_datasets) != len(acquisitions):
        raise ValueError(
            "Il faut exactement un jeu de données observées par tir."
        )

    lower_bound = lower_velocity**2
    upper_bound = upper_velocity**2

    shape = initial_squared_velocity.shape
    model = np.clip(
        np.asarray(initial_squared_velocity, dtype=np.float64).copy(),
        lower_bound,
        upper_bound,
    )

    observed_norm = math.sqrt(
        float(sum(float(np.sum(d**2)) for d in observed_datasets))
    )

    true_velocity_field = None
    true_velocity_norm = 0.0

    if true_squared_velocity is not None:
        true_velocity_field = np.sqrt(true_squared_velocity)
        true_velocity_norm = float(np.linalg.norm(true_velocity_field))

    history = InversionHistory()

    memory_s: deque = deque(maxlen=memory_size)
    memory_y: deque = deque(maxlen=memory_size)
    memory_rho: deque = deque(maxlen=memory_size)

    objective_value, raw_gradient, pseudo_hessian = (
        objective_and_gradient_multishot(
            model,
            observed_datasets,
            acquisitions,
            return_pseudo_hessian=True,
        )
    )
    history.simulation_count += len(acquisitions)

    gradient = smooth_field(raw_gradient, gradient_smoothing_points)

    if use_preconditioner:
        preconditioner = build_preconditioner(
            pseudo_hessian,
            damping=preconditioner_damping,
            smoothing_points=gradient_smoothing_points,
        ).ravel()
    else:
        preconditioner = np.ones(model.size, dtype=np.float64)

    initial_gradient_norm = nrm2(gradient.ravel())

    if verbose:
        print(
            f"{'iter':>4} | {'Phi':>14} | {'||g||':>12} | "
            f"{'pas':>10} | {'E_d':>8} | {'E_m':>8} | {'E_cible':>8}"
        )
        print("-" * 80)

    for iteration in range(max_iterations + 1):
        gradient_flat = gradient.ravel()
        gradient_norm = nrm2(gradient_flat)

        # Phi = 0.5 * ||r||^2, donc ||r|| = sqrt(2 * Phi). Inutile de
        # relancer une simulation pour mesurer l'erreur sur les données.
        data_error = float("nan")

        if observed_norm > 0.0:
            data_error = (
                math.sqrt(max(2.0 * objective_value, 0.0)) / observed_norm
            )

        model_error = float("nan")
        target_error = float("nan")

        if true_velocity_field is not None and true_velocity_norm > 0.0:
            velocity_field = np.sqrt(model)
            difference = velocity_field - true_velocity_field

            model_error = (
                float(np.linalg.norm(difference)) / true_velocity_norm
            )

            if target_mask is not None and np.any(target_mask):
                target_error = float(
                    np.linalg.norm(difference[target_mask])
                ) / float(np.linalg.norm(true_velocity_field[target_mask]))

        history.objectives.append(objective_value)
        history.gradient_norms.append(gradient_norm)
        history.data_errors.append(data_error)
        history.model_errors.append(model_error)
        history.target_errors.append(target_error)

        if verbose:
            step_text = (
                f"{history.step_lengths[-1]:10.3e}"
                if history.step_lengths
                else f"{'-':>10}"
            )
            print(
                f"{iteration:4d} | {objective_value:14.6e} | "
                f"{gradient_norm:12.4e} | {step_text} | "
                f"{data_error:8.4f} | {model_error:8.4f} | "
                f"{target_error:8.4f}"
            )

        if iteration == max_iterations:
            break

        if gradient_norm <= gradient_tolerance * initial_gradient_norm:
            if verbose:
                print("Arret : le gradient est devenu negligeable.")
            break

        direction_flat = _two_loop_recursion(
            gradient_flat,
            memory_s,
            memory_y,
            memory_rho,
            preconditioner,
        )

        directional_derivative = dot(gradient_flat, direction_flat)

        if directional_derivative >= 0.0:
            direction_flat = -gradient_flat * preconditioner
            directional_derivative = dot(gradient_flat, direction_flat)
            memory_s.clear()
            memory_y.clear()
            memory_rho.clear()

        if memory_s:
            step_length = 1.0
        else:
            direction_norm = nrm2(direction_flat)
            step_length = (
                0.05 * float(np.mean(model)) / direction_norm
                if direction_norm > 0.0
                else 1.0
            )

        accepted = False

        for _ in range(max_line_search_trials):
            candidate_flat = axpy(
                direction_flat, model.ravel().copy(), step_length
            )
            candidate = np.clip(
                candidate_flat.reshape(shape), lower_bound, upper_bound
            )

            candidate_objective = objective_only_multishot(
                candidate, observed_datasets, acquisitions
            )
            history.simulation_count += len(acquisitions)

            sufficient_decrease = (
                objective_value
                + armijo_coefficient
                * step_length
                * directional_derivative
            )

            if candidate_objective <= sufficient_decrease:
                accepted = True
                break

            step_length *= 0.5

        if not accepted:
            history.rejected_steps.append(iteration)

            if verbose:
                print(
                    f"Iteration {iteration} : aucun pas admissible apres "
                    f"{max_line_search_trials} rebroussements. Arret."
                )
            break

        new_objective, new_raw_gradient = (
            objective_and_gradient_multishot(
                candidate, observed_datasets, acquisitions
            )
        )
        history.simulation_count += len(acquisitions)

        new_gradient = smooth_field(
            new_raw_gradient, gradient_smoothing_points
        )

        s_vector = (candidate - model).ravel()
        y_vector = (new_gradient - gradient).ravel()

        curvature = dot(y_vector, s_vector)

        # Condition de courbure : garantit que l'approximation de
        # l'inverse de la hessienne reste définie positive.
        if curvature > 1.0e-18 * nrm2(s_vector) * nrm2(y_vector):
            memory_s.append(s_vector)
            memory_y.append(y_vector)
            memory_rho.append(1.0 / curvature)

        history.step_lengths.append(step_length)

        model = candidate
        objective_value = new_objective
        gradient = new_gradient

    return model, history

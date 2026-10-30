"""Problème inverse : fonction objectif et gradient par état adjoint.

Formulation retenue :

    Phi(m) = 0.5 * || F(m) - d_obs ||_2^2

où ``m = v**2`` est le paramètre inversé, ``F`` l'opérateur de modélisation
directe et ``d_obs`` les données synthétiques observées.

Le paramétrage par ``m = v**2`` est choisi parce que c'est la quantité qui
intervient linéairement dans l'équation d'onde discrétisée, ce qui rend le
gradient adjoint exact. Le gradient par rapport à la vitesse s'en déduit
par la règle de dérivation composée :

    dPhi/dv = 2 * v * dPhi/dm

Les données sont synthétiques. Aucun résultat obtenu ici ne constitue
l'observation d'une structure géologique réelle.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seismic.wave.ricker import ricker_wavelet

from .propagator import (
    accumulate_adjoint_gradient,
    build_sponge_profile,
    build_source_kernel,
    propagate_forward,
)

FloatArray = NDArray[np.float64]

# Valeurs reprises du solveur de référence pour rester cohérent avec le
# jeu de données synthétiques déjà produit par le projet.
DEFAULT_SOURCE_X_M = 60.0
DEFAULT_SOURCE_Z_M = 6.0
DEFAULT_SOURCE_FREQUENCY_HZ = 25.0
DEFAULT_SOURCE_STRENGTH_NUMERIC = 1.0e6
DEFAULT_SOURCE_SIGMA_M = 4.0
DEFAULT_RECEIVER_Z_M = 4.0
DEFAULT_DURATION_S = 0.42
DEFAULT_DT_S = 4.0e-4


@dataclass(frozen=True)
class Acquisition:
    """Géométrie et paramètres numériques d'une acquisition virtuelle.

    Attributes:
        x_m: Axe horizontal de la grille.
        z_m: Axe vertical de la grille.
        dt_s: Pas temporel, en secondes.
        number_of_steps: Nombre de pas temporels.
        source_kernel: Distribution spatiale normalisée de la source.
        source_time_series: Amplitude de la source à chaque pas.
        receiver_iz: Indice de profondeur des récepteurs.
        receiver_ix: Indices horizontaux des récepteurs.
        sigma: Profil absorbant.
        source_x_m: Position horizontale effective de la source.
        source_z_m: Profondeur effective de la source.
    """

    x_m: FloatArray
    z_m: FloatArray
    dt_s: float
    number_of_steps: int
    source_kernel: FloatArray
    source_time_series: FloatArray
    receiver_iz: int
    receiver_ix: NDArray[np.int_]
    sigma: FloatArray
    source_x_m: float
    source_z_m: float

    @property
    def dx_m(self) -> float:
        """Pas horizontal de la grille, en mètres."""

        return float(self.x_m[1] - self.x_m[0])

    @property
    def dz_m(self) -> float:
        """Pas vertical de la grille, en mètres."""

        return float(self.z_m[1] - self.z_m[0])

    @property
    def receiver_x_m(self) -> FloatArray:
        """Positions horizontales effectives des récepteurs."""

        return self.x_m[self.receiver_ix].copy()

    @property
    def sample_times_s(self) -> FloatArray:
        """Instants d'échantillonnage des récepteurs."""

        return (
            np.arange(self.number_of_steps + 1, dtype=np.float64)
            * self.dt_s
        )


def build_acquisition(
    x_m: FloatArray,
    z_m: FloatArray,
    *,
    duration_s: float = DEFAULT_DURATION_S,
    dt_s: float = DEFAULT_DT_S,
    source_x_m: float = DEFAULT_SOURCE_X_M,
    source_z_m: float = DEFAULT_SOURCE_Z_M,
    source_frequency_hz: float = DEFAULT_SOURCE_FREQUENCY_HZ,
    source_strength_numeric: float = DEFAULT_SOURCE_STRENGTH_NUMERIC,
    source_sigma_m: float = DEFAULT_SOURCE_SIGMA_M,
    receiver_x_m: FloatArray | None = None,
    receiver_z_m: float = DEFAULT_RECEIVER_Z_M,
    sponge_width_m: float = 40.0,
    sponge_strength_s_inv: float = 80.0,
) -> Acquisition:
    """Construit une acquisition virtuelle sur une grille donnée.

    Par défaut, la géométrie reproduit celle du jeu de données synthétiques
    du projet : une source proche de la surface et une ligne de récepteurs
    de 80 m à 580 m espacés de 20 m.

    Args:
        x_m: Axe horizontal de la grille.
        z_m: Axe vertical de la grille.
        duration_s: Durée de l'enregistrement, en secondes.
        dt_s: Pas temporel, en secondes.
        source_x_m: Position horizontale de la source.
        source_z_m: Profondeur de la source.
        source_frequency_hz: Fréquence centrale de la source.
        source_strength_numeric: Coefficient numérique non calibré.
        source_sigma_m: Écart-type spatial de la source.
        receiver_x_m: Positions horizontales des récepteurs.
        receiver_z_m: Profondeur des récepteurs.
        sponge_width_m: Largeur de la zone absorbante.
        sponge_strength_s_inv: Intensité de l'atténuation.

    Returns:
        L'acquisition prête à être utilisée par le modèle direct.

    Raises:
        ValueError: Si une position est hors du domaine.
    """

    x_m = np.asarray(x_m, dtype=np.float64)
    z_m = np.asarray(z_m, dtype=np.float64)

    nx = x_m.size
    nz = z_m.size

    dx_m = float(x_m[1] - x_m[0])
    dz_m = float(z_m[1] - z_m[0])

    number_of_steps = int(np.ceil(duration_s / dt_s))

    if receiver_x_m is None:
        receiver_stop = min(580.0, float(x_m[-1] - dx_m)) + 0.5 * dx_m
        receiver_positions = np.arange(
            80.0, receiver_stop, 20.0, dtype=np.float64
        )
    else:
        receiver_positions = np.asarray(receiver_x_m, dtype=np.float64)

    if receiver_positions.ndim != 1 or receiver_positions.size == 0:
        raise ValueError("Il faut au moins une position de récepteur.")

    receiver_ix = np.array(
        [int(np.argmin(np.abs(x_m - position))) for position in receiver_positions],
        dtype=int,
    )
    receiver_iz = int(np.argmin(np.abs(z_m - receiver_z_m)))

    if np.any(receiver_ix < 1) or np.any(receiver_ix >= nx - 1):
        raise ValueError(
            "Les récepteurs doivent rester à l'intérieur du domaine."
        )

    if not (1 <= receiver_iz < nz - 1):
        raise ValueError(
            "Les récepteurs doivent être à l'intérieur du domaine."
        )

    source_ix = int(np.argmin(np.abs(x_m - source_x_m)))
    source_iz = int(np.argmin(np.abs(z_m - source_z_m)))

    if not (1 <= source_ix < nx - 1) or not (1 <= source_iz < nz - 1):
        raise ValueError("La source doit être à l'intérieur du domaine.")

    source_kernel = build_source_kernel(
        x_m,
        z_m,
        source_x_m,
        source_z_m,
        source_sigma_m=source_sigma_m,
    )

    times_s = np.arange(number_of_steps + 1, dtype=np.float64) * dt_s

    source_time_series = source_strength_numeric * ricker_wavelet(
        times_s,
        source_frequency_hz,
        1.5 / source_frequency_hz,
    )

    sigma = build_sponge_profile(
        nz,
        nx,
        dx_m,
        dz_m,
        sponge_width_m=sponge_width_m,
        sponge_strength_s_inv=sponge_strength_s_inv,
    )

    return Acquisition(
        x_m=x_m,
        z_m=z_m,
        dt_s=dt_s,
        number_of_steps=number_of_steps,
        source_kernel=source_kernel,
        source_time_series=source_time_series,
        receiver_iz=receiver_iz,
        receiver_ix=receiver_ix,
        sigma=sigma,
        source_x_m=float(x_m[source_ix]),
        source_z_m=float(z_m[source_iz]),
    )


def forward_data(
    squared_velocity: FloatArray,
    acquisition: Acquisition,
    *,
    store_wavefield: bool = False,
    wavefield_dtype: np.dtype = np.float32,
):
    """Calcule les données prédites ``F(m)`` pour un modèle donné.

    Args:
        squared_velocity: Carré de la vitesse, forme (nz, nx).
        acquisition: Géométrie et paramètres numériques.
        store_wavefield: Conserve le champ complet pour l'état adjoint.
        wavefield_dtype: Type de stockage du champ complet.

    Returns:
        Le résultat de la propagation directe.
    """

    return propagate_forward(
        squared_velocity,
        dt_s=acquisition.dt_s,
        number_of_steps=acquisition.number_of_steps,
        dx_m=acquisition.dx_m,
        dz_m=acquisition.dz_m,
        source_kernel=acquisition.source_kernel,
        source_time_series=acquisition.source_time_series,
        receiver_iz=acquisition.receiver_iz,
        receiver_ix=acquisition.receiver_ix,
        sigma=acquisition.sigma,
        store_wavefield=store_wavefield,
        wavefield_dtype=wavefield_dtype,
    )


def objective_only(
    squared_velocity: FloatArray,
    observed_data: FloatArray,
    acquisition: Acquisition,
) -> float:
    """Évalue la fonction objectif sans calculer le gradient.

    Args:
        squared_velocity: Carré de la vitesse, forme (nz, nx).
        observed_data: Données observées, forme (nt + 1, nr).
        acquisition: Géométrie et paramètres numériques.

    Returns:
        La valeur de ``Phi(m)``.
    """

    predicted = forward_data(squared_velocity, acquisition).receiver_signals
    residual = predicted - observed_data

    return 0.5 * float(np.sum(residual**2))


def objective_and_gradient(
    squared_velocity: FloatArray,
    observed_data: FloatArray,
    acquisition: Acquisition,
    *,
    wavefield_dtype: np.dtype = np.float64,
    return_pseudo_hessian: bool = False,
):
    """Évalue ``Phi(m)`` et son gradient exact par état adjoint.

    Le coût est d'une propagation directe et d'une rétro-propagation,
    indépendamment du nombre de paramètres.

    Args:
        squared_velocity: Carré de la vitesse, forme (nz, nx).
        observed_data: Données observées, forme (nt + 1, nr).
        acquisition: Géométrie et paramètres numériques.
        wavefield_dtype: Type de stockage du champ direct. ``float64``
            donne un gradient exact ; ``float32`` divise la mémoire par
            deux au prix d'une légère perte de précision.
        return_pseudo_hessian: Renvoie aussi la diagonale du pseudo-hessien,
            utilisée comme préconditionneur d'illumination.

    Returns:
        Le couple ``(Phi, dPhi/dm)``, ou le triplet
        ``(Phi, dPhi/dm, pseudo_hessien)`` si demandé.

    Raises:
        ValueError: Si les données observées n'ont pas la bonne forme.
    """

    result = forward_data(
        squared_velocity,
        acquisition,
        store_wavefield=True,
        wavefield_dtype=wavefield_dtype,
    )

    predicted = result.receiver_signals

    if observed_data.shape != predicted.shape:
        raise ValueError(
            "Les données observées doivent avoir la forme "
            f"{predicted.shape}, reçu {observed_data.shape}."
        )

    residual = predicted - observed_data
    objective_value = 0.5 * float(np.sum(residual**2))

    gradient, pseudo_hessian = accumulate_adjoint_gradient(
        np.asarray(squared_velocity, dtype=np.float64),
        result.wavefield,
        residual,
        dt_s=acquisition.dt_s,
        dx_m=acquisition.dx_m,
        dz_m=acquisition.dz_m,
        receiver_iz=acquisition.receiver_iz,
        receiver_ix=acquisition.receiver_ix,
        sigma=acquisition.sigma,
        accumulate_pseudo_hessian=return_pseudo_hessian,
    )

    if return_pseudo_hessian:
        return objective_value, gradient, pseudo_hessian

    return objective_value, gradient


def objective_and_gradient_multishot(
    squared_velocity: FloatArray,
    observed_datasets: list[FloatArray],
    acquisitions: list[Acquisition],
    *,
    wavefield_dtype: np.dtype = np.float64,
    return_pseudo_hessian: bool = False,
):
    """Somme la fonction objectif et le gradient sur plusieurs tirs.

    Les tirs sont indépendants : chacun fournit sa propre simulation
    directe et sa propre rétro-propagation, et leurs contributions
    s'additionnent. Multiplier les tirs élargit la couverture angulaire,
    ce qui est nécessaire pour contraindre une cible éloignée de la source.

    Args:
        squared_velocity: Carré de la vitesse, forme (nz, nx).
        observed_datasets: Données observées, une matrice par tir.
        acquisitions: Acquisitions, une par tir.
        wavefield_dtype: Type de stockage du champ direct.
        return_pseudo_hessian: Renvoie aussi le pseudo-hessien cumulé.

    Returns:
        Le couple ``(Phi, dPhi/dm)``, ou le triplet incluant le
        pseudo-hessien cumulé.

    Raises:
        ValueError: Si le nombre de jeux de données ne correspond pas au
            nombre d'acquisitions.
    """

    if len(observed_datasets) != len(acquisitions):
        raise ValueError(
            "Il faut exactement un jeu de données observées par tir."
        )

    if not acquisitions:
        raise ValueError("Il faut au moins un tir.")

    total_objective = 0.0
    total_gradient = np.zeros_like(
        np.asarray(squared_velocity, dtype=np.float64)
    )
    total_pseudo_hessian = (
        np.zeros_like(total_gradient) if return_pseudo_hessian else None
    )

    for observed, acquisition in zip(observed_datasets, acquisitions):
        outcome = objective_and_gradient(
            squared_velocity,
            observed,
            acquisition,
            wavefield_dtype=wavefield_dtype,
            return_pseudo_hessian=return_pseudo_hessian,
        )

        if return_pseudo_hessian:
            value, gradient, pseudo_hessian = outcome
            total_pseudo_hessian += pseudo_hessian
        else:
            value, gradient = outcome

        total_objective += value
        total_gradient += gradient

    if return_pseudo_hessian:
        return total_objective, total_gradient, total_pseudo_hessian

    return total_objective, total_gradient


def objective_only_multishot(
    squared_velocity: FloatArray,
    observed_datasets: list[FloatArray],
    acquisitions: list[Acquisition],
) -> float:
    """Somme la fonction objectif sur plusieurs tirs, sans gradient.

    Args:
        squared_velocity: Carré de la vitesse, forme (nz, nx).
        observed_datasets: Données observées, une matrice par tir.
        acquisitions: Acquisitions, une par tir.

    Returns:
        La valeur totale de ``Phi(m)``.
    """

    return float(
        sum(
            objective_only(squared_velocity, observed, acquisition)
            for observed, acquisition in zip(
                observed_datasets, acquisitions
            )
        )
    )


def gradient_with_respect_to_velocity(
    velocity_m_s: FloatArray,
    gradient_with_respect_to_m: FloatArray,
) -> FloatArray:
    """Convertit un gradient en ``m = v**2`` vers un gradient en vitesse.

    Args:
        velocity_m_s: Champ de vitesse, forme (nz, nx).
        gradient_with_respect_to_m: Gradient par rapport à ``m``.

    Returns:
        Le gradient par rapport à la vitesse.
    """

    return 2.0 * velocity_m_s * gradient_with_respect_to_m

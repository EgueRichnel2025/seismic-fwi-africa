"""Propagateur bas niveau partagé par le modèle direct et l'état adjoint.

Ce module reprend exactement le schéma numérique de
``seismic.wave.acoustic_2d`` mais l'expose sous une forme adaptée à
l'inversion :

- le modèle est paramétré par ``m = v**2`` (le carré de la vitesse), car
  c'est la quantité qui intervient linéairement dans l'équation ;
- le champ complet peut être conservé, ce dont l'état adjoint a besoin ;
- une passe adjointe est fournie, dérivée du schéma **discret** et non
  d'une discrétisation de l'adjoint continu.

Schéma direct, sur les points intérieurs :

    (1 + s*dt) u^{n+1} = 2 u^n - (1 - s*dt) u^{n-1}
                         + dt^2 * m * L u^n
                         + dt^2 * f^n

où ``L`` est le laplacien à cinq points avec conditions de Dirichlet
homogènes sur les quatre bords, ``s`` le profil absorbant et ``f`` le
terme source.

Adjoint discret exact du même schéma :

    (1 + s*dt) p^n = 2 p^{n+1} - (1 - s*dt) p^{n+2}
                     + dt^2 * L (m * p^{n+1})
                     - P^T r^n

L'ordre des opérateurs change : le direct applique ``m * L(.)``, l'adjoint
applique ``L(m * .)``. C'est la transposition de ``diag(m) @ L``, licite
car ``L`` est symétrique sous conditions de Dirichlet.

Gradient par rapport à ``m`` :

    dPhi/dm = - dt^2 * somme_n  p^{n+1} * (L u^n)

Les amplitudes sont synthétiques et ne représentent pas une pression
calibrée sur un site réel.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ForwardResult:
    """Sortie d'une propagation directe.

    Attributes:
        receiver_signals: Champ aux récepteurs, de forme (nt + 1, nr).
        sample_times_s: Instants d'échantillonnage, de forme (nt + 1,).
        wavefield: Champ complet (nt + 1, nz, nx) ou None s'il n'a pas
            été conservé.
        cfl_number: Nombre CFL normalisé effectivement utilisé.
    """

    receiver_signals: FloatArray
    sample_times_s: FloatArray
    wavefield: NDArray | None
    cfl_number: float


def build_sponge_profile(
    nz: int,
    nx: int,
    dx_m: float,
    dz_m: float,
    sponge_width_m: float = 40.0,
    sponge_strength_s_inv: float = 80.0,
) -> FloatArray:
    """Construit le profil absorbant utilisé par le solveur de référence.

    Les côtés et le fond sont amortis ; la surface supérieure ne l'est pas.

    Args:
        nz: Nombre de points en profondeur.
        nx: Nombre de points horizontaux.
        dx_m: Pas horizontal, en mètres.
        dz_m: Pas vertical, en mètres.
        sponge_width_m: Largeur indicative de la zone absorbante.
        sponge_strength_s_inv: Intensité de l'atténuation, en s^-1.

    Returns:
        Le profil d'amortissement, de forme (nz, nx).

    Raises:
        ValueError: Si la zone absorbante est trop large pour la grille.
    """

    sigma = np.zeros((nz, nx), dtype=np.float64)

    if sponge_width_m <= 0.0 or sponge_strength_s_inv <= 0.0:
        return sigma

    width_x = max(2, int(round(sponge_width_m / dx_m)))
    width_z = max(2, int(round(sponge_width_m / dz_m)))

    if width_x >= nx // 2 or width_z >= nz // 2:
        raise ValueError(
            "La zone absorbante est trop large par rapport à la grille."
        )

    x_indices = np.arange(nx)
    distance_to_side = np.minimum(x_indices, nx - 1 - x_indices)
    profile_x = np.clip((width_x - distance_to_side) / width_x, 0.0, 1.0)

    z_indices = np.arange(nz)
    distance_to_bottom = nz - 1 - z_indices
    profile_z = np.clip((width_z - distance_to_bottom) / width_z, 0.0, 1.0)

    sigma_x = sponge_strength_s_inv * profile_x**2
    sigma_z = sponge_strength_s_inv * profile_z**2

    return sigma_z[:, np.newaxis] + sigma_x[np.newaxis, :]


def build_source_kernel(
    x_m: FloatArray,
    z_m: FloatArray,
    source_x_m: float,
    source_z_m: float,
    source_sigma_m: float = 4.0,
) -> FloatArray:
    """Construit la distribution spatiale gaussienne normalisée de la source.

    La normalisation impose ``somme(noyau) * dx * dz = 1``, de sorte que la
    largeur physique ne dépende pas du pas de la grille.

    Args:
        x_m: Axe horizontal.
        z_m: Axe vertical.
        source_x_m: Position horizontale de la source.
        source_z_m: Profondeur de la source.
        source_sigma_m: Écart-type spatial, en mètres.

    Returns:
        Le noyau spatial, de forme (nz, nx), nul sur les bords.

    Raises:
        ValueError: Si la normalisation est impossible.
    """

    if source_sigma_m <= 0.0:
        raise ValueError(
            "La largeur spatiale de la source doit être strictement positive."
        )

    dx_m = float(x_m[1] - x_m[0])
    dz_m = float(z_m[1] - z_m[0])

    source_ix = int(np.argmin(np.abs(x_m - source_x_m)))
    source_iz = int(np.argmin(np.abs(z_m - source_z_m)))

    x_grid_m, z_grid_m = np.meshgrid(x_m, z_m)
    distance_squared_m2 = (x_grid_m - x_m[source_ix]) ** 2 + (
        z_grid_m - z_m[source_iz]
    ) ** 2

    kernel = np.exp(-0.5 * distance_squared_m2 / source_sigma_m**2)

    kernel[0, :] = 0.0
    kernel[-1, :] = 0.0
    kernel[:, 0] = 0.0
    kernel[:, -1] = 0.0

    integral = float(np.sum(kernel) * dx_m * dz_m)

    if not math.isfinite(integral) or integral <= 0.0:
        raise ValueError("Impossible de normaliser la source spatiale.")

    return kernel / integral


def _laplacian_interior(
    field: FloatArray,
    dx_m: float,
    dz_m: float,
) -> FloatArray:
    """Laplacien à cinq points évalué sur les points intérieurs."""

    center = field[1:-1, 1:-1]

    laplacian_x = (
        field[1:-1, 2:] - 2.0 * center + field[1:-1, :-2]
    ) / dx_m**2

    laplacian_z = (
        field[2:, 1:-1] - 2.0 * center + field[:-2, 1:-1]
    ) / dz_m**2

    return laplacian_x + laplacian_z


def compute_cfl_number(
    squared_velocity: FloatArray,
    dt_s: float,
    dx_m: float,
    dz_m: float,
) -> float:
    """Calcule le nombre CFL normalisé du schéma acoustique 2D.

    La valeur retournée doit rester inférieure ou égale à un.

    Args:
        squared_velocity: Carré de la vitesse, en m^2/s^2.
        dt_s: Pas temporel, en secondes.
        dx_m: Pas horizontal, en mètres.
        dz_m: Pas vertical, en mètres.

    Returns:
        Le nombre CFL normalisé.
    """

    max_velocity_m_s = float(np.sqrt(np.max(squared_velocity)))

    return (
        max_velocity_m_s
        * dt_s
        * math.sqrt(1.0 / dx_m**2 + 1.0 / dz_m**2)
    )


def propagate_forward(
    squared_velocity: FloatArray,
    *,
    dt_s: float,
    number_of_steps: int,
    dx_m: float,
    dz_m: float,
    source_kernel: FloatArray,
    source_time_series: FloatArray,
    receiver_iz: int,
    receiver_ix: NDArray[np.int_],
    sigma: FloatArray,
    store_wavefield: bool = False,
    wavefield_dtype: np.dtype = np.float32,
    check_stability: bool = True,
) -> ForwardResult:
    """Propage l'onde et enregistre le champ aux récepteurs.

    Args:
        squared_velocity: Carré de la vitesse ``m = v**2``, forme (nz, nx).
        dt_s: Pas temporel, en secondes.
        number_of_steps: Nombre de pas temporels.
        dx_m: Pas horizontal, en mètres.
        dz_m: Pas vertical, en mètres.
        source_kernel: Distribution spatiale normalisée de la source.
        source_time_series: Amplitude de la source à chaque pas, de taille
            au moins ``number_of_steps + 1``. Elle inclut déjà le
            coefficient numérique d'intensité.
        receiver_iz: Indice de profondeur des récepteurs.
        receiver_ix: Indices horizontaux des récepteurs.
        sigma: Profil absorbant, forme (nz, nx).
        store_wavefield: Conserve le champ complet, requis par l'adjoint.
        wavefield_dtype: Type de stockage du champ complet.
        check_stability: Vérifie la condition CFL avant de propager.

    Returns:
        Les signaux aux récepteurs et, si demandé, le champ complet.

    Raises:
        ValueError: Si la configuration est instable ou incohérente.
    """

    squared_velocity = np.asarray(squared_velocity, dtype=np.float64)

    if squared_velocity.ndim != 2:
        raise ValueError("Le modèle doit être une matrice (nz, nx).")

    if not np.all(np.isfinite(squared_velocity)) or np.any(
        squared_velocity <= 0.0
    ):
        raise ValueError(
            "Le carré de la vitesse doit être fini et strictement positif."
        )

    if number_of_steps < 1:
        raise ValueError("Le nombre de pas doit être strictement positif.")

    if source_time_series.size < number_of_steps + 1:
        raise ValueError(
            "La série temporelle de source est trop courte pour la durée "
            "demandée."
        )

    cfl_number = compute_cfl_number(squared_velocity, dt_s, dx_m, dz_m)

    if check_stability and cfl_number > 1.0 + 1e-12:
        raise ValueError(
            f"Pas temporel instable : CFL = {cfl_number:.4f} > 1."
        )

    nz, nx = squared_velocity.shape

    u_previous = np.zeros((nz, nx), dtype=np.float64)
    u_current = np.zeros((nz, nx), dtype=np.float64)
    u_next = np.zeros((nz, nx), dtype=np.float64)

    receiver_signals = np.zeros(
        (number_of_steps + 1, receiver_ix.size),
        dtype=np.float64,
    )

    wavefield = None

    if store_wavefield:
        wavefield = np.zeros(
            (number_of_steps + 1, nz, nx),
            dtype=wavefield_dtype,
        )

    sigma_inner = sigma[1:-1, 1:-1]
    damping_denominator = 1.0 + sigma_inner * dt_s
    damping_previous = 1.0 - sigma_inner * dt_s
    model_inner = squared_velocity[1:-1, 1:-1]
    kernel_inner = source_kernel[1:-1, 1:-1]

    for step in range(1, number_of_steps + 1):
        laplacian = _laplacian_interior(u_current, dx_m, dz_m)

        u_next.fill(0.0)

        u_next[1:-1, 1:-1] = (
            2.0 * u_current[1:-1, 1:-1]
            - damping_previous * u_previous[1:-1, 1:-1]
            + dt_s**2 * model_inner * laplacian
            + dt_s**2 * float(source_time_series[step]) * kernel_inner
        ) / damping_denominator

        receiver_signals[step, :] = u_next[receiver_iz, receiver_ix]

        u_previous, u_current, u_next = u_current, u_next, u_previous

        if wavefield is not None:
            wavefield[step] = u_current.astype(wavefield_dtype, copy=False)

    return ForwardResult(
        receiver_signals=receiver_signals,
        sample_times_s=np.arange(
            number_of_steps + 1, dtype=np.float64
        )
        * dt_s,
        wavefield=wavefield,
        cfl_number=cfl_number,
    )


def accumulate_adjoint_gradient(
    squared_velocity: FloatArray,
    wavefield: NDArray,
    residual_at_receivers: FloatArray,
    *,
    dt_s: float,
    dx_m: float,
    dz_m: float,
    receiver_iz: int,
    receiver_ix: NDArray[np.int_],
    sigma: FloatArray,
    accumulate_pseudo_hessian: bool = False,
) -> tuple[FloatArray, FloatArray | None]:
    """Rétro-propage le résidu et accumule le gradient par rapport à m.

    La récurrence adjointe est l'adjoint exact du schéma direct :

        (1 + s*dt) p^n = 2 p^{n+1} - (1 - s*dt) p^{n+2}
                         + dt^2 * L (m * p^{n+1})
                         - P^T r^n

    et le gradient s'accumule selon :

        dPhi/dm = - dt^2 * somme_n  p^{n+1} * (L u^n)

    Args:
        squared_velocity: Carré de la vitesse ``m = v**2``.
        wavefield: Champ direct complet, forme (nt + 1, nz, nx).
        residual_at_receivers: Résidu ``P u - d_obs``, forme (nt + 1, nr).
        dt_s: Pas temporel, en secondes.
        dx_m: Pas horizontal, en mètres.
        dz_m: Pas vertical, en mètres.
        receiver_iz: Indice de profondeur des récepteurs.
        receiver_ix: Indices horizontaux des récepteurs.
        sigma: Profil absorbant.
        accumulate_pseudo_hessian: Accumule également la diagonale du
            pseudo-hessien ``somme_n (L u^n)^2``, qui sert de
            préconditionneur d'illumination.

    Returns:
        Le couple ``(gradient, pseudo_hessien)``, le second valant None
        s'il n'a pas été demandé. Les deux ont la forme (nz, nx).

    Raises:
        ValueError: Si les dimensions sont incohérentes.
    """

    number_of_samples = wavefield.shape[0]

    if residual_at_receivers.shape[0] != number_of_samples:
        raise ValueError(
            "Le résidu et le champ direct doivent avoir le même nombre "
            "d'échantillons temporels."
        )

    nz, nx = squared_velocity.shape

    adjoint_current = np.zeros((nz, nx), dtype=np.float64)
    adjoint_next = np.zeros((nz, nx), dtype=np.float64)
    adjoint_new = np.zeros((nz, nx), dtype=np.float64)
    weighted = np.zeros((nz, nx), dtype=np.float64)

    gradient = np.zeros((nz, nx), dtype=np.float64)

    pseudo_hessian = (
        np.zeros((nz, nx), dtype=np.float64)
        if accumulate_pseudo_hessian
        else None
    )

    sigma_inner = sigma[1:-1, 1:-1]
    damping_denominator = 1.0 + sigma_inner * dt_s
    damping_previous = 1.0 - sigma_inner * dt_s

    # adjoint_current joue le rôle de p^{n+1}, adjoint_next celui de p^{n+2}.
    for step in range(number_of_samples - 1, -1, -1):
        # Corrélation du champ direct au pas n avec l'adjoint au pas n+1.
        if step < number_of_samples - 1:
            forward_slice = np.asarray(wavefield[step], dtype=np.float64)
            laplacian_forward = _laplacian_interior(
                forward_slice, dx_m, dz_m
            )

            gradient[1:-1, 1:-1] -= (
                dt_s**2 * adjoint_current[1:-1, 1:-1] * laplacian_forward
            )

            if pseudo_hessian is not None:
                pseudo_hessian[1:-1, 1:-1] += laplacian_forward**2

        np.multiply(squared_velocity, adjoint_current, out=weighted)
        laplacian_adjoint = _laplacian_interior(weighted, dx_m, dz_m)

        adjoint_new.fill(0.0)

        adjoint_new[1:-1, 1:-1] = (
            2.0 * adjoint_current[1:-1, 1:-1]
            - damping_previous * adjoint_next[1:-1, 1:-1]
            + dt_s**2 * laplacian_adjoint
        ) / damping_denominator

        # Injection du résidu aux positions des récepteurs.
        adjoint_new[receiver_iz, receiver_ix] -= (
            residual_at_receivers[step, :]
            / damping_denominator[receiver_iz - 1, receiver_ix - 1]
        )

        adjoint_next, adjoint_current, adjoint_new = (
            adjoint_current,
            adjoint_new,
            adjoint_next,
        )

    return gradient, pseudo_hessian

"""Propagation acoustique 2D par différences finies.

Le modèle décrit un champ scalaire u(x, z, t) dans un milieu de
vitesse v(x, z).

Hypothèses :
- milieu acoustique ;
- densité constante ;
- grille régulière ;
- différences finies d'ordre deux en espace et en temps ;
- surface supérieure à pression nulle ;
- zone absorbante simplifiée sur les côtés et en profondeur.

Les amplitudes de la source sont synthétiques et ne représentent pas
une mesure de pression calibrée sur un site réel.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .ricker import ricker_wavelet

from seismic.model.velocity_model import VelocityModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WaveSimulationResult:
    """Résultats d'une simulation de propagation acoustique."""

    times_s: FloatArray
    snapshots: FloatArray
    max_abs_per_step: FloatArray
    dt_s: float
    dx_m: float
    dz_m: float
    cfl_number: float
    source_x_m: float
    source_z_m: float
    source_frequency_hz: float
    receiver_x_m: FloatArray
    receiver_z_m: float
    receiver_signals: FloatArray
    sample_times_s: FloatArray


def _uniform_spacing(axis: FloatArray, axis_name: str) -> float:
    """Vérifie qu'un axe est régulier et retourne son pas."""

    if axis.ndim != 1 or axis.size < 3:
        raise ValueError(
            f"L'axe {axis_name} doit contenir au moins trois points."
        )

    differences = np.diff(axis)

    if np.any(differences <= 0.0):
        raise ValueError(
            f"L'axe {axis_name} doit être strictement croissant."
        )

    spacing = float(differences[0])

    if not np.allclose(
        differences,
        spacing,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            f"L'axe {axis_name} doit être régulièrement espacé."
        )

    return spacing


def simulate_acoustic_wave(
    model: VelocityModel,
    *,
    duration_s: float = 0.24,
    dt_s: float = 4.0e-4,
    source_x_m: float = 60.0,
    source_z_m: float = 6.0,
    source_frequency_hz: float = 25.0,
    source_strength_numeric: float = 1.0e6,
    source_sigma_m: float = 4.0,
    receiver_x_m: FloatArray | None = None,
    receiver_z_m: float = 4.0,
    sponge_width_m: float = 40.0,
    sponge_strength_s_inv: float = 80.0,
    snapshot_stride: int = 10,
) -> WaveSimulationResult:
    """Simule la propagation d'une onde acoustique 2D.

    Le champ est initialement nul. La source est une impulsion de Ricker
    placée près de la surface. Le terme source est injecté dans l'équation
    discrétisée.

    Args:
        model: Modèle de vitesse construit par le module géophysique.
        duration_s: Durée demandée de simulation, en secondes.
        dt_s: Pas temporel, en secondes.
        source_x_m: Position horizontale de la source, en mètres.
        source_z_m: Profondeur de la source, en mètres.
        source_frequency_hz: Fréquence centrale de la source, en hertz.
        source_strength_numeric: Coefficient numérique non calibré de la source.
        source_sigma_m: Écart-type spatial de la source gaussienne, en mètres.
        receiver_x_m: Positions horizontales des récepteurs en mètres.
            Si None, une ligne de récepteurs est créée automatiquement.
        receiver_z_m: Profondeur des récepteurs, en mètres.
        sponge_width_m: Largeur indicative de la zone absorbante, en mètres.
        sponge_strength_s_inv: Intensité de l'atténuation, en s^-1.
        snapshot_stride: Enregistre un champ toutes les N itérations.

    Returns:
        Les champs enregistrés, leurs instants, le maximum absolu du champ
        à chaque pas et les paramètres numériques de la simulation.

    Raises:
        ValueError: Si la grille, les paramètres ou la stabilité numérique
            ne sont pas valides.
    """

    velocity = np.asarray(model.velocity_m_s, dtype=np.float64)
    x_m = np.asarray(model.x_m, dtype=np.float64)
    z_m = np.asarray(model.z_m, dtype=np.float64)

    if velocity.shape != (z_m.size, x_m.size):
        raise ValueError(
            "La matrice des vitesses doit avoir la forme (Nz, Nx)."
        )

    if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0.0):
        raise ValueError(
            "Toutes les vitesses doivent être finies et positives."
        )

    dx_m = _uniform_spacing(x_m, "x")
    dz_m = _uniform_spacing(z_m, "z")

    scalar_parameters = {
        "duration_s": duration_s,
        "dt_s": dt_s,
        "source_x_m": source_x_m,
        "source_z_m": source_z_m,
        "source_frequency_hz": source_frequency_hz,
        "source_strength_numeric": source_strength_numeric,
        "source_sigma_m": source_sigma_m,
        "receiver_z_m": receiver_z_m,
        "sponge_width_m": sponge_width_m,
        "sponge_strength_s_inv": sponge_strength_s_inv,
    }

    for name, value in scalar_parameters.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"Le paramètre {name} doit être fini.")

    if duration_s <= 0.0:
        raise ValueError("La durée doit être strictement positive.")

    if dt_s <= 0.0:
        raise ValueError("Le pas temporel doit être strictement positif.")

    if source_frequency_hz <= 0.0:
        raise ValueError("La fréquence doit être strictement positive.")

    if source_strength_numeric <= 0.0:
        raise ValueError("Le coefficient numérique de la source doit être positif.")

    if source_sigma_m <= 0.0:
        raise ValueError("La largeur spatiale de la source doit être strictement positive.")

    if sponge_width_m < 0.0:
        raise ValueError("La largeur absorbante ne peut pas être négative.")

    if sponge_strength_s_inv < 0.0:
        raise ValueError("L'intensité absorbante ne peut pas être négative.")

    if (
        not isinstance(snapshot_stride, (int, np.integer))
        or isinstance(snapshot_stride, bool)
        or snapshot_stride < 1
    ):
        raise ValueError("snapshot_stride doit être un entier positif.")

    nx = x_m.size
    nz = z_m.size

    # Indices des points les plus proches de la source.
    source_ix = int(np.argmin(np.abs(x_m - source_x_m)))
    source_iz = int(np.argmin(np.abs(z_m - source_z_m)))

    if not (0.0 <= source_x_m <= x_m[-1]):
        raise ValueError("La source horizontale est hors du modèle.")

    if not (0.0 <= source_z_m <= z_m[-1]):
        raise ValueError("La profondeur de la source est hors du modèle.")

    # La source ne doit pas être située sur un bord fixe.
    if not (1 <= source_ix < nx - 1):
        raise ValueError("La source doit être à l'intérieur du modèle.")

    if not (1 <= source_iz < nz - 1):
        raise ValueError("La source doit être à l'intérieur du modèle.")

    # Positions des récepteurs virtuels.
    # Ils sont placés à l'intérieur du domaine, car le champ sur la
    # surface supérieure est imposé à zéro dans notre modèle actuel.
    if receiver_x_m is None:
        receiver_stop = (
            min(580.0, float(x_m[-1] - dx_m)) + 0.5 * dx_m
        )
        receiver_positions_x_m = np.arange(
            80.0,
            receiver_stop,
            20.0,
            dtype=np.float64,
        )
    else:
        receiver_positions_x_m = np.asarray(
            receiver_x_m,
            dtype=np.float64,
        )

    if (
        receiver_positions_x_m.ndim != 1
        or receiver_positions_x_m.size == 0
    ):
        raise ValueError(
            "Il faut fournir au moins une position de récepteur."
        )

    if not np.all(np.isfinite(receiver_positions_x_m)):
        raise ValueError(
            "Les positions des récepteurs doivent être finies."
        )

    if np.any(receiver_positions_x_m < x_m[0]) or np.any(
        receiver_positions_x_m > x_m[-1]
    ):
        raise ValueError(
            "Une position de récepteur est hors du modèle."
        )

    if not (0.0 <= receiver_z_m <= z_m[-1]):
        raise ValueError(
            "La profondeur des récepteurs est hors du modèle."
        )

    receiver_ix = np.array(
        [
            int(np.argmin(np.abs(x_m - position)))
            for position in receiver_positions_x_m
        ],
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

    # Les positions enregistrées correspondent exactement aux nœuds
    # de la grille numérique utilisés pour prélever le champ.
    receiver_positions_x_m = x_m[receiver_ix].copy()
    receiver_z_actual_m = float(z_m[receiver_iz])

    # Condition de stabilité du schéma acoustique 2D.
    vmax_m_s = float(np.max(velocity))

    maximum_stable_dt_s = 1.0 / (
        vmax_m_s
        * math.sqrt(1.0 / dx_m**2 + 1.0 / dz_m**2)
    )

    cfl_number = (
        vmax_m_s
        * dt_s
        * math.sqrt(1.0 / dx_m**2 + 1.0 / dz_m**2)
    )

    if cfl_number > 1.0 + 1e-12:
        raise ValueError(
            "Pas temporel instable : "
            f"CFL = {cfl_number:.4f} > 1. "
            f"Utilise dt <= {maximum_stable_dt_s:.8f} s."
        )

    number_of_steps = int(math.ceil(duration_s / dt_s))
    velocity_squared = velocity**2

    # Zone absorbante simplifiée sur les côtés et au fond.
    # La surface supérieure n'est pas absorbée.
    sigma = np.zeros((nz, nx), dtype=np.float64)

    if sponge_width_m > 0.0:
        width_x = max(2, int(round(sponge_width_m / dx_m)))
        width_z = max(2, int(round(sponge_width_m / dz_m)))

        if width_x >= nx // 2 or width_z >= nz // 2:
            raise ValueError(
                "La zone absorbante est trop large par rapport à la grille."
            )

        x_indices = np.arange(nx)
        distance_to_side = np.minimum(
            x_indices,
            nx - 1 - x_indices,
        )

        profile_x = np.clip(
            (width_x - distance_to_side) / width_x,
            0.0,
            1.0,
        )

        z_indices = np.arange(nz)

        # Seul le bord inférieur est absorbant dans la direction z.
        distance_to_bottom = nz - 1 - z_indices

        profile_z = np.clip(
            (width_z - distance_to_bottom) / width_z,
            0.0,
            1.0,
        )

        sigma_x = sponge_strength_s_inv * profile_x**2
        sigma_z = sponge_strength_s_inv * profile_z**2

        sigma = (
            sigma_z[:, np.newaxis]
            + sigma_x[np.newaxis, :]
        )

    # Trois états temporels successifs.
    u_previous = np.zeros((nz, nx), dtype=np.float64)
    u_current = np.zeros((nz, nx), dtype=np.float64)
    u_next = np.zeros((nz, nx), dtype=np.float64)

    snapshots = [u_current.copy()]
    snapshot_times = [0.0]

    max_abs_per_step = np.zeros(
        number_of_steps + 1,
        dtype=np.float64,
    )

    # Les signaux sont enregistrés à chaque pas temporel, même lorsque
    # les champs complets ne sont sauvegardés qu'occasionnellement.
    receiver_signals = np.zeros(
        (number_of_steps + 1, receiver_ix.size),
        dtype=np.float64,
    )
    sample_times_s = (
        np.arange(number_of_steps + 1, dtype=np.float64) * dt_s
    )

    # Délai temporel de la source de Ricker.
    source_delay_s = 1.5 / source_frequency_hz

    # Distribution spatiale gaussienne de largeur physique fixe.
    # Sa largeur ne dépend pas du pas spatial de la grille.
    x_grid_m, z_grid_m = np.meshgrid(x_m, z_m)
    distance_squared_m2 = (
        (x_grid_m - x_m[source_ix]) ** 2
        + (z_grid_m - z_m[source_iz]) ** 2
    )
    source_kernel = np.exp(
        -0.5 * distance_squared_m2 / source_sigma_m**2
    )

    # Les bords sont fixes : aucune source ne doit y être injectée.
    source_kernel[0, :] = 0.0
    source_kernel[-1, :] = 0.0
    source_kernel[:, 0] = 0.0
    source_kernel[:, -1] = 0.0

    # Normalisation de l'intégrale spatiale :
    # somme(kernel) * dx * dz = 1.
    source_integral = float(
        np.sum(source_kernel) * dx_m * dz_m
    )

    if not math.isfinite(source_integral) or source_integral <= 0.0:
        raise ValueError(
            "Impossible de normaliser la source spatiale."
        )

    source_kernel /= source_integral

    for step in range(1, number_of_steps + 1):
        time_s = step * dt_s

        center = u_current[1:-1, 1:-1]

        laplacian_x = (
            u_current[1:-1, 2:]
            - 2.0 * center
            + u_current[1:-1, :-2]
        ) / dx_m**2

        laplacian_z = (
            u_current[2:, 1:-1]
            - 2.0 * center
            + u_current[:-2, 1:-1]
        ) / dz_m**2

        # Schéma en temps d'ordre deux :
        # u(n+1) = 2u(n) - u(n-1) + dt² v² Laplacien(u(n)).
        u_next.fill(0.0)

        sigma_inner = sigma[1:-1, 1:-1]
        damping_denominator = 1.0 + sigma_inner * dt_s

        u_next[1:-1, 1:-1] = (
            2.0 * center
            - (1.0 - sigma_inner * dt_s)
            * u_previous[1:-1, 1:-1]
            + dt_s**2
            * velocity_squared[1:-1, 1:-1]
            * (laplacian_x + laplacian_z)
        ) / damping_denominator

        # Évaluation du signal de source à l'instant courant.
        ricker_value = float(
            ricker_wavelet(
                time_s,
                source_frequency_hz,
                source_delay_s,
            )
        )

        u_next[1:-1, 1:-1] += (
            dt_s**2
            * source_strength_numeric
            * ricker_value
            * source_kernel[1:-1, 1:-1]
            / damping_denominator
        )

        # L'amortissement est déjà intégré dans le schéma temporel.

        # Surface supérieure à pression nulle.
        # Les autres bords restent fixes à zéro.
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0

        # Enregistrement simultané des récepteurs à cet instant.
        receiver_signals[step, :] = u_next[
            receiver_iz,
            receiver_ix,
        ]

        max_abs_per_step[step] = float(np.max(np.abs(u_next)))

        # Avancement du temps sans recopier toute la matrice.
        u_previous, u_current, u_next = (
            u_current,
            u_next,
            u_previous,
        )

        if (
            step % snapshot_stride == 0
            or step == number_of_steps
        ):
            snapshots.append(u_current.copy())
            snapshot_times.append(time_s)

    return WaveSimulationResult(
        times_s=np.asarray(snapshot_times, dtype=np.float64),
        snapshots=np.stack(snapshots, axis=0),
        max_abs_per_step=max_abs_per_step,
        dt_s=dt_s,
        dx_m=dx_m,
        dz_m=dz_m,
        cfl_number=cfl_number,
        source_x_m=float(x_m[source_ix]),
        source_z_m=float(z_m[source_iz]),
        source_frequency_hz=source_frequency_hz,
        receiver_x_m=receiver_positions_x_m,
        receiver_z_m=receiver_z_actual_m,
        receiver_signals=receiver_signals,
        sample_times_s=sample_times_s,
    )

"""Construction d'un modèle synthétique 2D de vitesse sismique.

Le modèle représente une coupe verticale x-z du sous-sol.
Les profondeurs sont positives vers le bas.
Les vitesses sont exprimées en mètres par seconde.

Les paramètres sont des hypothèses synthétiques et ne représentent
pas des mesures géophysiques effectuées sur un site réel.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Dimensions physiques du modèle.
X_EXTENT_M = 600.0
Z_EXTENT_M = 300.0

# Paramètres géologiques synthétiques.
INTERFACE_1_M = 60.0
INTERFACE_2_M = 180.0

VELOCITY_LAYER_1_M_S = 1200.0
VELOCITY_LAYER_2_M_S = 2000.0
VELOCITY_SUBSTRATUM_M_S = 2800.0

# Anomalie elliptique à faible vitesse.
ANOMALY_CENTER_X_M = 360.0
ANOMALY_CENTER_Z_M = 125.0
ANOMALY_HALF_WIDTH_X_M = 55.0
ANOMALY_HALF_WIDTH_Z_M = 28.0
ANOMALY_VELOCITY_M_S = 1500.0


@dataclass(frozen=True)
class VelocityModel:
    """Modèle de vitesse sur une grille régulière.

    Attributes:
        x_m: Coordonnées horizontales des points de grille.
        z_m: Profondeurs des points de grille, positives vers le bas.
        velocity_m_s: Matrice des vitesses, de forme (Nz, Nx).
        anomaly_mask: Masque booléen identifiant l'anomalie.
    """

    x_m: FloatArray
    z_m: FloatArray
    velocity_m_s: FloatArray
    anomaly_mask: BoolArray


def _build_axis(extent_m: float, spacing_m: float, axis_name: str) -> FloatArray:
    """Construit un axe régulier en vérifiant la compatibilité du pas."""

    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError(
            f"Le pas {axis_name} doit être fini et strictement positif."
        )

    interval_count_float = extent_m / spacing_m
    interval_count = int(round(interval_count_float))

    if not np.isclose(
        interval_count_float,
        interval_count,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"Le pas {axis_name} = {spacing_m} m doit diviser exactement "
            f"l'étendue de {extent_m} m."
        )

    return np.linspace(
        0.0,
        extent_m,
        interval_count + 1,
        dtype=np.float64,
    )


def build_synthetic_velocity_model(
    dx_m: float = 2.0,
    dz_m: float = 2.0,
) -> VelocityModel:
    """Construit le modèle 2D synthétique de référence.

    Args:
        dx_m: Pas horizontal en mètres.
        dz_m: Pas vertical en mètres.

    Returns:
        Un objet contenant la grille, les vitesses et le masque
        de l'anomalie.

    Raises:
        ValueError: Si un pas est invalide ou incompatible avec
            les dimensions physiques du modèle.
    """

    x_m = _build_axis(X_EXTENT_M, dx_m, "dx")
    z_m = _build_axis(Z_EXTENT_M, dz_m, "dz")

    # La première dimension correspond à la profondeur z.
    # La deuxième dimension correspond à la position horizontale x.
    velocity_m_s = np.empty(
        (z_m.size, x_m.size),
        dtype=np.float64,
    )

    # Couche superficielle.
    velocity_m_s[z_m < INTERFACE_1_M, :] = VELOCITY_LAYER_1_M_S

    # Couche intermédiaire.
    middle_layer = (
        (z_m >= INTERFACE_1_M)
        & (z_m < INTERFACE_2_M)
    )
    velocity_m_s[middle_layer, :] = VELOCITY_LAYER_2_M_S

    # Substratum.
    velocity_m_s[z_m >= INTERFACE_2_M, :] = VELOCITY_SUBSTRATUM_M_S

    # Coordonnées de chaque point de la grille.
    X, Z = np.meshgrid(x_m, z_m, indexing="xy")

    # Définition géométrique d'une anomalie elliptique.
    anomaly_mask = (
        ((X - ANOMALY_CENTER_X_M) / ANOMALY_HALF_WIDTH_X_M) ** 2
        + ((Z - ANOMALY_CENTER_Z_M) / ANOMALY_HALF_WIDTH_Z_M) ** 2
        <= 1.0
    )

    # Remplacement local de la vitesse de la couche intermédiaire.
    velocity_m_s[anomaly_mask] = ANOMALY_VELOCITY_M_S

    return VelocityModel(
        x_m=x_m,
        z_m=z_m,
        velocity_m_s=velocity_m_s,
        anomaly_mask=anomaly_mask,
    )

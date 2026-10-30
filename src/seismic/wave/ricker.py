"""Ondelette de Ricker utilisée comme source sismique synthétique."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def ricker_wavelet(
    time_s: ArrayLike,
    frequency_hz: float,
    center_time_s: float | None = None,
) -> FloatArray:
    """Calcule l'amplitude d'une ondelette de Ricker.

    Formule :
        s(t) = (1 - 2*tau**2) * exp(-tau**2)

    avec :
        tau = pi * f0 * (t - t0)

    Args:
        time_s: Instants d'évaluation, en secondes.
        frequency_hz: Fréquence centrale positive, en hertz.
        center_time_s: Temps central, en secondes.
            Par défaut : 1.5 / frequency_hz.

    Returns:
        Les amplitudes numériques de l'ondelette.

    Raises:
        ValueError: Si les paramètres contiennent des valeurs invalides.
    """
    try:
        frequency = float(frequency_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "La fréquence doit être un nombre réel positif."
        ) from exc

    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError(
            "La fréquence doit être finie et strictement positive."
        )

    try:
        times = np.asarray(time_s, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Les instants doivent être des nombres réels."
        ) from exc

    if not np.all(np.isfinite(times)):
        raise ValueError(
            "Les instants doivent contenir uniquement des valeurs finies."
        )

    if center_time_s is None:
        center = 1.5 / frequency
    else:
        try:
            center = float(center_time_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Le temps central doit être un nombre réel fini."
            ) from exc

        if not math.isfinite(center):
            raise ValueError(
                "Le temps central doit être un nombre réel fini."
            )

    tau = np.pi * frequency * (times - center)
    amplitudes = (1.0 - 2.0 * tau**2) * np.exp(-tau**2)

    return np.asarray(amplitudes, dtype=np.float64)

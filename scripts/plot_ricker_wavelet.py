"""Génère la figure scientifique de la source Ricker."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from seismic.wave.ricker import ricker_wavelet


# Paramètres de l'expérience synthétique.
FREQUENCY_HZ = 25.0
CENTER_TIME_S = 1.5 / FREQUENCY_HZ
TIME_START_S = 0.0
TIME_END_S = 0.12
NUMBER_OF_SAMPLES = 1201

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "figures"
    / "ricker_wavelet.png"
)


def main() -> None:
    """Calcule la Ricker, produit la figure et affiche les paramètres."""

    times_s = np.linspace(
        TIME_START_S,
        TIME_END_S,
        NUMBER_OF_SAMPLES,
    )

    amplitudes = ricker_wavelet(
        time_s=times_s,
        frequency_hz=FREQUENCY_HZ,
        center_time_s=CENTER_TIME_S,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.plot(
        times_s * 1000.0,
        amplitudes,
        color="#2463A6",
        linewidth=2.4,
        label="Ondelette de Ricker",
    )

    ax.axhline(
        0.0,
        color="#555555",
        linewidth=0.9,
        linestyle="-",
    )

    ax.axvline(
        CENTER_TIME_S * 1000.0,
        color="#C27A27",
        linewidth=1.5,
        linestyle="--",
        label=f"Temps central : {CENTER_TIME_S * 1000:.1f} ms",
    )

    ax.scatter(
        [CENTER_TIME_S * 1000.0],
        [1.0],
        color="#C27A27",
        s=45,
        zorder=5,
    )

    ax.annotate(
        "Amplitude maximale = 1",
        xy=(CENTER_TIME_S * 1000.0, 1.0),
        xytext=(CENTER_TIME_S * 1000.0 + 8.0, 0.72),
        arrowprops={
            "arrowstyle": "->",
            "color": "#555555",
        },
    )

    ax.set_title(
        "Source sismique synthétique — ondelette de Ricker",
        pad=14,
    )
    ax.set_xlabel("Temps (ms)")
    ax.set_ylabel("Amplitude normalisée")

    ax.set_xlim(TIME_START_S * 1000.0, TIME_END_S * 1000.0)
    ax.set_ylim(-0.45, 1.15)

    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.8,
        alpha=0.65,
    )

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("=== SOURCE SISMIQUE DE RICKER ===")
    print(f"Fréquence centrale : {FREQUENCY_HZ:.1f} Hz")
    print(f"Temps central      : {CENTER_TIME_S:.4f} s")
    print(f"Nombre d'échantillons : {NUMBER_OF_SAMPLES}")
    print(f"Amplitude maximale : {np.max(amplitudes):.6f}")
    print(f"Figure enregistrée : {OUTPUT_PATH}")
    print("Statut : OK")


if __name__ == "__main__":
    main()

"""Visualisation du modèle synthétique de vitesse sismique."""

from pathlib import Path

import matplotlib

# Permet de générer l'image sans nécessiter de fenêtre graphique.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from seismic.model.velocity_model import (
    INTERFACE_1_M,
    INTERFACE_2_M,
    ANOMALY_CENTER_X_M,
    ANOMALY_CENTER_Z_M,
    build_synthetic_velocity_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "model_2d_synthetic.png"
)


def main() -> None:
    """Construit le modèle, produit la figure et affiche les informations."""

    model = build_synthetic_velocity_model()

    fig, ax = plt.subplots(
        figsize=(12, 6.5),
        constrained_layout=True,
    )

    image = ax.pcolormesh(
        model.x_m,
        model.z_m,
        model.velocity_m_s,
        shading="auto",
        cmap="viridis",
        vmin=1200.0,
        vmax=2800.0,
    )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Vitesse sismique v(x, z) [m/s]")

    # Interfaces entre les couches.
    ax.axhline(
        INTERFACE_1_M,
        color="white",
        linestyle="--",
        linewidth=1.5,
    )

    ax.axhline(
        INTERFACE_2_M,
        color="white",
        linestyle="--",
        linewidth=1.5,
    )

    # Contour de l'anomalie.
    ax.contour(
        model.x_m,
        model.z_m,
        model.anomaly_mask.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=2.0,
    )

    # Labels indicatifs des couches.
    ax.text(
        12,
        30,
        "Couche superficielle\n1200 m/s",
        color="white",
        fontsize=9,
        verticalalignment="center",
    )

    ax.text(
        12,
        120,
        "Couche intermédiaire\n2000 m/s",
        color="white",
        fontsize=9,
        verticalalignment="center",
    )

    ax.text(
        12,
        235,
        "Substratum\n2800 m/s",
        color="black",
        fontsize=9,
        verticalalignment="center",
    )

    # La source et les géophones indiquent une géométrie d'acquisition
    # envisagée pour les prochaines étapes.
    source_x_m = 60.0

    ax.scatter(
        [source_x_m],
        [0.0],
        marker="*",
        s=220,
        facecolor="crimson",
        edgecolor="black",
        linewidth=0.7,
        label="Source (position proposée)",
        zorder=5,
        clip_on=False,
    )

    geophone_x_m = np.arange(80.0, 580.0 + 0.1, 20.0)

    ax.scatter(
        geophone_x_m,
        np.zeros_like(geophone_x_m),
        marker="v",
        s=42,
        facecolor="white",
        edgecolor="black",
        linewidth=0.7,
        label="Géophones (positions proposées)",
        zorder=5,
        clip_on=False,
    )

    ax.annotate(
        "Anomalie de vitesse\n(hypothèse synthétique)",
        xy=(ANOMALY_CENTER_X_M, ANOMALY_CENTER_Z_M),
        xytext=(430, 150),
        color="white",
        fontsize=9,
        arrowprops={
            "arrowstyle": "->",
            "color": "white",
            "linewidth": 1.2,
        },
    )

    ax.set_title(
        "Modèle synthétique 2D de vitesse sismique\n"
        "Hypothèses de travail — aucune donnée de terrain",
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel("Distance horizontale x (m)")
    ax.set_ylabel("Profondeur z (m), positive vers le bas")

    ax.set_xlim(0, 600)
    ax.set_ylim(310, -15)
    ax.set_yticks(np.arange(0, 301, 50))

    ax.legend(
        loc="lower right",
        framealpha=0.95,
        fontsize=8,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=== Modèle synthétique 2D ===")
    print(f"Étendue horizontale : {model.x_m[-1]:.0f} m")
    print(f"Profondeur maximale : {model.z_m[-1]:.0f} m")
    print(f"Pas horizontal      : {model.x_m[1] - model.x_m[0]:.1f} m")
    print(f"Pas vertical        : {model.z_m[1] - model.z_m[0]:.1f} m")
    print(f"Grille              : {model.z_m.size} x {model.x_m.size}")
    print(f"Nombre de points    : {model.velocity_m_s.size}")
    print(
        "Vitesses            : "
        f"{model.velocity_m_s.min():.0f} à "
        f"{model.velocity_m_s.max():.0f} m/s"
    )
    print(
        "Points dans l'anomalie : "
        f"{np.count_nonzero(model.anomaly_mask)}"
    )
    print(f"Figure enregistrée  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Lance la simulation acoustique 2D et génère ses figures."""

from pathlib import Path

import matplotlib

# Rend possible l'exécution depuis un terminal sans fenêtre graphique.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from seismic.model import build_synthetic_velocity_model
from seismic.wave import simulate_acoustic_wave


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

ANIMATION_PATH = FIGURES_DIR / "wave_propagation.gif"


def main() -> None:
    """Exécute le calcul et enregistre les résultats."""

    model = build_synthetic_velocity_model()

    result = simulate_acoustic_wave(
        model,
        duration_s=0.24,
        dt_s=4.0e-4,
        source_x_m=60.0,
        source_z_m=6.0,
        source_frequency_hz=25.0,
        source_strength_numeric=1.0e6,
        sponge_width_m=40.0,
        sponge_strength_s_inv=80.0,
        snapshot_stride=10,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    global_amplitude = float(
        np.max(np.abs(result.snapshots))
    )

    if not np.isfinite(global_amplitude) or global_amplitude <= 0.0:
        raise RuntimeError(
            "Le champ d'onde est nul ou non fini : "
            "impossible de produire une animation valide."
        )

    # Échelle symétrique identique pour toutes les images.
    # Cela évite qu'une normalisation différente à chaque instant
    # donne une impression trompeuse d'évolution de l'amplitude.
    color_limit = global_amplitude

    fig, ax = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
    )

    image = ax.imshow(
        result.snapshots[0],
        extent=(
            model.x_m[0],
            model.x_m[-1],
            model.z_m[-1],
            model.z_m[0],
        ),
        origin="upper",
        cmap="seismic",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="bilinear",
        aspect="auto",
    )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        "Champ d'onde u (amplitude numérique normalisée)"
    )

    # Interfaces du modèle de vitesse.
    ax.axhline(
        60.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.axhline(
        180.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    # Contour de l'anomalie de vitesse pour repérer sa position.
    ax.contour(
        model.x_m,
        model.z_m,
        model.anomaly_mask.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.1,
    )

    ax.scatter(
        [result.source_x_m],
        [result.source_z_m],
        marker="*",
        s=150,
        color="gold",
        edgecolors="black",
        linewidths=0.7,
        label="Source sismique",
        zorder=5,
    )

    ax.set_title("Propagation acoustique 2D — t = 0,000 s")
    ax.set_xlabel("Distance horizontale x (m)")
    ax.set_ylabel("Profondeur z (m), positive vers le bas")
    ax.set_xlim(model.x_m[0], model.x_m[-1])
    ax.set_ylim(model.z_m[-1], model.z_m[0])
    ax.legend(loc="lower right")

    def update(frame_index: int):
        """Met à jour le champ affiché à l'écran."""

        image.set_data(result.snapshots[frame_index])

        time_s = result.times_s[frame_index]

        ax.set_title(
            f"Propagation acoustique 2D — t = {time_s:.3f} s"
        )

        return image,

    # Enregistrer également un champ à un instant intermédiaire.
    # Cet instant sera conservé séparément pour le rapport final.
    target_time_s = 0.12

    snapshot_index = int(
        np.argmin(
            np.abs(result.times_s - target_time_s)
        )
    )

    snapshot_time_s = float(result.times_s[snapshot_index])

    image.set_data(result.snapshots[snapshot_index])
    ax.set_title(
        f"Champ d'onde acoustique 2D — t = {snapshot_time_s:.3f} s"
    )

    snapshot_path = (
        FIGURES_DIR
        / f"wavefield_t_{snapshot_time_s:.3f}s.png"
    )

    fig.savefig(
        snapshot_path,
        dpi=180,
        bbox_inches="tight",
    )

    # Restaurer l'état initial pour commencer l'animation à t = 0.
    image.set_data(result.snapshots[0])
    ax.set_title("Propagation acoustique 2D — t = 0,000 s")

    animation = FuncAnimation(
        fig,
        update,
        frames=len(result.times_s),
        interval=100,
        blit=False,
        repeat=True,
    )

    animation.save(
        ANIMATION_PATH,
        writer=PillowWriter(fps=10),
        dpi=85,
    )

    plt.close(fig)

    print("=== SIMULATION ACOUSTIQUE 2D ===")
    print(f"Nombre de points       : {model.velocity_m_s.size}")
    print(f"Dimensions de la grille: {model.velocity_m_s.shape}")
    print(f"Pas horizontal         : {result.dx_m:.4f} m")
    print(f"Pas vertical           : {result.dz_m:.4f} m")
    print(f"Pas temporel           : {result.dt_s:.7f} s")
    print(f"Nombre de pas temporels: {result.max_abs_per_step.size - 1}")
    print(f"Durée effective        : {result.times_s[-1]:.4f} s")
    print(f"Fréquence source       : {result.source_frequency_hz:.1f} Hz")
    print(f"Nombre de champs sauvés: {result.snapshots.shape[0]}")
    print(f"Nombre CFL             : {result.cfl_number:.4f}")
    print(f"Amplitude maximale     : {global_amplitude:.6e}")
    print(f"Champ statique         : {snapshot_path}")
    print(f"Animation              : {ANIMATION_PATH}")


if __name__ == "__main__":
    main()

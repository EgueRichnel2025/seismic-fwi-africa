"""Construit les données synthétiques initiales du problème inverse."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from seismic.model import build_synthetic_velocity_model
from seismic.model.velocity_model import VelocityModel
from seismic.wave import simulate_acoustic_wave


ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = ROOT / "results" / "figures"
DATA_DIR = ROOT / "experiments" / "output"

OUTPUT_DATA = DATA_DIR / "synthetic_fwi_dataset.npz"
OUTPUT_METRICS = DATA_DIR / "synthetic_fwi_metrics.json"

# Paramètres communs aux deux simulations.
DURATION_S = 0.42
DT_S = 4.0e-4
SOURCE_X_M = 60.0
SOURCE_Z_M = 6.0
SOURCE_FREQUENCY_HZ = 25.0
RECEIVER_Z_M = 4.0

RECEIVER_X_M = np.arange(
    80.0,
    581.0,
    20.0,
    dtype=np.float64,
)

# L'anomalie se trouve dans la deuxième couche, dont la vitesse
# de référence est de 2000 m/s.
HOST_LAYER_VELOCITY_M_S = 2000.0


def create_initial_model(true_model: VelocityModel) -> VelocityModel:
    """Retire l'anomalie en rétablissant la vitesse de la couche hôte."""

    anomaly_mask = np.asarray(
        true_model.anomaly_mask,
        dtype=bool,
    )

    if not np.any(anomaly_mask):
        raise ValueError(
            "Le modèle de référence ne contient aucune anomalie."
        )

    true_velocity = np.asarray(
        true_model.velocity_m_s,
        dtype=np.float64,
    )

    initial_velocity = true_velocity.copy()
    initial_velocity[anomaly_mask] = HOST_LAYER_VELOCITY_M_S

    return VelocityModel(
        x_m=np.asarray(true_model.x_m, dtype=np.float64).copy(),
        z_m=np.asarray(true_model.z_m, dtype=np.float64).copy(),
        velocity_m_s=initial_velocity,
        anomaly_mask=np.zeros_like(anomaly_mask, dtype=bool),
    )


def run_forward_model(model: VelocityModel):
    """Simule les données enregistrées par les récepteurs virtuels."""

    return simulate_acoustic_wave(
        model,
        duration_s=DURATION_S,
        dt_s=DT_S,
        source_x_m=SOURCE_X_M,
        source_z_m=SOURCE_Z_M,
        source_frequency_hz=SOURCE_FREQUENCY_HZ,
        receiver_x_m=RECEIVER_X_M,
        receiver_z_m=RECEIVER_Z_M,
        sponge_width_m=40.0,
        sponge_strength_s_inv=80.0,
        snapshot_stride=25,
    )


def plot_velocity_models(
    true_model: VelocityModel,
    initial_model: VelocityModel,
    residual_velocity: np.ndarray,
) -> Path:
    """Affiche les modèles de vitesse et leur différence."""

    output = FIGURES_DIR / "true_vs_initial_velocity_model.png"

    extent = [
        float(true_model.x_m[0]),
        float(true_model.x_m[-1]),
        float(true_model.z_m[-1]),
        float(true_model.z_m[0]),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            true_model.velocity_m_s,
            "Modèle vrai synthétique",
            "Vitesse (m/s)",
            "viridis",
            1200.0,
            2800.0,
        ),
        (
            axes[1],
            initial_model.velocity_m_s,
            "Modèle initial sans anomalie",
            "Vitesse (m/s)",
            "viridis",
            1200.0,
            2800.0,
        ),
    ]

    for ax, values, title, colorbar_label, cmap, vmin, vmax in panels:
        image = ax.imshow(
            values,
            origin="upper",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Distance horizontale (m)")
        ax.set_ylabel("Profondeur (m)")
        figure.colorbar(
            image,
            ax=ax,
            label=colorbar_label,
            fraction=0.046,
            pad=0.04,
        )

    max_difference = float(np.max(np.abs(residual_velocity)))

    image = axes[2].imshow(
        residual_velocity,
        origin="upper",
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-max_difference,
        vmax=max_difference,
    )

    axes[2].set_title("Différence : vrai − initial")
    axes[2].set_xlabel("Distance horizontale (m)")
    axes[2].set_ylabel("Profondeur (m)")

    figure.colorbar(
        image,
        ax=axes[2],
        label="Différence de vitesse (m/s)",
        fraction=0.046,
        pad=0.04,
    )

    figure.suptitle(
        "Construction du problème inverse — modèle synthétique",
        fontsize=15,
    )

    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return output


def plot_seismograms(
    times_s: np.ndarray,
    receiver_x_m: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    residual: np.ndarray,
) -> Path:
    """Compare les données observées, prédites et leur résidu."""

    output = FIGURES_DIR / "observed_predicted_residual.png"

    observed_predicted_scale = float(
        max(
            np.max(np.abs(observed)),
            np.max(np.abs(predicted)),
        )
    )

    residual_scale = float(np.max(np.abs(residual)))

    if observed_predicted_scale <= 0.0:
        raise ValueError("Les sismogrammes observés et prédits sont nuls.")

    if residual_scale <= 0.0:
        residual_scale = 1.0e-30

    time_ms = times_s * 1000.0

    panels = [
        (
            observed,
            "Données observées synthétiques",
            observed_predicted_scale,
        ),
        (
            predicted,
            "Données prédites par le modèle initial",
            observed_predicted_scale,
        ),
        (
            residual,
            "Résidu : observé − prédit",
            residual_scale,
        ),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6),
        sharey=True,
        constrained_layout=True,
    )

    for ax, (data, title, scale) in zip(axes, panels):
        image = ax.pcolormesh(
            receiver_x_m,
            time_ms,
            data,
            shading="auto",
            cmap="seismic",
            vmin=-scale,
            vmax=scale,
        )

        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Position du récepteur (m)")
        ax.grid(False)

        figure.colorbar(
            image,
            ax=ax,
            label="Amplitude numérique",
            fraction=0.046,
            pad=0.04,
        )

    axes[0].set_ylabel("Temps (ms)")

    figure.suptitle(
        "Données sismiques synthétiques et résidu",
        fontsize=15,
    )

    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return output


def main() -> None:
    """Exécute les deux simulations et enregistre les résultats."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== CONSTRUCTION DU PROBLÈME INVERSE SYNTHÉTIQUE ===")

    true_model = build_synthetic_velocity_model()
    initial_model = create_initial_model(true_model)

    true_result = run_forward_model(true_model)
    initial_result = run_forward_model(initial_model)

    # Les données doivent être comparables aux mêmes instants
    # et aux mêmes positions de réception.
    if not np.allclose(
        true_result.sample_times_s,
        initial_result.sample_times_s,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Les deux simulations ne partagent pas les mêmes temps."
        )

    if not np.allclose(
        true_result.receiver_x_m,
        initial_result.receiver_x_m,
    ):
        raise RuntimeError(
            "Les positions des récepteurs ne correspondent pas."
        )

    observed = true_result.receiver_signals.copy()
    predicted = initial_result.receiver_signals.copy()
    residual = observed - predicted

    residual_norm = float(np.linalg.norm(residual))
    observed_norm = float(np.linalg.norm(observed))

    if observed_norm <= 0.0:
        raise RuntimeError("Les données observées sont nulles.")

    objective = 0.5 * residual_norm**2
    relative_data_residual = residual_norm / observed_norm
    rms_residual = float(np.sqrt(np.mean(residual**2)))

    velocity_difference = (
        true_model.velocity_m_s - initial_model.velocity_m_s
    )

    anomaly_count = int(np.count_nonzero(true_model.anomaly_mask))

    np.savez_compressed(
        OUTPUT_DATA,
        x_m=true_model.x_m,
        z_m=true_model.z_m,
        true_velocity_m_s=true_model.velocity_m_s,
        initial_velocity_m_s=initial_model.velocity_m_s,
        anomaly_mask=true_model.anomaly_mask,
        sample_times_s=true_result.sample_times_s,
        receiver_x_m=true_result.receiver_x_m,
        receiver_z_m=np.array(true_result.receiver_z_m),
        observed_data=observed,
        predicted_data=predicted,
        residual=residual,
    )

    metrics = {
        "experiment": "synthetic_initial_model_residual",
        "synthetic_data": True,
        "same_forward_solver_for_both_models": True,
        "same_spatial_grid": True,
        "same_time_step": True,
        "source_frequency_hz": SOURCE_FREQUENCY_HZ,
        "time_step_s": DT_S,
        "duration_s": DURATION_S,
        "host_layer_velocity_m_s": HOST_LAYER_VELOCITY_M_S,
        "receiver_count": int(RECEIVER_X_M.size),
        "sample_count": int(true_result.sample_times_s.size),
        "anomaly_grid_points": anomaly_count,
        "observed_norm": observed_norm,
        "residual_norm": residual_norm,
        "relative_data_residual": relative_data_residual,
        "relative_data_residual_percent": 100.0 * relative_data_residual,
        "residual_rms": rms_residual,
        "initial_objective": objective,
        "maximum_velocity_difference_m_s": float(
            np.max(np.abs(velocity_difference))
        ),
        "limitations": [
            "Modèle synthétique, pas de données de terrain.",
            "Même solveur et même grille pour les deux simulations.",
            "Cette expérience initialise le problème inverse ; "
            "elle ne constitue pas encore une inversion.",
        ],
    }

    OUTPUT_METRICS.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    velocity_figure = plot_velocity_models(
        true_model,
        initial_model,
        velocity_difference,
    )

    data_figure = plot_seismograms(
        true_result.sample_times_s,
        true_result.receiver_x_m,
        observed,
        predicted,
        residual,
    )

    print(f"Grille du modèle : {true_model.velocity_m_s.shape}")
    print(f"Nombre de récepteurs : {RECEIVER_X_M.size}")
    print(f"Nombre d'échantillons temporels : {true_result.sample_times_s.size}")
    print(f"Points de grille dans l'anomalie : {anomaly_count}")
    print()
    print("=== DIFFÉRENCE ENTRE LES DONNÉES ===")
    print(f"Norme des données observées : {observed_norm:.6e}")
    print(f"Norme du résidu             : {residual_norm:.6e}")
    print(f"Résidu relatif              : {100.0 * relative_data_residual:.4f} %")
    print(f"RMS du résidu               : {rms_residual:.6e}")
    print(f"Fonction objectif initiale  : {objective:.6e}")
    print()
    print(f"Figure des modèles : {velocity_figure}")
    print(f"Figure des sismogrammes : {data_figure}")
    print(f"Données numériques : {OUTPUT_DATA}")
    print(f"Métriques : {OUTPUT_METRICS}")
    print()
    print(
        "STATUT : données synthétiques produites. "
        "Aucune inversion n'a encore été exécutée."
    )


if __name__ == "__main__":
    main()

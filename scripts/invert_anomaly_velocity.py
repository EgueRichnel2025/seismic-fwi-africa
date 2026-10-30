"""Première inversion paramétrique : estimation de la vitesse d'une anomalie."""

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

DATA_PATH = ROOT / "experiments" / "output" / "synthetic_fwi_dataset.npz"
REPORT_PATH = ROOT / "experiments" / "output" / "anomaly_velocity_inversion.json"
FIGURE_PATH = ROOT / "results" / "figures" / "anomaly_velocity_inversion.png"

# Paramètres identiques à ceux de la génération des données de référence.
DURATION_S = 0.42
DT_S = 4.0e-4

SOURCE_X_M = 60.0
SOURCE_Z_M = 6.0
SOURCE_FREQUENCY_HZ = 25.0

RECEIVER_Z_M = 4.0
SPONGE_WIDTH_M = 40.0
SPONGE_STRENGTH_S_INV = 80.0

# Première fenêtre expérimentale d'inversion.
WINDOW_START_S = 0.24
WINDOW_END_S = 0.36

# Grille de recherche initiale.
CANDIDATE_VELOCITIES_M_S = np.arange(
    1200.0,
    2001.0,
    100.0,
)


def load_observed_data():
    """Charge les données observées et vérifie leur cohérence."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Jeu de données introuvable : {DATA_PATH}. "
            "Exécute d'abord create_synthetic_observations.py."
        )

    with np.load(DATA_PATH) as data:
        times_s = data["sample_times_s"].copy()
        receiver_x_m = data["receiver_x_m"].copy()
        observed = data["observed_data"].copy()
        x_m = data["x_m"].copy()
        z_m = data["z_m"].copy()

    if observed.shape != (times_s.size, receiver_x_m.size):
        raise ValueError(
            "Les dimensions des observations et des axes sont incompatibles."
        )

    if not np.all(np.isfinite(observed)):
        raise ValueError("Les données observées contiennent des valeurs non finies.")

    if not np.any(
        (times_s >= WINDOW_START_S)
        & (times_s <= WINDOW_END_S)
    ):
        raise ValueError("La fenêtre temporelle d'inversion est vide.")

    return times_s, receiver_x_m, observed, x_m, z_m


def build_candidate_model(
    base_model: VelocityModel,
    anomaly_velocity_m_s: float,
) -> VelocityModel:
    """Construit un modèle où seule la vitesse de l'anomalie varie."""

    anomaly_mask = np.asarray(
        base_model.anomaly_mask,
        dtype=bool,
    )

    if not np.any(anomaly_mask):
        raise ValueError(
            "Le modèle de référence ne possède aucune anomalie identifiée."
        )

    velocity = np.asarray(
        base_model.velocity_m_s,
        dtype=np.float64,
    ).copy()

    velocity[anomaly_mask] = anomaly_velocity_m_s

    return VelocityModel(
        x_m=np.asarray(base_model.x_m, dtype=np.float64).copy(),
        z_m=np.asarray(base_model.z_m, dtype=np.float64).copy(),
        velocity_m_s=velocity,
        anomaly_mask=anomaly_mask.copy(),
    )


def evaluate_candidate(
    anomaly_velocity_m_s: float,
    base_model: VelocityModel,
    receiver_x_m: np.ndarray,
    observed_window: np.ndarray,
) -> dict[str, float]:
    """Simule un modèle candidat et évalue son écart aux observations."""

    candidate_model = build_candidate_model(
        base_model,
        anomaly_velocity_m_s,
    )

    result = simulate_acoustic_wave(
        candidate_model,
        duration_s=DURATION_S,
        dt_s=DT_S,
        source_x_m=SOURCE_X_M,
        source_z_m=SOURCE_Z_M,
        source_frequency_hz=SOURCE_FREQUENCY_HZ,
        receiver_x_m=receiver_x_m,
        receiver_z_m=RECEIVER_Z_M,
        sponge_width_m=SPONGE_WIDTH_M,
        sponge_strength_s_inv=SPONGE_STRENGTH_S_INV,
        snapshot_stride=25,
    )

    predicted = result.receiver_signals

    if predicted.shape[0] < observed_window.shape[0]:
        raise ValueError(
            "La simulation candidate est plus courte que les observations."
        )

    # Le masque temporel sera appliqué par l'appelant : cette fonction
    # reçoit ici les observations correspondant à la fenêtre retenue.
    if predicted.shape[1] != observed_window.shape[1]:
        raise ValueError(
            "Le nombre de récepteurs simulés est incorrect."
        )

    # Les observations de la fenêtre sont alignées sur les échantillons
    # de la simulation complète.
    return {
        "velocity_m_s": float(anomaly_velocity_m_s),
        "objective": float("nan"),
        "relative_residual": float("nan"),
    }


def main() -> None:
    """Évalue une grille de vitesses candidates et sauvegarde les résultats."""

    times_s, receiver_x_m, observed, x_m, z_m = load_observed_data()

    time_mask = (
        (times_s >= WINDOW_START_S)
        & (times_s <= WINDOW_END_S)
    )

    observed_window = observed[time_mask, :]

    observed_norm = float(np.linalg.norm(observed_window))

    if observed_norm <= 0.0:
        raise ValueError(
            "La norme des observations dans la fenêtre choisie est nulle."
        )

    base_model = build_synthetic_velocity_model()

    if (
        base_model.velocity_m_s.shape != (z_m.size, x_m.size)
        or not np.allclose(base_model.x_m, x_m)
        or not np.allclose(base_model.z_m, z_m)
    ):
        raise ValueError(
            "La grille du modèle ne correspond pas à celle des observations."
        )

    results = []

    print("=== PREMIÈRE INVERSION PARAMÉTRIQUE ===")
    print(
        f"Fenêtre temporelle : "
        f"{WINDOW_START_S:.3f} à {WINDOW_END_S:.3f} s"
    )
    print(f"Nombre de récepteurs : {receiver_x_m.size}")
    print(f"Nombre d'échantillons retenus : {observed_window.shape[0]}")
    print()
    print("Vitesse candidate | Fonction objectif | Résidu relatif (%)")

    for velocity_m_s in CANDIDATE_VELOCITIES_M_S:
        model = build_candidate_model(
            base_model,
            float(velocity_m_s),
        )

        simulation = simulate_acoustic_wave(
            model,
            duration_s=DURATION_S,
            dt_s=DT_S,
            source_x_m=SOURCE_X_M,
            source_z_m=SOURCE_Z_M,
            source_frequency_hz=SOURCE_FREQUENCY_HZ,
            receiver_x_m=receiver_x_m,
            receiver_z_m=RECEIVER_Z_M,
            sponge_width_m=SPONGE_WIDTH_M,
            sponge_strength_s_inv=SPONGE_STRENGTH_S_INV,
            snapshot_stride=25,
        )

        if not np.allclose(
            simulation.sample_times_s,
            times_s,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Les instants de simulation diffèrent des observations."
            )

        predicted_window = simulation.receiver_signals[time_mask, :]
        residual = observed_window - predicted_window

        residual_norm = float(np.linalg.norm(residual))
        objective = 0.5 * residual_norm**2
        relative_residual = residual_norm / observed_norm

        candidate = {
            "velocity_m_s": float(velocity_m_s),
            "objective": objective,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "relative_residual_percent": 100.0 * relative_residual,
        }

        results.append(candidate)

        print(
            f"{velocity_m_s:16.1f} | "
            f"{objective:17.8e} | "
            f"{100.0 * relative_residual:18.6f}"
        )

    best = min(results, key=lambda item: item["objective"])

    objectives = np.array(
        [item["objective"] for item in results],
        dtype=np.float64,
    )

    baseline = next(
        (
            item["objective"]
            for item in results
            if item["velocity_m_s"] == 2000.0
        ),
        None,
    )

    if baseline is None or baseline <= 0.0:
        normalized_objectives = np.full_like(objectives, np.nan)
    else:
        normalized_objectives = objectives / baseline

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(9.5, 5.8))

    ax.plot(
        CANDIDATE_VELOCITIES_M_S,
        objectives,
        marker="o",
        linewidth=1.9,
        label="Fonction objectif",
    )

    ax.axvline(
        best["velocity_m_s"],
        linestyle="--",
        linewidth=1.5,
        label=f"Meilleure candidate : {best['velocity_m_s']:.0f} m/s",
    )

    ax.set_title("Estimation de la vitesse de l'anomalie")
    ax.set_xlabel("Vitesse candidate (m/s)")
    ax.set_ylabel("Fonction objectif Φ")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    report = {
        "experiment": "synthetic_parametric_anomaly_velocity_inversion",
        "synthetic_data": True,
        "fixed_parameters": {
            "anomaly_geometry_known": True,
            "source_frequency_hz": SOURCE_FREQUENCY_HZ,
            "time_step_s": DT_S,
            "duration_s": DURATION_S,
            "window_start_s": WINDOW_START_S,
            "window_end_s": WINDOW_END_S,
            "receiver_count": int(receiver_x_m.size),
        },
        "candidate_results": results,
        "best_candidate": best,
        "baseline_objective_at_2000_m_s": baseline,
        "normalized_objectives": [
            (
                float(value)
                if np.isfinite(value)
                else None
            )
            for value in normalized_objectives
        ],
        "limitations": [
            "L'expérience estime seulement la vitesse de l'anomalie.",
            "La position et la géométrie de l'anomalie sont fixées.",
            "Les données et les prédictions utilisent le même solveur.",
            "Les données sont synthétiques, sans bruit de terrain.",
            "Cette recherche discrète n'est pas une FWI complète.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== MEILLEURE CANDIDATE ===")
    print(f"Vitesse estimée : {best['velocity_m_s']:.1f} m/s")
    print(f"Fonction objectif : {best['objective']:.8e}")
    print(f"Résidu relatif : {best['relative_residual_percent']:.6f} %")
    print()
    print(f"Figure : {FIGURE_PATH}")
    print(f"Rapport : {REPORT_PATH}")
    print()
    print(
        "Attention : première estimation paramétrique contrôlée, "
        "pas encore une reconstruction générale du sous-sol."
    )


if __name__ == "__main__":
    main()

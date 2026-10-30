"""Première inversion locale de la géométrie et de la vitesse d'une anomalie."""

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
REPORT_PATH = ROOT / "experiments" / "output" / "anomaly_geometry_inversion.json"
FIGURE_PATH = ROOT / "results" / "figures" / "anomaly_geometry_inversion.png"

# Paramètres conservés identiques à ceux des données de référence.
DURATION_S = 0.42
DT_S = 4.0e-4
SOURCE_X_M = 60.0
SOURCE_Z_M = 6.0
SOURCE_FREQUENCY_HZ = 25.0
RECEIVER_Z_M = 4.0

WINDOW_START_S = 0.24
WINDOW_END_S = 0.36

# Géométrie de l'ellipse conservée fixe pour cette première étape.
SEMI_AXIS_X_M = 55.0
SEMI_AXIS_Z_M = 28.0
HOST_VELOCITY_M_S = 2000.0

# Paramètres : [centre x, centre z, vitesse de l'anomalie].
INITIAL_PARAMETERS = np.array([344.0, 136.0, 1700.0])

# Perturbations utilisées pour approximer les dérivées.
FINITE_DIFFERENCE_STEPS = np.array([4.0, 4.0, 50.0])

# Échelles numériques des paramètres pour le système de moindres carrés.
PARAMETER_SCALES = np.array([20.0, 20.0, 200.0])

BOUNDS_LOWER = np.array([280.0, 90.0, 1000.0])
BOUNDS_UPPER = np.array([440.0, 160.0, 2000.0])


def load_reference_data():
    """Charge les observations synthétiques et les temps d'acquisition."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Données introuvables : {DATA_PATH}. "
            "Exécute d'abord scripts/create_synthetic_observations.py."
        )

    with np.load(DATA_PATH) as data:
        times_s = data["sample_times_s"].copy()
        receiver_x_m = data["receiver_x_m"].copy()
        observed = data["observed_data"].copy()

    if observed.shape != (times_s.size, receiver_x_m.size):
        raise ValueError("Dimensions des données incompatibles.")

    if not np.all(np.isfinite(observed)):
        raise ValueError("Les observations contiennent des valeurs non finies.")

    time_mask = (
        (times_s >= WINDOW_START_S)
        & (times_s <= WINDOW_END_S)
    )

    if not np.any(time_mask):
        raise ValueError("La fenêtre temporelle d'inversion est vide.")

    observed_window = observed[time_mask].ravel()

    if np.linalg.norm(observed_window) <= 0.0:
        raise ValueError("Les observations de la fenêtre sont nulles.")

    return times_s, receiver_x_m, observed_window, time_mask


def build_candidate_model(
    parameters: np.ndarray,
    base_model: VelocityModel,
) -> VelocityModel:
    """Construit un modèle candidat depuis trois paramètres."""

    center_x_m, center_z_m, anomaly_velocity_m_s = parameters

    x_grid, z_grid = np.meshgrid(
        base_model.x_m,
        base_model.z_m,
    )

    original_mask = np.asarray(
        base_model.anomaly_mask,
        dtype=bool,
    )

    # Rétablir la vitesse hôte dans la région de l'anomalie de référence.
    background_velocity = np.asarray(
        base_model.velocity_m_s,
        dtype=np.float64,
    ).copy()

    background_velocity[original_mask] = HOST_VELOCITY_M_S

    # Construire une nouvelle ellipse à partir des paramètres candidats.
    candidate_mask = (
        ((x_grid - center_x_m) / SEMI_AXIS_X_M) ** 2
        + ((z_grid - center_z_m) / SEMI_AXIS_Z_M) ** 2
        <= 1.0
    )

    candidate_velocity = background_velocity.copy()
    candidate_velocity[candidate_mask] = anomaly_velocity_m_s

    return VelocityModel(
        x_m=np.asarray(base_model.x_m, dtype=np.float64).copy(),
        z_m=np.asarray(base_model.z_m, dtype=np.float64).copy(),
        velocity_m_s=candidate_velocity,
        anomaly_mask=candidate_mask,
    )


def forward_data(
    parameters: np.ndarray,
    base_model: VelocityModel,
    receiver_x_m: np.ndarray,
    time_mask: np.ndarray,
) -> np.ndarray:
    """Calcule les signaux prédits dans la fenêtre retenue."""

    model = build_candidate_model(parameters, base_model)

    result = simulate_acoustic_wave(
        model,
        duration_s=DURATION_S,
        dt_s=DT_S,
        source_x_m=SOURCE_X_M,
        source_z_m=SOURCE_Z_M,
        source_frequency_hz=SOURCE_FREQUENCY_HZ,
        receiver_x_m=receiver_x_m,
        receiver_z_m=RECEIVER_Z_M,
        sponge_width_m=40.0,
        sponge_strength_s_inv=80.0,
        snapshot_stride=25,
    )

    return result.receiver_signals[time_mask].ravel()


def objective(residual: np.ndarray) -> float:
    """Fonction objectif des moindres carrés."""

    return 0.5 * float(np.dot(residual, residual))


def main() -> None:
    """Construit la Jacobienne et teste une correction locale."""

    times_s, receiver_x_m, observed, time_mask = load_reference_data()
    base_model = build_synthetic_velocity_model()

    parameters = INITIAL_PARAMETERS.copy()

    print("=== INVERSION LOCALE : GÉOMÉTRIE ET VITESSE ===")
    print(
        f"Fenêtre temporelle : {WINDOW_START_S:.3f}–"
        f"{WINDOW_END_S:.3f} s"
    )
    print(f"Nombre de récepteurs : {receiver_x_m.size}")
    print(f"Nombre d'observations retenues : {observed.size}")
    print()
    print("Paramètres initiaux :")
    print(f"  Centre x : {parameters[0]:.2f} m")
    print(f"  Centre z : {parameters[1]:.2f} m")
    print(f"  Vitesse  : {parameters[2]:.2f} m/s")
    print()

    predicted = forward_data(
        parameters,
        base_model,
        receiver_x_m,
        time_mask,
    )

    residual = observed - predicted
    initial_objective = objective(residual)

    # Différences finies : une simulation supplémentaire par paramètre.
    jacobian = np.empty(
        (observed.size, parameters.size),
        dtype=np.float64,
    )

    for column, step in enumerate(FINITE_DIFFERENCE_STEPS):
        perturbed_parameters = parameters.copy()
        perturbed_parameters[column] += step

        perturbed_prediction = forward_data(
            perturbed_parameters,
            base_model,
            receiver_x_m,
            time_mask,
        )

        jacobian[:, column] = (
            perturbed_prediction - predicted
        ) / step

    # Mise à l'échelle des colonnes pour équilibrer les unités.
    scaled_jacobian = jacobian * PARAMETER_SCALES[np.newaxis, :]

    singular_values = np.linalg.svd(
        scaled_jacobian,
        compute_uv=False,
    )

    if singular_values[-1] <= singular_values[0] * 1.0e-12:
        print(
            "AVERTISSEMENT : la Jacobienne est mal conditionnée. "
            "La correction peut être instable."
        )

    condition_number = float(
        singular_values[0] / max(singular_values[-1], 1.0e-30)
    )

    # Résoudre J_scaled * q ≈ résidu, puis retrouver Δp.
    scaled_update, _, rank, _ = np.linalg.lstsq(
        scaled_jacobian,
        residual,
        rcond=None,
    )

    update = PARAMETER_SCALES * scaled_update

    print("=== DIAGNOSTIC DE LA JACOBIENNE ===")
    print(f"Dimensions de J : {jacobian.shape}")
    print(f"Rang estimé : {rank}")
    print(f"Conditionnement de J mise à l'échelle : {condition_number:.6e}")
    print("Valeurs singulières :")
    for value in singular_values:
        print(f"  {value:.6e}")

    print()
    print("Correction proposée par les moindres carrés :")
    print(f"  Δx : {update[0]:+.6f} m")
    print(f"  Δz : {update[1]:+.6f} m")
    print(f"  Δv : {update[2]:+.6f} m/s")

    # Recherche linéaire : retenir la première correction admissible
    # qui diminue effectivement la fonction objectif.
    accepted_parameters = parameters.copy()
    accepted_prediction = predicted.copy()
    accepted_objective = initial_objective
    accepted_alpha = 0.0

    trial_alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]

    for alpha in trial_alphas:
        candidate_parameters = parameters + alpha * update

        if np.any(candidate_parameters < BOUNDS_LOWER):
            continue

        if np.any(candidate_parameters > BOUNDS_UPPER):
            continue

        candidate_prediction = forward_data(
            candidate_parameters,
            base_model,
            receiver_x_m,
            time_mask,
        )

        candidate_residual = observed - candidate_prediction
        candidate_objective = objective(candidate_residual)

        print(
            f"Essai alpha={alpha:.5f} : "
            f"Phi={candidate_objective:.8e}"
        )

        if candidate_objective < accepted_objective:
            accepted_parameters = candidate_parameters
            accepted_prediction = candidate_prediction
            accepted_objective = candidate_objective
            accepted_alpha = alpha
            break

    accepted = accepted_alpha > 0.0

    if accepted:
        accepted_residual = observed - accepted_prediction
    else:
        accepted_residual = residual.copy()
        print(
            "Aucune correction testée n'a diminué la fonction objectif. "
            "Nous conserverons le point initial pour analyser la sensibilité."
        )

    observed_norm = float(np.linalg.norm(observed))
    relative_residual_initial = float(
        np.linalg.norm(residual) / observed_norm
    )
    relative_residual_final = float(
        np.linalg.norm(accepted_residual) / observed_norm
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "local_gauss_newton_anomaly_geometry",
        "synthetic_data": True,
        "parameter_names": ["center_x_m", "center_z_m", "anomaly_velocity_m_s"],
        "initial_parameters": parameters.tolist(),
        "jacobian_shape": list(jacobian.shape),
        "jacobian_rank": int(rank),
        "scaled_jacobian_condition_number": condition_number,
        "singular_values": singular_values.tolist(),
        "finite_difference_steps": FINITE_DIFFERENCE_STEPS.tolist(),
        "proposed_update": update.tolist(),
        "accepted": accepted,
        "accepted_line_search_alpha": accepted_alpha,
        "final_parameters": accepted_parameters.tolist(),
        "initial_objective": initial_objective,
        "final_objective": accepted_objective,
        "initial_relative_residual": relative_residual_initial,
        "final_relative_residual": relative_residual_final,
        "limitations": [
            "Les demi-axes de l'ellipse sont supposés connus.",
            "Données synthétiques générées avec le même solveur.",
            "Une seule correction locale est évaluée.",
            "Le résultat ne constitue pas encore une FWI complète.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    figure, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.bar(
        ["Avant la correction", "Après la correction"],
        [initial_objective, accepted_objective],
    )

    ax.set_title("Effet de la correction locale sur la fonction objectif")
    ax.set_ylabel("Fonction objectif Φ")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print()
    print("=== RÉSULTAT DE LA CORRECTION LOCALE ===")
    print(f"Correction acceptée : {'oui' if accepted else 'non'}")
    print(f"Facteur alpha : {accepted_alpha:.5f}")
    print(f"Fonction objectif initiale : {initial_objective:.8e}")
    print(f"Fonction objectif finale : {accepted_objective:.8e}")
    print(f"Résidu relatif initial : {100.0 * relative_residual_initial:.6f} %")
    print(f"Résidu relatif final : {100.0 * relative_residual_final:.6f} %")
    print()
    print(f"Figure : {FIGURE_PATH}")
    print(f"Rapport : {REPORT_PATH}")
    print(
        "Note : il s'agit d'une première correction locale. "
        "La convergence complète reste à démontrer."
    )


if __name__ == "__main__":
    main()

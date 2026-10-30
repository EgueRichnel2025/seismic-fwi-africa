"""Inversion locale itérative de la géométrie et de la vitesse d'une anomalie."""

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
REPORT_PATH = (
    ROOT
    / "experiments"
    / "output"
    / "anomaly_geometry_inversion_iterative.json"
)
FIGURE_PATH = (
    ROOT
    / "results"
    / "figures"
    / "anomaly_geometry_inversion_iterative.png"
)

# Paramètres utilisés pour produire les données de référence.
DURATION_S = 0.42
DT_S = 4.0e-4
SOURCE_X_M = 60.0
SOURCE_Z_M = 6.0
SOURCE_FREQUENCY_HZ = 25.0
RECEIVER_Z_M = 4.0

WINDOW_START_S = 0.24
WINDOW_END_S = 0.36

# Hypothèses de géométrie : les demi-axes restent fixes.
SEMI_AXIS_X_M = 55.0
SEMI_AXIS_Z_M = 28.0
HOST_VELOCITY_M_S = 2000.0

# Paramètres : centre x, centre z, vitesse de l'anomalie.
INITIAL_PARAMETERS = np.array([344.0, 136.0, 1700.0])
TRUE_PARAMETERS = np.array([360.0, 125.0, 1500.0])

# Pas pour les différences finies et échelles des paramètres.
FINITE_DIFFERENCE_STEPS = np.array([4.0, 4.0, 50.0])
PARAMETER_SCALES = np.array([20.0, 20.0, 200.0])

# Bornes choisies pour conserver l'anomalie dans la deuxième couche.
BOUNDS_LOWER = np.array([280.0, 90.0, 1000.0])
BOUNDS_UPPER = np.array([440.0, 150.0, 2000.0])

MAX_ITERATIONS = 6
LINE_SEARCH_ALPHAS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
RELATIVE_IMPROVEMENT_TOLERANCE = 1.0e-4


def load_reference_data():
    """Charge les observations synthétiques et définit la fenêtre temporelle."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Données introuvables : {DATA_PATH}. "
            "Exécute d'abord create_synthetic_observations.py."
        )

    with np.load(DATA_PATH) as data:
        times_s = data["sample_times_s"].copy()
        receiver_x_m = data["receiver_x_m"].copy()
        observed = data["observed_data"].copy()
        x_m = data["x_m"].copy()
        z_m = data["z_m"].copy()

    if observed.shape != (times_s.size, receiver_x_m.size):
        raise ValueError("Dimensions incompatibles dans les données.")

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

    return times_s, receiver_x_m, observed_window, x_m, z_m, time_mask


def build_candidate_model(
    parameters: np.ndarray,
    base_model: VelocityModel,
) -> VelocityModel:
    """Construit un modèle candidat avec une ellipse paramétrique."""

    center_x_m, center_z_m, anomaly_velocity_m_s = parameters

    x_grid, z_grid = np.meshgrid(base_model.x_m, base_model.z_m)

    original_mask = np.asarray(base_model.anomaly_mask, dtype=bool)

    background_velocity = np.asarray(
        base_model.velocity_m_s,
        dtype=np.float64,
    ).copy()

    # Supprimer l'anomalie de référence afin de reconstruire
    # systématiquement le modèle à partir des paramètres candidats.
    background_velocity[original_mask] = HOST_VELOCITY_M_S

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
    """Calcule les données prédites dans la fenêtre d'inversion."""

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


def calculate_jacobian(
    parameters: np.ndarray,
    prediction: np.ndarray,
    base_model: VelocityModel,
    receiver_x_m: np.ndarray,
    time_mask: np.ndarray,
) -> np.ndarray:
    """Approxime la Jacobienne par différences finies."""

    jacobian = np.empty(
        (prediction.size, parameters.size),
        dtype=np.float64,
    )

    for column, step in enumerate(FINITE_DIFFERENCE_STEPS):
        delta = float(step)

        # Préférer une différence avant, puis arrière si la borne est atteinte.
        if parameters[column] + delta <= BOUNDS_UPPER[column]:
            signed_delta = delta
        elif parameters[column] - delta >= BOUNDS_LOWER[column]:
            signed_delta = -delta
        else:
            raise ValueError(
                f"Impossible de perturber le paramètre {column} "
                "dans les bornes autorisées."
            )

        perturbed = parameters.copy()
        perturbed[column] += signed_delta

        perturbed_prediction = forward_data(
            perturbed,
            base_model,
            receiver_x_m,
            time_mask,
        )

        jacobian[:, column] = (
            perturbed_prediction - prediction
        ) / signed_delta

    if not np.all(np.isfinite(jacobian)):
        raise ValueError("La Jacobienne contient des valeurs non finies.")

    return jacobian


def objective(residual: np.ndarray) -> float:
    """Fonction objectif des moindres carrés."""

    return 0.5 * float(np.dot(residual, residual))


def main() -> None:
    """Exécute plusieurs corrections locales avec recherche linéaire."""

    (
        times_s,
        receiver_x_m,
        observed,
        x_m,
        z_m,
        time_mask,
    ) = load_reference_data()

    base_model = build_synthetic_velocity_model()

    if (
        base_model.velocity_m_s.shape != (z_m.size, x_m.size)
        or not np.allclose(base_model.x_m, x_m)
        or not np.allclose(base_model.z_m, z_m)
    ):
        raise ValueError(
            "Le maillage du modèle diffère de celui des données observées."
        )

    parameters = INITIAL_PARAMETERS.copy()
    history = []
    final_jacobian_diagnostics = {}

    observed_norm = float(np.linalg.norm(observed))

    print("=== INVERSION ITÉRATIVE DE L'ANOMALIE ===")
    print(f"Fenêtre : {WINDOW_START_S:.3f}–{WINDOW_END_S:.3f} s")
    print(f"Observations : {observed.size}")
    print()
    print("Paramètres de référence connus pour cette expérience synthétique :")
    print(
        f"  x = {TRUE_PARAMETERS[0]:.1f} m, "
        f"z = {TRUE_PARAMETERS[1]:.1f} m, "
        f"v = {TRUE_PARAMETERS[2]:.1f} m/s"
    )
    print()
    print(
        "Itération | x (m) | z (m) | v (m/s) | "
        "Phi avant | Phi après | alpha"
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        prediction = forward_data(
            parameters,
            base_model,
            receiver_x_m,
            time_mask,
        )

        residual = observed - prediction
        current_objective = objective(residual)

        jacobian = calculate_jacobian(
            parameters,
            prediction,
            base_model,
            receiver_x_m,
            time_mask,
        )

        scaled_jacobian = jacobian * PARAMETER_SCALES[np.newaxis, :]

        singular_values = np.linalg.svd(
            scaled_jacobian,
            compute_uv=False,
        )

        condition_number = float(
            singular_values[0]
            / max(float(singular_values[-1]), 1.0e-30)
        )

        scaled_update, _, rank, _ = np.linalg.lstsq(
            scaled_jacobian,
            residual,
            rcond=None,
        )

        # Limiter les déplacements normalisés pour éviter une correction
        # locale trop grande.
        scaled_update = np.clip(scaled_update, -2.0, 2.0)
        update = PARAMETER_SCALES * scaled_update

        accepted = False
        accepted_alpha = 0.0
        accepted_parameters = parameters.copy()
        accepted_objective = current_objective

        for alpha in LINE_SEARCH_ALPHAS:
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

            if candidate_objective < current_objective:
                accepted = True
                accepted_alpha = alpha
                accepted_parameters = candidate_parameters
                accepted_objective = candidate_objective
                break

        history.append(
            {
                "iteration": iteration,
                "parameters_before": parameters.tolist(),
                "objective_before": current_objective,
                "proposed_update": update.tolist(),
                "accepted": accepted,
                "alpha": accepted_alpha,
                "parameters_after": accepted_parameters.tolist(),
                "objective_after": accepted_objective,
                "jacobian_rank": int(rank),
                "scaled_jacobian_condition_number": condition_number,
                "singular_values": singular_values.tolist(),
            }
        )

        print(
            f"{iteration:9d} | "
            f"{parameters[0]:6.2f} | "
            f"{parameters[1]:6.2f} | "
            f"{parameters[2]:7.2f} | "
            f"{current_objective:.4e} | "
            f"{accepted_objective:.4e} | "
            f"{accepted_alpha:.5f}"
        )

        final_jacobian_diagnostics = {
            "rank": int(rank),
            "condition_number": condition_number,
            "singular_values": singular_values.tolist(),
        }

        if not accepted:
            print(
                "Arrêt : aucune correction testée ne diminue "
                "la fonction objectif."
            )
            break

        relative_improvement = (
            current_objective - accepted_objective
        ) / max(current_objective, 1.0e-30)

        actual_scaled_step = (
            accepted_alpha * update / PARAMETER_SCALES
        )

        parameters = accepted_parameters

        if (
            relative_improvement < RELATIVE_IMPROVEMENT_TOLERANCE
            or np.linalg.norm(actual_scaled_step) < 1.0e-3
        ):
            print("Arrêt : critère de convergence locale atteint.")
            break

    final_prediction = forward_data(
        parameters,
        base_model,
        receiver_x_m,
        time_mask,
    )
    final_residual = observed - final_prediction
    final_objective = objective(final_residual)

    initial_prediction = forward_data(
        INITIAL_PARAMETERS,
        base_model,
        receiver_x_m,
        time_mask,
    )
    initial_residual = observed - initial_prediction
    initial_objective = objective(initial_residual)

    initial_relative_residual = (
        float(np.linalg.norm(initial_residual)) / observed_norm
    )
    final_relative_residual = (
        float(np.linalg.norm(final_residual)) / observed_norm
    )

    reduction_percent = (
        100.0 * (initial_objective - final_objective)
        / max(initial_objective, 1.0e-30)
    )

    report = {
        "experiment": "iterative_local_anomaly_geometry_inversion",
        "synthetic_data": True,
        "initial_parameters": INITIAL_PARAMETERS.tolist(),
        "reference_parameters": TRUE_PARAMETERS.tolist(),
        "final_parameters": parameters.tolist(),
        "fixed_semi_axes_m": [SEMI_AXIS_X_M, SEMI_AXIS_Z_M],
        "time_window_s": [WINDOW_START_S, WINDOW_END_S],
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "objective_reduction_percent": reduction_percent,
        "initial_relative_residual": initial_relative_residual,
        "final_relative_residual": final_relative_residual,
        "iterations": history,
        "final_jacobian_diagnostics": final_jacobian_diagnostics,
        "limitations": [
            "Données synthétiques produites par le même solveur.",
            "Demi-axes de l'anomalie fixés.",
            "Une seule anomalie elliptique est considérée.",
            "L'inversion est locale et peut dépendre de l'initialisation.",
            "Le résultat ne constitue pas une FWI complète.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    objective_history = [initial_objective]

    for record in history:
        if record["accepted"]:
            objective_history.append(record["objective_after"])

    figure, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        range(len(objective_history)),
        objective_history,
        marker="o",
        linewidth=1.9,
    )

    ax.set_title("Convergence de la fonction objectif")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Fonction objectif Phi")
    ax.grid(True, linestyle=":", alpha=0.6)

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print()
    print("=== RÉSULTAT FINAL ===")
    print(f"x estimé : {parameters[0]:.3f} m")
    print(f"z estimé : {parameters[1]:.3f} m")
    print(f"Vitesse estimée : {parameters[2]:.3f} m/s")
    print()
    print(f"Fonction objectif initiale : {initial_objective:.8e}")
    print(f"Fonction objectif finale : {final_objective:.8e}")
    print(f"Réduction de la fonction objectif : {reduction_percent:.3f} %")
    print(f"Résidu relatif initial : {100.0 * initial_relative_residual:.6f} %")
    print(f"Résidu relatif final : {100.0 * final_relative_residual:.6f} %")
    print()
    print(f"Figure : {FIGURE_PATH}")
    print(f"Rapport : {REPORT_PATH}")
    print(
        "Note : une baisse de la fonction objectif indique une meilleure "
        "adéquation aux données synthétiques ; elle ne garantit pas "
        "à elle seule une reconstruction unique ou exacte."
    )


if __name__ == "__main__":
    main()

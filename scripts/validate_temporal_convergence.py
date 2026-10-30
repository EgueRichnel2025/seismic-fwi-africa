"""Étudie l'influence du pas temporel à grille spatiale constante."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from seismic.model.velocity_model import VelocityModel
from seismic.wave import simulate_acoustic_wave


VELOCITY_M_S = 1500.0
WIDTH_M = 240.0
DEPTH_M = 120.0
DX_M = 2.0
SOURCE_X_M = 60.0
SOURCE_Z_M = 50.0
FREQUENCY_HZ = 25.0
DURATION_S = 0.16
SNAPSHOT_INTERVAL_S = 0.01
EARLY_LIMIT_S = 0.08


def run_simulation(dt_s: float):
    """Simule le même milieu pour un pas temporel donné."""

    x_m = np.arange(0.0, WIDTH_M + DX_M, DX_M)
    z_m = np.arange(0.0, DEPTH_M + DX_M, DX_M)

    velocity = np.full(
        (z_m.size, x_m.size),
        VELOCITY_M_S,
        dtype=np.float64,
    )

    model = VelocityModel(
        x_m=x_m,
        z_m=z_m,
        velocity_m_s=velocity,
        anomaly_mask=np.zeros_like(velocity, dtype=bool),
    )

    snapshot_stride = round(SNAPSHOT_INTERVAL_S / dt_s)

    return simulate_acoustic_wave(
        model,
        duration_s=DURATION_S,
        dt_s=dt_s,
        source_x_m=SOURCE_X_M,
        source_z_m=SOURCE_Z_M,
        source_frequency_hz=FREQUENCY_HZ,
        sponge_width_m=40.0,
        sponge_strength_s_inv=80.0,
        snapshot_stride=snapshot_stride,
    )


def compare_fields(reference: np.ndarray, candidate: np.ndarray):
    """Compare les amplitudes et les formes des champs à chaque instant."""

    if reference.shape != candidate.shape:
        raise ValueError(
            "Impossible de comparer des champs de dimensions différentes."
        )

    reference_flat = reference.reshape(reference.shape[0], -1)
    candidate_flat = candidate.reshape(candidate.shape[0], -1)

    reference_norm = np.linalg.norm(reference_flat, axis=1)
    candidate_norm = np.linalg.norm(candidate_flat, axis=1)

    difference_norm = np.linalg.norm(
        reference_flat - candidate_flat,
        axis=1,
    )

    scale = max(float(np.max(reference_norm)), 1.0e-30)
    threshold = scale * 1.0e-10

    amplitude_errors = np.full(reference.shape[0], np.nan)
    shape_errors = np.full(reference.shape[0], np.nan)
    norm_ratios = np.full(reference.shape[0], np.nan)

    valid = (
        (reference_norm > threshold)
        & (candidate_norm > threshold)
    )

    amplitude_errors[valid] = (
        difference_norm[valid] / reference_norm[valid]
    )

    reference_unit = (
        reference_flat[valid]
        / reference_norm[valid, np.newaxis]
    )
    candidate_unit = (
        candidate_flat[valid]
        / candidate_norm[valid, np.newaxis]
    )

    shape_errors[valid] = np.linalg.norm(
        reference_unit - candidate_unit,
        axis=1,
    )

    norm_ratios[valid] = (
        candidate_norm[valid] / reference_norm[valid]
    )

    return amplitude_errors, shape_errors, norm_ratios


def summarize(
    times_s: np.ndarray,
    errors: np.ndarray,
    early_limit_s: float,
) -> tuple[float, float, float]:
    """Retourne la moyenne précoce, la moyenne globale et le maximum."""

    early = (
        (times_s <= early_limit_s)
        & np.isfinite(errors)
    )
    valid = np.isfinite(errors)

    if not np.any(early) or not np.any(valid):
        raise ValueError(
            "Impossible de calculer les statistiques de convergence."
        )

    return (
        float(np.mean(errors[early])),
        float(np.mean(errors[valid])),
        float(np.max(errors[valid])),
    )


def main() -> None:
    configurations = [
        ("A", 4.0e-4),
        ("B", 2.0e-4),
        ("D", 1.0e-4),
    ]

    results = {}

    print("=== CONVERGENCE TEMPORELLE ===")
    print(f"Vitesse imposée : {VELOCITY_M_S:.1f} m/s")
    print(f"Pas spatial     : {DX_M:.1f} m")
    print(f"Durée           : {DURATION_S:.3f} s")
    print()

    for label, dt_s in configurations:
        print(f"Simulation {label} : dt = {dt_s * 1000.0:.3f} ms")

        result = run_simulation(dt_s)
        results[label] = result

        print(f"  Nombre de CFL : {result.cfl_number:.4f}")
        print(f"  Dimensions des champs : {result.snapshots.shape}")
        print()

    times = results["A"].times_s

    for label in ("B", "D"):
        if not np.allclose(
            times,
            results[label].times_s,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                "Les champs n'ont pas été enregistrés aux mêmes instants."
            )

    comparisons = [
        ("A contre B", "B", "A"),
        ("B contre D", "D", "B"),
        ("A contre D", "D", "A"),
    ]

    curves = {}

    print("=== RÉSULTATS DES COMPARAISONS ===")

    for description, reference_label, candidate_label in comparisons:
        errors, shapes, ratios = compare_fields(
            results[reference_label].snapshots,
            results[candidate_label].snapshots,
        )

        early_mean, global_mean, global_max = summarize(
            times,
            errors,
            EARLY_LIMIT_S,
        )

        curves[description] = errors

        target_index = int(np.argmin(np.abs(times - EARLY_LIMIT_S)))

        print(description)
        print(
            "  Erreur moyenne avant 0.08 s : "
            f"{early_mean:.6e}"
        )
        print(
            "  Erreur relative moyenne globale : "
            f"{global_mean:.6e}"
        )
        print(
            "  Erreur relative maximale globale : "
            f"{global_max:.6e}"
        )

        if np.isfinite(shapes[target_index]):
            print(
                "  Écart entre formes normalisées à 0.08 s : "
                f"{shapes[target_index]:.6e}"
            )
            print(
                "  Rapport des normes à 0.08 s : "
                f"{ratios[target_index]:.6e}"
            )
        else:
            print(
                "  Comparaison normalisée à 0.08 s indisponible."
            )

        print()

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
        / "temporal_convergence.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for label, errors in curves.items():
        valid = np.isfinite(errors)

        ax.plot(
            times[valid],
            errors[valid],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=label,
        )

    ax.axvline(
        EARLY_LIMIT_S,
        linestyle="--",
        linewidth=1.2,
        label=f"Limite de la fenêtre précoce : {EARLY_LIMIT_S:.2f} s",
    )

    ax.set_title("Influence du pas temporel sur le champ d'onde")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Erreur relative de la norme L2")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure enregistrée : {output_path}")
    print(
        "Interprétation : une diminution cohérente des écarts "
        "est un indice de convergence, pas une preuve suffisante "
        "d'exactitude physique."
    )


if __name__ == "__main__":
    main()

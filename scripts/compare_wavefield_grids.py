"""Compare les champs d'onde pour plusieurs résolutions numériques."""

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
SOURCE_X_M = 60.0
SOURCE_Z_M = 50.0
FREQUENCY_HZ = 25.0
DURATION_S = 0.16
SNAPSHOT_INTERVAL_S = 0.01


def build_homogeneous_model(dx_m: float) -> VelocityModel:
    """Construit un milieu homogène à une résolution donnée."""

    dz_m = dx_m

    x_m = np.arange(0.0, WIDTH_M + 0.5 * dx_m, dx_m)
    z_m = np.arange(0.0, DEPTH_M + 0.5 * dz_m, dz_m)

    velocity = np.full(
        (z_m.size, x_m.size),
        VELOCITY_M_S,
        dtype=np.float64,
    )

    return VelocityModel(
        x_m=x_m,
        z_m=z_m,
        velocity_m_s=velocity,
        anomaly_mask=np.zeros_like(velocity, dtype=bool),
    )


def simulate_grid(
    dx_m: float,
    dt_s: float,
):
    """Exécute une simulation et retourne son résultat."""

    model = build_homogeneous_model(dx_m)

    snapshot_stride = max(
        1,
        round(SNAPSHOT_INTERVAL_S / dt_s),
    )

    result = simulate_acoustic_wave(
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

    return result


def relative_field_errors(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> np.ndarray:
    """Calcule l'erreur relative des champs à chaque instant."""

    if reference.shape != comparison.shape:
        raise ValueError(
            "Les champs doivent avoir des dimensions identiques "
            "après mise aux mêmes coordonnées."
        )

    reference_norms = np.linalg.norm(
        reference.reshape(reference.shape[0], -1),
        axis=1,
    )

    difference_norms = np.linalg.norm(
        (reference - comparison).reshape(reference.shape[0], -1),
        axis=1,
    )

    reference_scale = float(np.max(reference_norms))

    if reference_scale <= 0.0:
        raise ValueError("Le champ de référence est nul.")

    threshold = reference_scale * 1.0e-10

    errors = np.full(reference.shape[0], np.nan)

    valid = reference_norms > threshold

    errors[valid] = (
        difference_norms[valid] / reference_norms[valid]
    )

    return errors


def summarize_errors(
    times_s: np.ndarray,
    errors: np.ndarray,
) -> dict[str, float]:
    """Résume les erreurs calculées en excluant les champs nuls."""

    valid = np.isfinite(errors)

    if not np.any(valid):
        raise ValueError("Aucune erreur relative exploitable.")

    valid_errors = errors[valid]
    valid_times = times_s[valid]

    final_index = np.flatnonzero(valid)[-1]

    return {
        "mean": float(np.mean(valid_errors)),
        "maximum": float(np.max(valid_errors)),
        "final": float(errors[final_index]),
        "final_time": float(times_s[final_index]),
    }


def main() -> None:
    """Compare les solutions numériques et enregistre les résultats."""

    configurations = [
        ("A : dx=2 m, dt=0.4 ms", 2.0, 4.0e-4),
        ("B : dx=2 m, dt=0.2 ms", 2.0, 2.0e-4),
        ("C : dx=1 m, dt=0.2 ms", 1.0, 2.0e-4),
    ]

    results = {}

    print("=== COMPARAISON DES CHAMPS D'ONDE ===")
    print(f"Vitesse imposée : {VELOCITY_M_S:.1f} m/s")
    print(f"Durée           : {DURATION_S:.3f} s")
    print()

    for label, dx_m, dt_s in configurations:
        print(f"Simulation : {label}")

        result = simulate_grid(dx_m, dt_s)
        results[label] = result

        print(f"  Nombre de CFL : {result.cfl_number:.4f}")
        print(f"  Dimensions des champs : {result.snapshots.shape}")
        print(f"  Instant final enregistré : {result.times_s[-1]:.6f} s")
        print()

    label_a, label_b, label_c = [
        label for label, _, _ in configurations
    ]

    result_a = results[label_a]
    result_b = results[label_b]
    result_c = results[label_c]

    # Les simulations ont été enregistrées à intervalles physiques
    # identiques. Vérifions-le avant de comparer les champs.
    if not np.allclose(
        result_a.times_s,
        result_b.times_s,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Les instants enregistrés diffèrent entre A et B."
        )

    if not np.allclose(
        result_a.times_s,
        result_c.times_s,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Les instants enregistrés diffèrent entre A et C."
        )

    # A et B possèdent la même grille spatiale.
    # Pour C, on garde un point sur deux dans chaque direction
    # afin de revenir aux coordonnées de la grille de 2 m.
    snapshots_a = result_a.snapshots
    snapshots_b = result_b.snapshots
    snapshots_c_on_coarse_grid = result_c.snapshots[:, ::2, ::2]

    if snapshots_a.shape != snapshots_c_on_coarse_grid.shape:
        raise RuntimeError(
            "La réduction du champ fin ne donne pas la grille attendue."
        )

    errors_ab = relative_field_errors(
        snapshots_a,
        snapshots_b,
    )

    errors_bc = relative_field_errors(
        snapshots_b,
        snapshots_c_on_coarse_grid,
    )

    errors_ac = relative_field_errors(
        snapshots_a,
        snapshots_c_on_coarse_grid,
    )

    comparisons = [
        ("A contre B : effet du pas temporel", errors_ab),
        ("B contre C : effet du pas spatial", errors_bc),
        ("A contre C : raffinement global", errors_ac),
    ]

    print("=== RÉSULTATS DES COMPARAISONS ===")

    for label, errors in comparisons:
        summary = summarize_errors(result_a.times_s, errors)

        print(label)
        print(f"  Erreur relative moyenne : {summary['mean']:.6e}")
        print(f"  Erreur relative maximale : {summary['maximum']:.6e}")
        print(f"  Erreur au dernier instant utile : {summary['final']:.6e}")
        print(f"  Dernier instant comparé : {summary['final_time']:.3f} s")
        print()


    # Diagnostic complémentaire avant les réflexions importantes.
    early_limit_s = 0.080
    early_mask = result_a.times_s <= early_limit_s
    diagnostic_index = int(
        np.argmin(np.abs(result_a.times_s - early_limit_s))
    )

    diagnostic_pairs = [
        ("A / B : pas temporel", snapshots_a, snapshots_b, errors_ab),
        (
            "B / C : pas spatial",
            snapshots_b,
            snapshots_c_on_coarse_grid,
            errors_bc,
        ),
        (
            "A / C : raffinement global",
            snapshots_a,
            snapshots_c_on_coarse_grid,
            errors_ac,
        ),
    ]

    print("=== DIAGNOSTIC DES CHAMPS À TEMPS PRÉCOCE ===")
    print(f"Fenêtre examinée : t <= {early_limit_s:.3f} s")
    print()

    for label, reference, comparison, raw_errors in diagnostic_pairs:
        valid_early = early_mask & np.isfinite(raw_errors)

        if np.any(valid_early):
            mean_early_error = float(
                np.mean(raw_errors[valid_early])
            )
        else:
            mean_early_error = float("nan")

        reference_flat = reference.reshape(reference.shape[0], -1)
        comparison_flat = comparison.reshape(comparison.shape[0], -1)

        reference_norms = np.linalg.norm(reference_flat, axis=1)
        comparison_norms = np.linalg.norm(comparison_flat, axis=1)

        # On écarte les instants où l'un des champs est pratiquement nul.
        reference_threshold = max(
            float(np.max(reference_norms)) * 1.0e-10,
            1.0e-30,
        )
        comparison_threshold = max(
            float(np.max(comparison_norms)) * 1.0e-10,
            1.0e-30,
        )

        valid_shapes = (
            (reference_norms > reference_threshold)
            & (comparison_norms > comparison_threshold)
        )

        shape_errors = np.full(reference.shape[0], np.nan)

        normalized_reference = (
            reference_flat[valid_shapes]
            / reference_norms[valid_shapes, np.newaxis]
        )
        normalized_comparison = (
            comparison_flat[valid_shapes]
            / comparison_norms[valid_shapes, np.newaxis]
        )

        shape_errors[valid_shapes] = np.linalg.norm(
            normalized_reference - normalized_comparison,
            axis=1,
        )

        index_is_valid = bool(valid_shapes[diagnostic_index])

        print(label)
        print(
            "  Erreur relative moyenne des amplitudes avant 0.08 s : "
            f"{mean_early_error:.6e}"
        )

        if index_is_valid:
            print(
                "  Écart entre formes normalisées à "
                f"{result_a.times_s[diagnostic_index]:.3f} s : "
                f"{shape_errors[diagnostic_index]:.6e}"
            )
            print(
                "  Rapport des normes au même instant "
                "(référence / comparaison) : "
                f"{reference_norms[diagnostic_index] / comparison_norms[diagnostic_index]:.6e}"
            )
        else:
            print(
                "  Forme normalisée à 0.08 s : non calculable "
                "(champ trop faible dans au moins une configuration)."
            )

        print()

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
        / "wavefield_grid_comparison.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for label, errors in comparisons:
        valid = np.isfinite(errors)

        ax.plot(
            result_a.times_s[valid],
            errors[valid],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=label,
        )

    ax.set_title("Écart relatif entre champs d'onde")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Erreur relative de la norme L2")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure enregistrée : {output_path}")
    print(
        "Note : les écarts comparent des solutions numériques. "
        "Ils ne constituent pas une comparaison à une solution exacte."
    )


if __name__ == "__main__":
    main()

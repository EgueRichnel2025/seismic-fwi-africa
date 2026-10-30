"""Analyse temporelle et spatiale du résidu des données synthétiques."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "experiments" / "output" / "synthetic_fwi_dataset.npz"
FIGURE_PATH = ROOT / "results" / "figures" / "residual_timing_analysis.png"
REPORT_PATH = ROOT / "experiments" / "output" / "residual_timing_analysis.json"

WINDOW_S = 0.03
EARLY_LIMIT_S = 0.24


def main() -> None:
    """Calcule les résidus temporels et leur distribution entre récepteurs."""

    if not DATA_PATH.exists():
        raise SystemExit(f"Jeu de données introuvable : {DATA_PATH}")

    with np.load(DATA_PATH) as data:
        times = data["sample_times_s"].copy()
        receiver_x = data["receiver_x_m"].copy()
        observed = data["observed_data"].copy()
        predicted = data["predicted_data"].copy()
        residual = data["residual"].copy()

    if observed.shape != predicted.shape or observed.shape != residual.shape:
        raise ValueError("Les tableaux observés, prédits et résiduels diffèrent.")

    if observed.shape != (times.size, receiver_x.size):
        raise ValueError("Les dimensions des données ne correspondent pas.")

    if not all(
        np.all(np.isfinite(values))
        for values in (times, observed, predicted, residual)
    ):
        raise ValueError("Les données contiennent des valeurs non finies.")

    report = {
        "window_width_s": WINDOW_S,
        "late_data_start_s": EARLY_LIMIT_S,
        "windows": [],
        "receiver_sensitivity_after_024_s": [],
    }

    # Analyse par fenêtres temporelles.
    edges = np.arange(
        0.0,
        times[-1] + WINDOW_S,
        WINDOW_S,
    )

    if edges[-1] < times[-1]:
        edges = np.append(edges, edges[-1] + WINDOW_S)

    print("=== ANALYSE TEMPORELLE DU RÉSIDU ===")
    print(f"Durée des données : {times[-1]:.3f} s")
    print(f"Largeur des fenêtres : {WINDOW_S * 1000:.0f} ms")
    print()
    print("Fenêtre (s) | Norme observée | Norme résidu | Résidu relatif (%)")

    window_centers = []
    relative_errors = []
    residual_norms = []

    for index in range(len(edges) - 1):
        start = edges[index]
        stop = edges[index + 1]

        if index == len(edges) - 2:
            mask = (times >= start) & (times <= stop)
        else:
            mask = (times >= start) & (times < stop)

        if not np.any(mask):
            continue

        observed_part = observed[mask]
        residual_part = residual[mask]

        observed_norm = float(np.linalg.norm(observed_part))
        residual_norm = float(np.linalg.norm(residual_part))

        relative = (
            residual_norm / observed_norm
            if observed_norm > 0.0
            else None
        )

        relative_percent = (
            100.0 * relative
            if relative is not None
            else None
        )

        center = 0.5 * (start + min(stop, times[-1]))
        window_centers.append(center)
        residual_norms.append(residual_norm)
        relative_errors.append(
            relative_percent if relative_percent is not None else np.nan
        )

        report["windows"].append(
            {
                "start_s": float(start),
                "end_s": float(min(stop, times[-1])),
                "observed_norm": observed_norm,
                "residual_norm": residual_norm,
                "relative_residual": relative,
                "relative_residual_percent": relative_percent,
            }
        )

        rel_text = (
            f"{relative_percent:.6e}"
            if relative_percent is not None
            else "indéfini"
        )

        print(
            f"{start:5.2f}–{min(stop, times[-1]):5.2f} "
            f"| {observed_norm:14.6e} "
            f"| {residual_norm:12.6e} "
            f"| {rel_text}"
        )

    # Sensibilité de chaque récepteur pendant la période tardive.
    late_mask = times > EARLY_LIMIT_S

    if not np.any(late_mask):
        raise ValueError("Aucune donnée disponible après 0,24 s.")

    late_observed = observed[late_mask]
    late_residual = residual[late_mask]

    receiver_observed_norms = np.linalg.norm(late_observed, axis=0)
    receiver_residual_norms = np.linalg.norm(late_residual, axis=0)

    receiver_relative = np.divide(
        receiver_residual_norms,
        receiver_observed_norms,
        out=np.full(receiver_x.size, np.nan),
        where=receiver_observed_norms > 1.0e-15,
    )

    for index, x_position in enumerate(receiver_x):
        report["receiver_sensitivity_after_024_s"].append(
            {
                "receiver_x_m": float(x_position),
                "observed_norm": float(receiver_observed_norms[index]),
                "residual_norm": float(receiver_residual_norms[index]),
                "relative_residual": (
                    float(receiver_relative[index])
                    if np.isfinite(receiver_relative[index])
                    else None
                ),
            }
        )

    valid = np.flatnonzero(np.isfinite(receiver_relative))
    ranking = valid[
        np.argsort(receiver_relative[valid])[::-1]
    ]

    print()
    print("=== CINQ RÉCEPTEURS LES PLUS SENSIBLES APRÈS 0,24 s ===")
    print("Position (m) | Norme du résidu | Résidu relatif (%)")

    for index in ranking[:5]:
        print(
            f"{receiver_x[index]:12.1f} | "
            f"{receiver_residual_norms[index]:15.6e} | "
            f"{100.0 * receiver_relative[index]:.6e}"
        )

    # Graphique : résidu relatif par fenêtre et norme du résidu.
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        constrained_layout=True,
    )

    centers = np.asarray(window_centers)
    errors = np.asarray(relative_errors)
    norms = np.asarray(residual_norms)

    axes[0].plot(
        centers,
        errors,
        marker="o",
        linewidth=1.8,
    )
    axes[0].axvline(
        EARLY_LIMIT_S,
        linestyle="--",
        linewidth=1.4,
        label="Début de la fenêtre tardive",
    )
    axes[0].set_title("Résidu relatif par fenêtre temporelle")
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Résidu relatif (%)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(
        centers,
        norms,
        marker="o",
        linewidth=1.8,
    )
    axes[1].axvline(
        EARLY_LIMIT_S,
        linestyle="--",
        linewidth=1.4,
        label="0,24 s",
    )
    axes[1].set_title("Norme du résidu par fenêtre temporelle")
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_ylabel("Norme du résidu")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Figure enregistrée : {FIGURE_PATH}")
    print(f"Rapport enregistré : {REPORT_PATH}")


if __name__ == "__main__":
    main()

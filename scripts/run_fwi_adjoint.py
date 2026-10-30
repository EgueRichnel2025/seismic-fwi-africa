"""Inversion de forme d'onde complète par état adjoint.

Ce script exécute la chaîne scientifique de bout en bout :

    modele de reference -> donnees synthetiques observees
    -> modele initial -> gradient adjoint -> L-BFGS preconditionne
    -> modele reconstruit -> metriques et figures

Le modèle est synthétique. L'anomalie de vitesse est un objet de test
défini par le projet : elle ne constitue en aucun cas l'observation d'un
aquifère ou d'une structure géologique réelle.

Usage :
    python scripts/run_fwi_adjoint.py
    python scripts/run_fwi_adjoint.py --rapide
"""

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from seismic.hpc.backend import diagnose_backend
from seismic.model.velocity_model import build_synthetic_velocity_model
from seismic.inversion.fwi import build_acquisition, forward_data
from seismic.inversion.optimize import lbfgs_inversion
from seismic.visualization.style import apply_project_style

FIGURE_DIRECTORY = Path("results/figures")
OUTPUT_DIRECTORY = Path("experiments/output")

SOURCE_POSITIONS_M = (60.0, 180.0, 300.0, 420.0, 540.0)

# Conventions graphiques du projet : niveaux de gris, police Tahoma.
apply_project_style()


def _configure() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Inversion FWI par etat adjoint sur donnees synthetiques."
    )
    parser.add_argument(
        "--rapide",
        action="store_true",
        help=(
            "Execute une version grossiere (pas de 10 m) pour verifier la "
            "chaine en quelques secondes."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Nombre maximal d'iterations L-BFGS.",
    )

    return parser.parse_args()


def _plot_velocity_models(
    x_m,
    z_m,
    true_velocity,
    initial_velocity,
    final_velocity,
    path: Path,
) -> None:
    """Trace les trois modèles de vitesse sur une échelle commune."""

    minimum = float(
        min(true_velocity.min(), initial_velocity.min(), final_velocity.min())
    )
    maximum = float(
        max(true_velocity.max(), initial_velocity.max(), final_velocity.max())
    )

    extent = [x_m[0], x_m[-1], z_m[-1], z_m[0]]

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 10.0), sharex=True)

    panels = [
        (true_velocity, "Modele de reference (synthetique)"),
        (initial_velocity, "Modele initial (sans anomalie)"),
        (final_velocity, "Modele reconstruit par FWI adjointe"),
    ]

    for axis, (field, title) in zip(axes, panels):
        image = axis.imshow(
            field,
            cmap="gray_r",
            extent=extent,
            aspect="auto",
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_title(title)
        axis.set_ylabel("Profondeur z (m)")
        figure.colorbar(image, ax=axis, label="Vitesse (m/s)")

    axes[-1].set_xlabel("Distance x (m)")

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_model_difference(
    x_m, z_m, true_velocity, initial_velocity, final_velocity, path: Path
) -> None:
    """Trace l'erreur de modèle avant et après inversion."""

    before = np.abs(initial_velocity - true_velocity)
    after = np.abs(final_velocity - true_velocity)

    maximum = float(max(before.max(), after.max()))
    extent = [x_m[0], x_m[-1], z_m[-1], z_m[0]]

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    for axis, (field, title) in zip(
        axes,
        [
            (before, "Erreur absolue avant inversion"),
            (after, "Erreur absolue apres inversion"),
        ],
    ):
        image = axis.imshow(
            field,
            cmap="gray_r",
            extent=extent,
            aspect="auto",
            vmin=0.0,
            vmax=maximum,
        )
        axis.set_title(title)
        axis.set_ylabel("Profondeur z (m)")
        figure.colorbar(image, ax=axis, label="|v - v_vrai| (m/s)")

    axes[-1].set_xlabel("Distance x (m)")

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_convergence(history, path: Path) -> None:
    """Trace l'évolution de l'objectif et des erreurs."""

    iterations = np.arange(len(history.objectives))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axes[0].semilogy(
        iterations, history.objectives, color="black", marker="o",
        markersize=3.0, linewidth=1.2,
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\Phi(m)$")
    axes[0].set_title("Fonction objectif")
    axes[0].grid(True, which="both", linestyle=":", color="0.7")

    axes[1].plot(
        iterations, history.data_errors, color="black",
        linestyle="-", marker="o", markersize=3.0, linewidth=1.2,
        label=r"$E_d$ (donnees)",
    )

    if np.any(np.isfinite(history.target_errors)):
        axes[1].plot(
            iterations, history.target_errors, color="0.45",
            linestyle="--", marker="s", markersize=3.0, linewidth=1.2,
            label=r"$E_{cible}$ (zone anomalie)",
        )

    if np.any(np.isfinite(history.model_errors)):
        axes[1].plot(
            iterations, history.model_errors, color="0.7",
            linestyle="-.", marker="^", markersize=3.0, linewidth=1.2,
            label=r"$E_m$ (grille entiere)",
        )

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Erreur relative")
    axes[1].set_title("Erreurs")
    axes[1].legend(frameon=False, fontsize=10)
    axes[1].grid(True, linestyle=":", color="0.7")

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_seismograms(
    observed, initial_predicted, final_predicted, times_s, path: Path
) -> None:
    """Compare les sismogrammes observé, initial et final."""

    amplitude = float(np.max(np.abs(observed)))
    extent = [0, observed.shape[1], times_s[-1], times_s[0]]

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=True)

    panels = [
        (observed, "Observe (modele de reference)"),
        (observed - initial_predicted, "Residu initial"),
        (observed - final_predicted, "Residu final"),
    ]

    for axis, (field, title) in zip(axes, panels):
        axis.imshow(
            field,
            cmap="gray",
            extent=extent,
            aspect="auto",
            vmin=-amplitude,
            vmax=amplitude,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Recepteur")

    axes[0].set_ylabel("Temps (s)")

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Exécute l'inversion FWI et produit les livrables."""

    arguments = _configure()

    spacing_m = 10.0 if arguments.rapide else 2.0
    time_step_s = 2.0e-3 if arguments.rapide else 4.0e-4
    instrument_depth_m = 10.0 if arguments.rapide else 4.0

    print("=== INVERSION DE FORME D'ONDE PAR ETAT ADJOINT ===")
    print()
    print(f"Mode          : {'rapide' if arguments.rapide else 'complet'}")
    print(f"Pas de grille : {spacing_m} m")
    print(f"Pas temporel  : {time_step_s} s")
    print(f"Tirs          : {len(SOURCE_POSITIONS_M)}")

    backend = diagnose_backend()
    print(f"Backend BLAS  : {backend.provider}")
    print(f"Intel oneMKL  : {'OUI' if backend.uses_onemkl else 'NON'}")
    print()

    model = build_synthetic_velocity_model(dx_m=spacing_m, dz_m=spacing_m)

    acquisitions = [
        build_acquisition(
            model.x_m,
            model.z_m,
            dt_s=time_step_s,
            source_x_m=source_x_m,
            source_z_m=instrument_depth_m,
            receiver_z_m=instrument_depth_m,
        )
        for source_x_m in SOURCE_POSITIONS_M
    ]

    print(f"Grille        : {model.velocity_m_s.shape} "
          f"({model.velocity_m_s.size} cellules)")
    print(f"Recepteurs    : {acquisitions[0].receiver_ix.size} par tir")
    print(f"Echantillons  : {acquisitions[0].number_of_steps + 1}")
    print(f"CFL           : "
          f"{forward_data(model.velocity_m_s**2, acquisitions[0]).cfl_number:.4f}")
    print()

    true_squared = model.velocity_m_s**2

    print("Generation des donnees synthetiques observees...")
    observed = [
        forward_data(true_squared, acquisition).receiver_signals
        for acquisition in acquisitions
    ]

    initial_velocity = model.velocity_m_s.copy()
    initial_velocity[model.anomaly_mask] = 2000.0
    initial_squared = initial_velocity**2

    initial_predicted = [
        forward_data(initial_squared, acquisition).receiver_signals
        for acquisition in acquisitions
    ]

    print("Inversion L-BFGS preconditionnee...")
    print()

    start_time = time.perf_counter()

    final_squared, history = lbfgs_inversion(
        initial_squared,
        observed,
        acquisitions,
        max_iterations=arguments.iterations,
        true_squared_velocity=true_squared,
        target_mask=model.anomaly_mask,
        gradient_smoothing_points=2.0,
    )

    elapsed_s = time.perf_counter() - start_time

    final_velocity = np.sqrt(final_squared)

    final_predicted = [
        forward_data(final_squared, acquisition).receiver_signals
        for acquisition in acquisitions
    ]

    anomaly_true = float(model.velocity_m_s[model.anomaly_mask].mean())
    anomaly_initial = float(initial_velocity[model.anomaly_mask].mean())
    anomaly_final = float(final_velocity[model.anomaly_mask].mean())

    print()
    print("=== RESULTATS ===")
    print(f"Duree               : {elapsed_s:.1f} s")
    print(f"Simulations directes: {history.simulation_count}")
    print(
        f"Phi                 : {history.objectives[0]:.6e} -> "
        f"{history.objectives[-1]:.6e} "
        f"(facteur {history.objectives[0] / history.objectives[-1]:.1f})"
    )
    print(
        f"E_d  (donnees)      : {history.data_errors[0]:.4f} -> "
        f"{history.data_errors[-1]:.4f}"
    )
    print(
        f"E_cible (anomalie)  : {history.target_errors[0]:.4f} -> "
        f"{history.target_errors[-1]:.4f}"
    )
    print(
        f"E_m (grille)        : {history.model_errors[0]:.4f} -> "
        f"{history.model_errors[-1]:.4f}"
    )
    print()
    print("Vitesse moyenne dans la zone de l'anomalie :")
    print(f"  reference : {anomaly_true:.1f} m/s")
    print(f"  initial   : {anomaly_initial:.1f} m/s")
    print(f"  reconstruit : {anomaly_final:.1f} m/s")

    recovery = (
        (anomaly_initial - anomaly_final)
        / (anomaly_initial - anomaly_true)
        if anomaly_initial != anomaly_true
        else float("nan")
    )
    print(f"  fraction du contraste recuperee : {recovery:.1%}")

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    suffix = "_rapide" if arguments.rapide else ""

    _plot_velocity_models(
        model.x_m, model.z_m, model.velocity_m_s, initial_velocity,
        final_velocity,
        FIGURE_DIRECTORY / f"fwi_velocity_models{suffix}.png",
    )
    _plot_model_difference(
        model.x_m, model.z_m, model.velocity_m_s, initial_velocity,
        final_velocity,
        FIGURE_DIRECTORY / f"fwi_model_error{suffix}.png",
    )
    _plot_convergence(
        history, FIGURE_DIRECTORY / f"fwi_convergence{suffix}.png"
    )
    _plot_seismograms(
        observed[0], initial_predicted[0], final_predicted[0],
        acquisitions[0].sample_times_s,
        FIGURE_DIRECTORY / f"fwi_seismograms{suffix}.png",
    )

    metrics = {
        "experiment": "synthetic_adjoint_state_fwi",
        "warning": (
            "Modele synthetique. L'anomalie est un objet de test et ne "
            "constitue pas l'observation d'un aquifere reel."
        ),
        "method": "FWI par etat adjoint, L-BFGS preconditionne",
        "configuration": {
            "grid_spacing_m": spacing_m,
            "grid_shape": list(model.velocity_m_s.shape),
            "time_step_s": time_step_s,
            "number_of_samples": acquisitions[0].number_of_steps + 1,
            "source_positions_m": list(SOURCE_POSITIONS_M),
            "receivers_per_shot": int(acquisitions[0].receiver_ix.size),
            "max_iterations": arguments.iterations,
        },
        "backend": backend.to_dict(),
        "runtime_seconds": elapsed_s,
        "forward_simulations": history.simulation_count,
        "history": {
            "objectives": history.objectives,
            "gradient_norms": history.gradient_norms,
            "step_lengths": history.step_lengths,
            "data_errors": history.data_errors,
            "model_errors": history.model_errors,
            "target_errors": history.target_errors,
            "rejected_steps": history.rejected_steps,
        },
        "anomaly_mean_velocity_m_s": {
            "reference": anomaly_true,
            "initial": anomaly_initial,
            "reconstructed": anomaly_final,
            "recovered_fraction": recovery,
        },
    }

    metrics_path = OUTPUT_DIRECTORY / f"fwi_adjoint_metrics{suffix}.json"

    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    model_path = OUTPUT_DIRECTORY / f"fwi_adjoint_model{suffix}.npz"
    np.savez_compressed(
        model_path,
        x_m=model.x_m,
        z_m=model.z_m,
        true_velocity_m_s=model.velocity_m_s,
        initial_velocity_m_s=initial_velocity,
        final_velocity_m_s=final_velocity,
        anomaly_mask=model.anomaly_mask,
    )

    print()
    print(f"Metriques : {metrics_path}")
    print(f"Modeles   : {model_path}")
    print(f"Figures   : {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()

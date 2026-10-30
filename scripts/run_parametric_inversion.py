"""Inversion paramétrique réduite par Gauss-Newton et LAPACK.

Cette inversion est complémentaire de la FWI par état adjoint, et doit
être nommée distinctement. Elle n'inverse pas les 45 451 cellules de la
grille mais les quelques dizaines de coefficients d'une base de fonctions
gaussiennes couvrant une zone d'intérêt.

Cette réduction a une conséquence directe sur l'algèbre linéaire : la
Jacobienne redevient constructible et dense, donc les équations normales
régularisées existent réellement et sont résolues par une factorisation
de Cholesky (LAPACK DPOSV), avec repli sur une résolution par SVD
(LAPACK DGELSD) si le conditionnement l'exige.

Le modèle est synthétique. Les résultats ne constituent pas l'observation
d'une structure géologique réelle.

Usage :
    python scripts/run_parametric_inversion.py
    python scripts/run_parametric_inversion.py --rapide
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
from seismic.inversion.parametric import (
    build_gaussian_basis,
    gauss_newton_inversion,
)
from seismic.visualization.style import apply_project_style

FIGURE_DIRECTORY = Path("results/figures")
OUTPUT_DIRECTORY = Path("experiments/output")

SOURCE_POSITIONS_M = (180.0, 360.0, 480.0)

# Zone d'interet : elle encadre l'anomalie sans en supposer la forme.
TARGET_X_RANGE_M = (240.0, 480.0)
TARGET_Z_RANGE_M = (80.0, 180.0)
BASIS_COUNT_X = 6
BASIS_COUNT_Z = 4

apply_project_style()


def _configure() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Inversion parametrique reduite par Gauss-Newton et LAPACK."
        )
    )
    parser.add_argument(
        "--rapide",
        action="store_true",
        help="Execute une version grossiere pour verifier la chaine.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="Nombre d'iterations de Gauss-Newton.",
    )

    return parser.parse_args()


def _plot_results(
    x_m, z_m, true_velocity, initial_velocity, final_velocity,
    history, path: Path,
) -> None:
    """Trace les modèles et la convergence de l'inversion réduite."""

    extent = [x_m[0], x_m[-1], z_m[-1], z_m[0]]
    minimum = float(min(true_velocity.min(), final_velocity.min()))
    maximum = float(max(true_velocity.max(), final_velocity.max()))

    figure = plt.figure(figsize=(11.0, 8.0))

    for index, (field, title) in enumerate(
        [
            (true_velocity, "Reference (synthetique)"),
            (initial_velocity, "Initial"),
            (final_velocity, "Reconstruit (base reduite)"),
        ]
    ):
        axis = figure.add_subplot(2, 2, index + 1)
        image = axis.imshow(
            field, cmap="gray_r", extent=extent, aspect="auto",
            vmin=minimum, vmax=maximum,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
        figure.colorbar(image, ax=axis, label="v (m/s)")

    axis = figure.add_subplot(2, 2, 4)
    axis.semilogy(
        np.arange(len(history.objectives)), history.objectives,
        color="black", marker="o", markersize=4.0, linewidth=1.2,
    )
    axis.set_xlabel("Iteration Gauss-Newton")
    axis.set_ylabel(r"$\Phi(m)$")
    axis.set_title("Fonction objectif", fontsize=11)
    axis.grid(True, which="both", linestyle=":", color="0.7")

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Exécute l'inversion paramétrique réduite."""

    arguments = _configure()

    spacing_m = 10.0 if arguments.rapide else 4.0
    time_step_s = 2.0e-3 if arguments.rapide else 8.0e-4
    instrument_depth_m = 10.0 if arguments.rapide else 4.0

    print("=== INVERSION PARAMETRIQUE REDUITE (GAUSS-NEWTON + LAPACK) ===")
    print()

    backend = diagnose_backend()

    model = build_synthetic_velocity_model(dx_m=spacing_m, dz_m=spacing_m)

    acquisitions = [
        build_acquisition(
            model.x_m, model.z_m, dt_s=time_step_s,
            source_x_m=source_x_m, source_z_m=instrument_depth_m,
            receiver_z_m=instrument_depth_m,
        )
        for source_x_m in SOURCE_POSITIONS_M
    ]

    basis = build_gaussian_basis(
        model.x_m, model.z_m,
        x_range_m=TARGET_X_RANGE_M, z_range_m=TARGET_Z_RANGE_M,
        number_x=BASIS_COUNT_X, number_z=BASIS_COUNT_Z,
    )

    number_of_data = sum(
        (acquisition.number_of_steps + 1) * acquisition.receiver_ix.size
        for acquisition in acquisitions
    )

    print(f"Backend BLAS        : {backend.provider}")
    print(f"Intel oneMKL        : {'OUI' if backend.uses_onemkl else 'NON'}")
    print(f"Grille              : {model.velocity_m_s.shape} "
          f"({model.velocity_m_s.size} cellules)")
    print(f"Parametres inverses : {basis.shape[0]} coefficients")
    print(f"Donnees             : {number_of_data} observations")
    print(f"Jacobienne reduite  : ({number_of_data}, {basis.shape[0]})")
    full_jacobian_bytes = (
        number_of_data * model.velocity_m_s.size * 8
    )
    print(
        f"  soit {number_of_data * basis.shape[0] * 8 / 1e6:.1f} Mo, "
        "contre "
        f"{full_jacobian_bytes / 1e9:.1f} Go "
        "pour une Jacobienne cellule par cellule"
    )
    print(
        f"Cout de construction: {basis.shape[0]} x {len(acquisitions)} = "
        f"{basis.shape[0] * len(acquisitions)} simulations par iteration"
    )
    print()

    true_squared = model.velocity_m_s**2
    observed = [
        forward_data(true_squared, acquisition).receiver_signals
        for acquisition in acquisitions
    ]

    initial_velocity = model.velocity_m_s.copy()
    initial_velocity[model.anomaly_mask] = 2000.0
    initial_squared = initial_velocity**2

    start_time = time.perf_counter()

    coefficients, final_squared, history = gauss_newton_inversion(
        initial_squared, basis, observed, acquisitions,
        max_iterations=arguments.iterations,
    )

    elapsed_s = time.perf_counter() - start_time

    final_velocity = np.sqrt(np.clip(final_squared, 1.0, None))

    anomaly_true = float(model.velocity_m_s[model.anomaly_mask].mean())
    anomaly_initial = float(initial_velocity[model.anomaly_mask].mean())
    anomaly_final = float(final_velocity[model.anomaly_mask].mean())

    print()
    print("=== RESULTATS ===")
    print(f"Duree          : {elapsed_s:.1f} s")
    print(
        f"Phi            : {history.objectives[0]:.6e} -> "
        f"{history.objectives[-1]:.6e} "
        f"(facteur {history.objectives[0] / history.objectives[-1]:.1f})"
    )
    print(f"Routines LAPACK: {', '.join(sorted(set(history.solver_used)))}")
    print(
        f"Conditionnement: "
        f"{min(history.condition_numbers):.3e} a "
        f"{max(history.condition_numbers):.3e}"
    )
    print()
    print("Vitesse moyenne dans la zone de l'anomalie :")
    print(f"  reference   : {anomaly_true:.1f} m/s")
    print(f"  initial     : {anomaly_initial:.1f} m/s")
    print(f"  reconstruit : {anomaly_final:.1f} m/s")

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    suffix = "_rapide" if arguments.rapide else ""

    _plot_results(
        model.x_m, model.z_m, model.velocity_m_s, initial_velocity,
        final_velocity, history,
        FIGURE_DIRECTORY / f"parametric_inversion{suffix}.png",
    )

    metrics = {
        "experiment": "synthetic_reduced_basis_gauss_newton",
        "warning": (
            "Modele synthetique. Estimation de coefficients sur une base "
            "reduite : ce n'est pas une FWI sur toute la grille."
        ),
        "method": "Gauss-Newton, base gaussienne reduite, LAPACK",
        "configuration": {
            "grid_spacing_m": spacing_m,
            "grid_shape": list(model.velocity_m_s.shape),
            "time_step_s": time_step_s,
            "source_positions_m": list(SOURCE_POSITIONS_M),
            "basis_functions": int(basis.shape[0]),
            "target_x_range_m": list(TARGET_X_RANGE_M),
            "target_z_range_m": list(TARGET_Z_RANGE_M),
            "number_of_data": number_of_data,
        },
        "backend": backend.to_dict(),
        "runtime_seconds": elapsed_s,
        "history": {
            "objectives": history.objectives,
            "step_lengths": history.step_lengths,
            "condition_numbers": history.condition_numbers,
            "solver_used": history.solver_used,
            "rank_estimates": history.rank_estimates,
        },
        "coefficients": coefficients.tolist(),
        "anomaly_mean_velocity_m_s": {
            "reference": anomaly_true,
            "initial": anomaly_initial,
            "reconstructed": anomaly_final,
        },
    }

    metrics_path = OUTPUT_DIRECTORY / f"parametric_metrics{suffix}.json"

    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    print()
    print(f"Metriques : {metrics_path}")
    print(f"Figure    : {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()

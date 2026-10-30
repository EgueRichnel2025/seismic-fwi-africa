"""Mesure le temps des noyaux BLAS/LAPACK aux tailles réelles du projet.

Le protocole est explicite : les tailles mesurées sont celles que
l'inversion manipule effectivement, chaque mesure est précédée d'un
échauffement et répétée, et c'est la médiane qui est rapportée. Aucun gain
n'est annoncé : le script mesure ce qui est, sur le backend qui est.

Tailles mesurées :
- vecteurs de ``Nz * Nx`` composantes, soit la taille du gradient FWI,
  sur lesquels portent DDOT, DAXPY et DNRM2 dans la récursion L-BFGS ;
- Jacobienne réduite ``(nd, nk)`` de l'inversion paramétrique, sur
  laquelle portent DGEMV, DGEMM, DPOSV et DGELSD.

Usage :
    python scripts/benchmark_linear_algebra.py
"""

import json
from pathlib import Path
from statistics import median
import time

import numpy as np

from seismic.hpc.backend import diagnose_backend
from seismic.linear_algebra.kernels import (
    axpy,
    dot,
    gemm,
    gemv,
    nrm2,
    solve_least_squares,
    solve_symmetric_positive_definite,
)

OUTPUT_DIRECTORY = Path("experiments/output")
OUTPUT_PATH = OUTPUT_DIRECTORY / "linear_algebra_benchmark.json"

# Taille du gradient FWI sur la grille de reference : 151 x 301.
GRADIENT_SIZE = 151 * 301

# Jacobienne reduite : 2 tirs x 1051 echantillons x 26 recepteurs.
JACOBIAN_ROWS = 2 * 1051 * 26
JACOBIAN_COLUMNS = 24

WARMUP_RUNS = 2
MEASURED_RUNS = 7


def _timed(function, *arguments) -> tuple[float, object]:
    """Mesure la durée médiane d'un appel, après échauffement."""

    for _ in range(WARMUP_RUNS):
        outcome = function(*arguments)

    durations = []

    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        outcome = function(*arguments)
        durations.append(time.perf_counter() - start)

    return median(durations), outcome


def main() -> None:
    """Exécute le banc de mesure et enregistre les résultats."""

    report = diagnose_backend()

    print("=== MESURES DES NOYAUX D'ALGEBRE LINEAIRE ===")
    print()
    print(f"Backend     : {report.provider}")
    print(f"oneMKL      : {'OUI' if report.uses_onemkl else 'NON'}")
    print(f"Plateforme  : {report.environment['platform']}")
    print(f"Repetitions : {MEASURED_RUNS} mesurees, {WARMUP_RUNS} d'echauffement")
    print()

    generator = np.random.default_rng(2026)

    x = generator.standard_normal(GRADIENT_SIZE)
    y = generator.standard_normal(GRADIENT_SIZE)
    jacobian = generator.standard_normal((JACOBIAN_ROWS, JACOBIAN_COLUMNS))
    residual = generator.standard_normal(JACOBIAN_ROWS)

    measurements: dict[str, dict] = {}

    cases = [
        ("DDOT", f"produit scalaire, n={GRADIENT_SIZE}", dot, (x, y)),
        ("DAXPY", f"mise a jour, n={GRADIENT_SIZE}", axpy, (x, y.copy(), 1.5)),
        ("DNRM2", f"norme, n={GRADIENT_SIZE}", nrm2, (x,)),
        (
            "DGEMV",
            f"J'.r, ({JACOBIAN_ROWS}, {JACOBIAN_COLUMNS})",
            lambda a, b: gemv(a, b, transpose=True),
            (jacobian, residual),
        ),
        (
            "DGEMM",
            f"J'.J, ({JACOBIAN_ROWS}, {JACOBIAN_COLUMNS})",
            lambda a: gemm(a, a, transpose_left=True),
            (jacobian,),
        ),
    ]

    print(f"{'routine':>8} | {'temps median (s)':>18} | description")
    print("-" * 70)

    for name, description, function, arguments in cases:
        duration, _ = _timed(function, *arguments)
        measurements[name] = {
            "description": description,
            "median_seconds": duration,
        }
        print(f"{name:>8} | {duration:18.6e} | {description}")

    normal_matrix = gemm(jacobian, jacobian, transpose_left=True)
    normal_matrix += 1.0e-3 * np.trace(normal_matrix) * np.eye(
        JACOBIAN_COLUMNS
    ) / JACOBIAN_COLUMNS
    right_hand_side = gemv(jacobian, residual, transpose=True)

    duration, posv_solution = _timed(
        solve_symmetric_positive_definite, normal_matrix, right_hand_side
    )
    measurements["DPOSV"] = {
        "description": f"Cholesky, n={JACOBIAN_COLUMNS}",
        "median_seconds": duration,
    }
    print(
        f"{'DPOSV':>8} | {duration:18.6e} | "
        f"Cholesky, n={JACOBIAN_COLUMNS}"
    )

    duration, gelsd_outcome = _timed(
        solve_least_squares, jacobian, residual
    )
    measurements["DGELSD"] = {
        "description": (
            f"moindres carres SVD, ({JACOBIAN_ROWS}, {JACOBIAN_COLUMNS})"
        ),
        "median_seconds": duration,
    }
    print(
        f"{'DGELSD':>8} | {duration:18.6e} | "
        f"moindres carres SVD, ({JACOBIAN_ROWS}, {JACOBIAN_COLUMNS})"
    )

    # Validation croisee : les deux routines doivent s'accorder avec NumPy.
    reference = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
    gelsd_error = float(
        np.max(np.abs(gelsd_outcome[0] - reference))
        / max(float(np.max(np.abs(reference))), 1.0e-300)
    )

    normal_reference = np.linalg.solve(normal_matrix, right_hand_side)
    posv_error = float(
        np.max(np.abs(posv_solution - normal_reference))
        / max(float(np.max(np.abs(normal_reference))), 1.0e-300)
    )

    print()
    print("--- Validation croisee contre NumPy ---")
    print(f"  DPOSV  : ecart relatif {posv_error:.3e}")
    print(f"  DGELSD : ecart relatif {gelsd_error:.3e}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    payload = {
        "backend": report.to_dict(),
        "protocol": {
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "statistic": "median",
        },
        "sizes": {
            "gradient_vector": GRADIENT_SIZE,
            "jacobian_rows": JACOBIAN_ROWS,
            "jacobian_columns": JACOBIAN_COLUMNS,
        },
        "measurements": measurements,
        "cross_validation": {
            "dposv_relative_error": posv_error,
            "dgelsd_relative_error": gelsd_error,
        },
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)

    print()
    print(f"Mesures enregistrees : {OUTPUT_PATH}")
    print()
    print(
        "Ces temps caracterisent le backend " f"{report.provider} "
        "sur cette machine. Ils ne constituent pas une comparaison entre "
        "backends : pour cela, relancer ce script dans un environnement "
        "lie a un autre fournisseur et comparer les fichiers produits."
    )


if __name__ == "__main__":
    main()

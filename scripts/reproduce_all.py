"""Rejoue l'ensemble de la chaîne scientifique en une commande.

Ordre d'exécution :

1. diagnostic du backend BLAS/LAPACK ;
2. mesure des noyaux d'algèbre linéaire ;
3. modèle de vitesse synthétique et ondelette de Ricker ;
4. propagation directe et validations numériques ;
5. données synthétiques observées ;
6. inversion de forme d'onde par état adjoint ;
7. inversion paramétrique réduite.

Usage :
    python scripts/reproduce_all.py
    python scripts/reproduce_all.py --rapide
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _configure() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Rejoue toutes les experiences du projet."
    )
    parser.add_argument(
        "--rapide",
        action="store_true",
        help=(
            "Execute les inversions sur une grille grossiere. "
            "Utile pour verifier la chaine en moins d'une minute."
        ),
    )
    parser.add_argument(
        "--sans-inversion",
        action="store_true",
        help="N'execute que le modele direct et les validations.",
    )

    return parser.parse_args()


def _run(script_name: str, extra_arguments: list[str]) -> bool:
    """Exécute un script du projet et rapporte son issue.

    Args:
        script_name: Nom du script dans ``scripts/``.
        extra_arguments: Arguments transmis au script.

    Returns:
        Vrai si le script s'est terminé sans erreur.
    """

    script_path = REPOSITORY_ROOT / "scripts" / script_name

    if not script_path.exists():
        print(f"  IGNORE : {script_name} est absent du depot.")
        return True

    command = [sys.executable, str(script_path), *extra_arguments]

    print()
    print("=" * 72)
    print(f"EXECUTION : {script_name} {' '.join(extra_arguments)}".strip())
    print("=" * 72)

    start_time = time.perf_counter()
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    elapsed_s = time.perf_counter() - start_time

    if completed.returncode == 0:
        print(f"-> termine en {elapsed_s:.1f} s")
        return True

    print(f"-> ECHEC (code {completed.returncode}) apres {elapsed_s:.1f} s")
    return False


def main() -> None:
    """Rejoue la chaîne complète et affiche une synthèse."""

    arguments = _configure()
    fast = ["--rapide"] if arguments.rapide else []

    steps: list[tuple[str, list[str]]] = [
        ("diagnose_blas_backend.py", []),
        ("benchmark_linear_algebra.py", []),
        ("plot_velocity_model.py", []),
        ("plot_ricker_wavelet.py", []),
        ("simulate_wave_propagation.py", []),
        ("validate_direct_arrival.py", []),
        ("validate_temporal_convergence.py", []),
        ("create_synthetic_observations.py", []),
        ("analyze_residual_timing.py", []),
    ]

    if not arguments.sans_inversion:
        steps.extend(
            [
                ("run_fwi_adjoint.py", fast),
                ("run_parametric_inversion.py", fast),
            ]
        )

    print("=== REPRODUCTION COMPLETE DE LA CHAINE SCIENTIFIQUE ===")
    print(f"Mode : {'rapide' if arguments.rapide else 'complet'}")
    print(f"Etapes : {len(steps)}")

    overall_start = time.perf_counter()
    failures: list[str] = []

    for script_name, extra in steps:
        if not _run(script_name, extra):
            failures.append(script_name)

    total_elapsed_s = time.perf_counter() - overall_start

    print()
    print("=" * 72)
    print("SYNTHESE")
    print("=" * 72)
    print(f"Duree totale : {total_elapsed_s:.1f} s")
    print(f"Etapes reussies : {len(steps) - len(failures)} / {len(steps)}")

    if failures:
        print()
        print("Etapes en echec :")

        for name in failures:
            print(f"  - {name}")

        sys.exit(1)

    print()
    print("Toutes les etapes ont abouti.")
    print("Figures  : results/figures/")
    print("Donnees  : experiments/output/")


if __name__ == "__main__":
    main()

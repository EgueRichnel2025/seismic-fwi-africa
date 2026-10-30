"""Diagnostique le fournisseur BLAS/LAPACK réellement chargé.

Ce script répond à une seule question, et il y répond honnêtement :
Intel oneMKL est-il utilisé par cet environnement, oui ou non ?

Usage :
    python scripts/diagnose_blas_backend.py
"""

import json
from pathlib import Path

from seismic.hpc.backend import diagnose_backend, format_report

OUTPUT_DIRECTORY = Path("experiments/output")
OUTPUT_PATH = OUTPUT_DIRECTORY / "blas_backend_report.json"


def main() -> None:
    """Affiche et enregistre le diagnostic de backend."""

    report = diagnose_backend()

    print(format_report(report))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as stream:
        json.dump(report.to_dict(), stream, indent=2, ensure_ascii=False)

    print()
    print(f"Rapport enregistre : {OUTPUT_PATH}")

    if not report.uses_onemkl:
        print()
        print(
            "Pour activer Intel oneMKL, l'approche la plus simple est un "
            "environnement conda-forge :"
        )
        print("    conda install -c conda-forge numpy scipy 'libblas=*=*mkl'")
        print(
            "Un NumPy installe depuis PyPI avec pip est lie a OpenBLAS et "
            "ne peut pas utiliser oneMKL."
        )


if __name__ == "__main__":
    main()

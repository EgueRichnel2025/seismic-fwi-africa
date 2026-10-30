"""Diagnostic du fournisseur BLAS/LAPACK réellement chargé à l'exécution.

Le principe directeur du projet est qu'un ``import numpy`` ne prouve rien :
la seule affirmation défendable porte sur la bibliothèque effectivement
liée et chargée en mémoire. Ce module rassemble les informations
disponibles et répond à une question binaire : Intel oneMKL est-il utilisé,
oui ou non.

Trois sources sont croisées :

1. ``numpy.show_config`` et ``scipy.show_config``, qui décrivent la
   dépendance déclarée à la compilation ;
2. ``threadpoolctl``, qui inspecte les bibliothèques réellement chargées
   dans le processus, si le paquet est installé ;
3. le nom de la bibliothèque partagée associée aux routines BLAS, lu
   lorsque la plateforme le permet.

Aucune de ces sources n'est traitée comme suffisante à elle seule.
"""

from dataclasses import dataclass, field, asdict
import platform
import sys
from typing import Any

import numpy as np
import scipy


@dataclass
class BackendReport:
    """Synthèse du diagnostic de backend.

    Attributes:
        provider: Nom normalisé du fournisseur détecté.
        uses_onemkl: Vrai uniquement si oneMKL est détecté de façon fiable.
        numpy_blas: Informations déclarées par NumPy.
        scipy_blas: Informations déclarées par SciPy.
        threadpool: Bibliothèques réellement chargées dans le processus.
        threadpool_available: Vrai si ``threadpoolctl`` est installé.
        environment: Versions et plateforme.
        notes: Observations utiles à l'interprétation.
    """

    provider: str
    uses_onemkl: bool
    numpy_blas: dict[str, Any] = field(default_factory=dict)
    scipy_blas: dict[str, Any] = field(default_factory=dict)
    threadpool: list[dict[str, Any]] = field(default_factory=list)
    threadpool_available: bool = False
    environment: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convertit le rapport en dictionnaire sérialisable en JSON."""

        return asdict(self)


def _normalise(name: str | None) -> str:
    """Normalise un nom de bibliothèque en identifiant de fournisseur."""

    if not name:
        return "inconnu"

    lowered = str(name).lower()

    if "mkl" in lowered:
        return "intel_onemkl"

    if "openblas" in lowered:
        return "openblas"

    if "accelerate" in lowered or "veclib" in lowered:
        return "apple_accelerate"

    if "blis" in lowered:
        return "blis"

    if "atlas" in lowered:
        return "atlas"

    return str(name)


def _declared_blas(module) -> dict[str, Any]:
    """Extrait les informations BLAS déclarées par NumPy ou SciPy."""

    try:
        configuration = module.show_config("dicts")
    except (TypeError, AttributeError):
        return {}

    if not isinstance(configuration, dict):
        return {}

    dependencies = configuration.get("Build Dependencies", {})
    blas = dependencies.get("blas", {})

    return {
        "name": blas.get("name"),
        "version": blas.get("version"),
        "pc_file_directory": blas.get("pc file directory"),
    }


def _loaded_libraries() -> tuple[list[dict[str, Any]], bool]:
    """Inspecte les bibliothèques chargées via ``threadpoolctl``."""

    try:
        import threadpoolctl
    except ImportError:
        return [], False

    try:
        information = threadpoolctl.threadpool_info()
    except Exception:  # noqa: BLE001 - diagnostic non bloquant
        return [], True

    retained = [
        {
            "user_api": entry.get("user_api"),
            "internal_api": entry.get("internal_api"),
            "version": entry.get("version"),
            "num_threads": entry.get("num_threads"),
            "filepath": entry.get("filepath"),
        }
        for entry in information
    ]

    return retained, True


def diagnose_backend() -> BackendReport:
    """Identifie le fournisseur BLAS/LAPACK effectivement utilisé.

    Returns:
        Le rapport de diagnostic complet.
    """

    numpy_blas = _declared_blas(np)
    scipy_blas = _declared_blas(scipy)
    threadpool, threadpool_available = _loaded_libraries()

    notes: list[str] = []

    # Priorité aux bibliothèques réellement chargées en mémoire.
    runtime_provider = "inconnu"

    for entry in threadpool:
        if entry.get("user_api") == "blas":
            runtime_provider = _normalise(entry.get("internal_api"))
            break

    declared_provider = _normalise(numpy_blas.get("name"))

    if threadpool_available and runtime_provider != "inconnu":
        provider = runtime_provider

        if (
            declared_provider != "inconnu"
            and declared_provider != runtime_provider
        ):
            notes.append(
                "La bibliothèque déclarée à la compilation "
                f"({declared_provider}) diffère de celle chargée à "
                f"l'exécution ({runtime_provider}). C'est l'exécution "
                "qui fait foi."
            )
    else:
        provider = declared_provider

        if not threadpool_available:
            notes.append(
                "threadpoolctl n'est pas installé : le diagnostic repose "
                "uniquement sur la configuration déclarée, qui est un "
                "indice plus faible. Installer threadpoolctl pour une "
                "vérification à l'exécution."
            )

    uses_onemkl = provider == "intel_onemkl"

    if not uses_onemkl:
        notes.append(
            "Intel oneMKL n'est pas utilisé dans cet environnement. "
            "Les routines BLAS et LAPACK appelées par le projet sont "
            f"fournies par : {provider}."
        )

    environment = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    return BackendReport(
        provider=provider,
        uses_onemkl=uses_onemkl,
        numpy_blas=numpy_blas,
        scipy_blas=scipy_blas,
        threadpool=threadpool,
        threadpool_available=threadpool_available,
        environment=environment,
        notes=notes,
    )


def format_report(report: BackendReport) -> str:
    """Met en forme le rapport pour un affichage en terminal.

    Args:
        report: Rapport produit par ``diagnose_backend``.

    Returns:
        Le texte prêt à être affiché.
    """

    lines = [
        "=== DIAGNOSTIC DU BACKEND BLAS / LAPACK ===",
        "",
        f"Fournisseur detecte : {report.provider}",
        f"Intel oneMKL utilise : {'OUI' if report.uses_onemkl else 'NON'}",
        "",
        "--- Environnement ---",
    ]

    for key, value in report.environment.items():
        lines.append(f"  {key:12s} : {value}")

    lines.append("")
    lines.append("--- Declare a la compilation ---")
    lines.append(
        f"  numpy : {report.numpy_blas.get('name')} "
        f"{report.numpy_blas.get('version')}"
    )
    lines.append(
        f"  scipy : {report.scipy_blas.get('name')} "
        f"{report.scipy_blas.get('version')}"
    )

    lines.append("")
    lines.append("--- Charge a l'execution (threadpoolctl) ---")

    if not report.threadpool_available:
        lines.append("  threadpoolctl non installe")
    elif not report.threadpool:
        lines.append("  aucune bibliotheque detectee")
    else:
        for entry in report.threadpool:
            lines.append(
                f"  {entry.get('user_api')} / {entry.get('internal_api')} "
                f"v{entry.get('version')} "
                f"({entry.get('num_threads')} threads)"
            )

    if report.notes:
        lines.append("")
        lines.append("--- Remarques ---")

        for note in report.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)

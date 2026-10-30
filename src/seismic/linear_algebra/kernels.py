"""Appels explicites aux routines BLAS et LAPACK utilisées par l'inversion.

Ce module ne crée aucune opération artificielle. Chaque routine exposée ici
correspond à un calcul réellement effectué par l'inversion sismique :

===================  ==========  ====================================
Opération            Routine     Rôle scientifique
===================  ==========  ====================================
produit scalaire     DDOT        courbure y'.s et coefficients de la
                                 récursion L-BFGS, décroissance
                                 directionnelle g'.d
mise à jour vect.    DAXPY       q <- q + alpha * y dans la récursion
                                 L-BFGS, mise à jour du modèle
norme euclidienne    DNRM2       norme du gradient et du résidu,
                                 critères d'arrêt et métriques
matrice-vecteur      DGEMV       application de la Jacobienne réduite
                                 J.dp et de sa transposée J'.r
matrice-matrice      DGEMM       matrice normale J'.J de l'inversion
                                 paramétrique réduite
système symétrique   DPOSV       résolution de (J'J + lambda I) dp = J'r
                                 par factorisation de Cholesky
moindres carrés      DGELSD      résolution par SVD lorsque la matrice
                                 normale est trop mal conditionnée
===================  ==========  ====================================

Les routines sont obtenues via ``scipy.linalg.get_blas_funcs`` et
``scipy.linalg.get_lapack_funcs``, qui renvoient les implémentations
réellement liées à l'installation (OpenBLAS, Intel oneMKL ou autre).
Le fournisseur effectif est diagnostiqué par ``seismic.hpc.backend``.

Les tableaux sont passés en ordre Fortran lorsque c'est utile afin
d'éviter des copies internes.
"""

import numpy as np
from numpy.typing import NDArray

from scipy.linalg import get_blas_funcs, get_lapack_funcs

FloatArray = NDArray[np.float64]

_SAMPLE = np.empty(0, dtype=np.float64)

# Résolution des routines une seule fois à l'import.
_ddot, _daxpy, _dnrm2, _dgemv, _dgemm = get_blas_funcs(
    ("dot", "axpy", "nrm2", "gemv", "gemm"),
    (_SAMPLE,),
)

_dposv, _dgelsd, _dgelsd_lwork = get_lapack_funcs(
    ("posv", "gelsd", "gelsd_lwork"),
    (_SAMPLE,),
)


def dot(x: FloatArray, y: FloatArray) -> float:
    """Produit scalaire ``x'.y`` via BLAS DDOT.

    Args:
        x: Premier vecteur.
        y: Second vecteur, de même taille.

    Returns:
        La valeur du produit scalaire.

    Raises:
        ValueError: Si les tailles diffèrent.
    """

    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()

    if x.size != y.size:
        raise ValueError("Les deux vecteurs doivent avoir la même taille.")

    return float(_ddot(x, y))


def axpy(
    x: FloatArray,
    y: FloatArray,
    alpha: float = 1.0,
) -> FloatArray:
    """Mise à jour vectorielle ``y <- alpha * x + y`` via BLAS DAXPY.

    Args:
        x: Vecteur ajouté.
        y: Vecteur mis à jour.
        alpha: Coefficient scalaire.

    Returns:
        Le vecteur résultant.

    Raises:
        ValueError: Si les tailles diffèrent.
    """

    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()

    if x.size != y.size:
        raise ValueError("Les deux vecteurs doivent avoir la même taille.")

    return _daxpy(x, y, a=float(alpha))


def nrm2(x: FloatArray) -> float:
    """Norme euclidienne ``||x||_2`` via BLAS DNRM2.

    DNRM2 applique une mise à l'échelle interne qui évite les
    dépassements de capacité, contrairement à un calcul naïf.

    Args:
        x: Vecteur dont on calcule la norme.

    Returns:
        La norme euclidienne.
    """

    x = np.ascontiguousarray(x, dtype=np.float64).ravel()

    return float(_dnrm2(x))


def gemv(
    matrix: FloatArray,
    vector: FloatArray,
    *,
    transpose: bool = False,
    alpha: float = 1.0,
) -> FloatArray:
    """Produit matrice-vecteur via BLAS DGEMV.

    Args:
        matrix: Matrice de forme (m, n).
        vector: Vecteur de taille n, ou m si ``transpose`` est vrai.
        transpose: Applique la transposée de la matrice.
        alpha: Coefficient scalaire.

    Returns:
        Le produit ``alpha * A.x`` ou ``alpha * A'.x``.

    Raises:
        ValueError: Si les dimensions sont incompatibles.
    """

    matrix = np.asfortranarray(matrix, dtype=np.float64)
    vector = np.ascontiguousarray(vector, dtype=np.float64).ravel()

    expected = matrix.shape[0] if transpose else matrix.shape[1]

    if vector.size != expected:
        raise ValueError(
            f"Le vecteur doit avoir {expected} éléments, "
            f"reçu {vector.size}."
        )

    return _dgemv(
        float(alpha),
        matrix,
        vector,
        trans=1 if transpose else 0,
    )


def gemm(
    left: FloatArray,
    right: FloatArray,
    *,
    transpose_left: bool = False,
    alpha: float = 1.0,
) -> FloatArray:
    """Produit matrice-matrice via BLAS DGEMM.

    Args:
        left: Matrice de gauche.
        right: Matrice de droite.
        transpose_left: Applique la transposée de la matrice de gauche.
        alpha: Coefficient scalaire.

    Returns:
        Le produit ``alpha * A.B`` ou ``alpha * A'.B``.
    """

    left = np.asfortranarray(left, dtype=np.float64)
    right = np.asfortranarray(right, dtype=np.float64)

    return _dgemm(
        float(alpha),
        left,
        right,
        trans_a=1 if transpose_left else 0,
    )


def solve_symmetric_positive_definite(
    matrix: FloatArray,
    right_hand_side: FloatArray,
) -> FloatArray:
    """Résout ``A.x = b`` avec A symétrique définie positive via LAPACK DPOSV.

    DPOSV effectue une factorisation de Cholesky. Elle est environ deux
    fois moins coûteuse qu'une factorisation LU et exploite la symétrie
    de la matrice normale régularisée ``J'J + lambda I``.

    Args:
        matrix: Matrice symétrique définie positive, forme (n, n).
        right_hand_side: Second membre, de taille n.

    Returns:
        La solution du système.

    Raises:
        np.linalg.LinAlgError: Si la matrice n'est pas définie positive.
    """

    matrix = np.asfortranarray(matrix, dtype=np.float64)
    right_hand_side = np.asfortranarray(
        np.asarray(right_hand_side, dtype=np.float64).reshape(-1, 1)
    )

    _, solution, info = _dposv(matrix, right_hand_side, lower=0)

    if info > 0:
        raise np.linalg.LinAlgError(
            "La matrice normale n'est pas définie positive : le mineur "
            f"principal d'ordre {info} est nul ou négatif. Augmenter la "
            "régularisation ou utiliser la résolution par SVD."
        )

    if info < 0:
        raise ValueError(
            f"Argument invalide numéro {-info} transmis à DPOSV."
        )

    return np.asarray(solution, dtype=np.float64).ravel()


def solve_least_squares(
    matrix: FloatArray,
    right_hand_side: FloatArray,
    *,
    rcond: float = -1.0,
) -> tuple[FloatArray, int, FloatArray]:
    """Résout ``min ||A.x - b||_2`` via LAPACK DGELSD.

    DGELSD repose sur une décomposition en valeurs singulières et traite
    correctement les matrices de rang déficient, ce qui arrive lorsque
    plusieurs paramètres du sous-sol produisent des données presque
    identiques.

    Args:
        matrix: Matrice de forme (m, n).
        right_hand_side: Second membre de taille m.
        rcond: Seuil relatif de troncature des valeurs singulières.
            Une valeur négative utilise la précision machine.

    Returns:
        Le triplet ``(solution, rang, valeurs_singulieres)``.

    Raises:
        np.linalg.LinAlgError: Si l'algorithme SVD ne converge pas.
    """

    matrix = np.asfortranarray(matrix, dtype=np.float64)
    right_hand_side = np.asarray(
        right_hand_side, dtype=np.float64
    ).ravel()

    number_of_rows, number_of_columns = matrix.shape

    if right_hand_side.size != number_of_rows:
        raise ValueError(
            f"Le second membre doit avoir {number_of_rows} éléments."
        )

    # DGELSD exige un second membre de taille max(m, n) et des tailles
    # d'espace de travail calculées par la routine de requête associée.
    padded = np.zeros(
        (max(number_of_rows, number_of_columns), 1),
        dtype=np.float64,
        order="F",
    )
    padded[:number_of_rows, 0] = right_hand_side

    work_size, integer_work_size, _ = _dgelsd_lwork(
        number_of_rows,
        number_of_columns,
        1,
        cond=float(rcond),
    )

    solution_full, singular_values, rank, info = _dgelsd(
        matrix,
        padded,
        int(np.real(work_size)),
        int(integer_work_size),
        cond=float(rcond),
    )

    if info > 0:
        raise np.linalg.LinAlgError(
            "L'algorithme SVD de DGELSD n'a pas convergé."
        )

    if info < 0:
        raise ValueError(
            f"Argument invalide numéro {-info} transmis à DGELSD."
        )

    solution = np.asarray(solution_full, dtype=np.float64).ravel()[
        :number_of_columns
    ]

    return solution, int(rank), np.asarray(singular_values)

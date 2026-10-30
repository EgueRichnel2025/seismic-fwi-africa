"""Noyaux d'algèbre linéaire réellement employés par l'inversion."""

from .kernels import (
    axpy,
    dot,
    gemm,
    gemv,
    nrm2,
    solve_least_squares,
    solve_symmetric_positive_definite,
)

__all__ = [
    "axpy",
    "dot",
    "gemm",
    "gemv",
    "nrm2",
    "solve_least_squares",
    "solve_symmetric_positive_definite",
]

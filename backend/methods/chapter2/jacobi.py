import numpy as np
from typing import Optional


def jacobi(
    A: list[list[float]],
    b: list[float],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100,
) -> dict:
    """
    Método de Jacobi para sistemas de ecuaciones lineales Ax = b.

    Parámetros:
        A       : Matriz de coeficientes (n x n)
        b       : Vector de términos independientes (n)
        x0      : Vector inicial (n)
        tol     : Tolerancia para criterio de parada
        max_iter: Máximo de iteraciones permitidas

    Retorna dict con:
        solution    : Solución aproximada
        iterations  : Lista de registros por iteración
        converged   : bool
        spectral_radius : float
        error_final : float
        message     : str
    """
    A  = np.array(A,  dtype=float)
    b  = np.array(b,  dtype=float)
    x  = np.array(x0, dtype=float)
    n  = len(b)

    # Verificar diagonal no nula
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(
                f"Elemento diagonal cero en posición ({i+1},{i+1}). "
                "Reorganice el sistema para que los pivotes sean distintos de cero."
            )

    iterations = []

    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)

        for i in range(n):
            sigma = b[i]
            for j in range(n):
                if j != i:
                    sigma -= A[i, j] * x[j]   # usa x^(k), NO el nuevo
            x_new[i] = sigma / A[i, i]

        abs_err = float(np.linalg.norm(x_new - x, ord=np.inf))
        rel_err = abs_err / (float(np.linalg.norm(x_new, ord=np.inf)) + 1e-15)
        residual = float(np.linalg.norm(b - A @ x_new, ord=np.inf))

        iterations.append({
            "k":        k,
            "x":        x_new.tolist(),
            "abs_err":  abs_err,
            "rel_err":  rel_err,
            "residual": residual,
        })

        if abs_err < tol:
            return {
                "solution":       x_new.tolist(),
                "iterations":     iterations,
                "converged":      True,
                "spectral_radius": _spectral_radius_jacobi(A),
                "error_final":    abs_err,
                "total_iter":     k,
                "message":        f"Convergió en {k} iteraciones.",
            }

        x = x_new

    return {
        "solution":       x.tolist(),
        "iterations":     iterations,
        "converged":      False,
        "spectral_radius": _spectral_radius_jacobi(A),
        "error_final":    iterations[-1]["abs_err"] if iterations else None,
        "total_iter":     max_iter,
        "message":        f"No convergió en {max_iter} iteraciones. Revise el radio espectral.",
    }


def _spectral_radius_jacobi(A: np.ndarray) -> float:
    """
    Calcula ρ(B_J) donde B_J = -D⁻¹(L+U) es la matriz de iteración de Jacobi.
    Si ρ < 1  →  convergencia garantizada.
    """
    D_inv = np.diag(1.0 / np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A,  1)
    B = -D_inv @ (L + U)
    eigenvalues = np.linalg.eigvals(B)
    return float(np.max(np.abs(eigenvalues)))
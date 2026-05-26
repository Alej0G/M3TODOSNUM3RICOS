import numpy as np


def gauss_seidel(
    A: list[list[float]],
    b: list[float],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100,
) -> dict:
    """
    Método de Gauss-Seidel para Ax = b.

    Diferencia clave con Jacobi: usa los valores YA actualizados
    en la misma iteración k+1 tan pronto están disponibles.

    Fórmula:
        x_i^(k+1) = (1/a_ii) * [b_i - Σ_{j<i} a_ij*x_j^(k+1) - Σ_{j>i} a_ij*x_j^(k)]
    """
    A  = np.array(A,  dtype=float)
    b  = np.array(b,  dtype=float)
    x  = np.array(x0, dtype=float)
    n  = len(b)

    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(
                f"Elemento diagonal cero en posición ({i+1},{i+1})."
            )

    iterations = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            sigma = b[i]
            for j in range(n):
                if j != i:
                    sigma -= A[i, j] * x[j]  # x ya tiene nuevos valores para j < i
            x[i] = sigma / A[i, i]

        abs_err = float(np.linalg.norm(x - x_old, ord=np.inf))
        rel_err = abs_err / (float(np.linalg.norm(x, ord=np.inf)) + 1e-15)
        residual = float(np.linalg.norm(b - A @ x, ord=np.inf))

        iterations.append({
            "k":        k,
            "x":        x.tolist(),
            "abs_err":  abs_err,
            "rel_err":  rel_err,
            "residual": residual,
        })

        if abs_err < tol:
            return {
                "solution":        x.tolist(),
                "iterations":      iterations,
                "converged":       True,
                "spectral_radius": _spectral_radius_gs(A),
                "error_final":     abs_err,
                "total_iter":      k,
                "message":         f"Convergió en {k} iteraciones.",
            }

    return {
        "solution":        x.tolist(),
        "iterations":      iterations,
        "converged":       False,
        "spectral_radius": _spectral_radius_gs(A),
        "error_final":     iterations[-1]["abs_err"] if iterations else None,
        "total_iter":      max_iter,
        "message":         f"No convergió en {max_iter} iteraciones.",
    }


def _spectral_radius_gs(A: np.ndarray) -> float:
    """
    Radio espectral de la matriz de iteración de Gauss-Seidel:
        B_GS = -(D + L)^{-1} * U
    """
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A,  1)
    try:
        DL_inv = np.linalg.inv(D + L)
    except np.linalg.LinAlgError:
        return float("inf")
    B = -DL_inv @ U
    eigenvalues = np.linalg.eigvals(B)
    return float(np.max(np.abs(eigenvalues)))
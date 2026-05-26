import numpy as np


def sor(
    A: list[list[float]],
    b: list[float],
    x0: list[float],
    omega: float = 1.25,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> dict:
    """
    Método SOR (Successive Over-Relaxation) para Ax = b.

    Combina Gauss-Seidel con un factor de relajación ω:
        x_i^(k+1) = (1-ω)*x_i^(k) + (ω/a_ii)*[b_i - Σ_{j≠i} a_ij*x_j]

    Rangos de ω:
        ω = 1.0  →  equivale exactamente a Gauss-Seidel
        ω > 1.0  →  sobre-relajación (acelera convergencia, rango típico 1.0–1.9)
        ω < 1.0  →  sub-relajación   (aumenta estabilidad)
        ω ≥ 2.0  →  siempre diverge (condición necesaria: 0 < ω < 2)
    """
    if not (0.0 < omega < 2.0):
        raise ValueError(
            f"El factor ω = {omega} está fuera del rango válido (0, 2). "
            "El método SOR requiere estrictamente 0 < ω < 2."
        )

    A  = np.array(A,  dtype=float)
    b  = np.array(b,  dtype=float)
    x  = np.array(x0, dtype=float)
    n  = len(b)

    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(f"Elemento diagonal cero en posición ({i+1},{i+1}).")

    iterations = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            # Paso Gauss-Seidel puro
            sigma = b[i]
            for j in range(n):
                if j != i:
                    sigma -= A[i, j] * x[j]
            x_gs = sigma / A[i, i]

            # Relajación SOR
            x[i] = (1.0 - omega) * x_old[i] + omega * x_gs

        abs_err = float(np.linalg.norm(x - x_old, ord=np.inf))
        rel_err = abs_err / (float(np.linalg.norm(x, ord=np.inf)) + 1e-15)
        residual = float(np.linalg.norm(b - A @ x, ord=np.inf))

        iterations.append({
            "k":        k,
            "x":        x.tolist(),
            "abs_err":  abs_err,
            "rel_err":  rel_err,
            "residual": residual,
            "omega":    omega,
        })

        if abs_err < tol:
            return {
                "solution":        x.tolist(),
                "iterations":      iterations,
                "converged":       True,
                "spectral_radius": _spectral_radius_sor(A, omega),
                "omega":           omega,
                "error_final":     abs_err,
                "total_iter":      k,
                "message":         f"Convergió en {k} iteraciones con ω = {omega}.",
            }

    return {
        "solution":        x.tolist(),
        "iterations":      iterations,
        "converged":       False,
        "spectral_radius": _spectral_radius_sor(A, omega),
        "omega":           omega,
        "error_final":     iterations[-1]["abs_err"] if iterations else None,
        "total_iter":      max_iter,
        "message":         f"No convergió en {max_iter} iteraciones con ω = {omega}.",
    }


def _spectral_radius_sor(A: np.ndarray, omega: float) -> float:
    """
    Matriz de iteración de SOR:
        B_SOR = (D + ω*L)^{-1} * [(1-ω)*D - ω*U]
    """
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A,  1)
    try:
        M_inv = np.linalg.inv(D + omega * L)
    except np.linalg.LinAlgError:
        return float("inf")
    B = M_inv @ ((1.0 - omega) * D - omega * U)
    eigenvalues = np.linalg.eigvals(B)
    return float(np.max(np.abs(eigenvalues)))
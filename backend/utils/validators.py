import numpy as np


def validate_input(data: dict) -> list[str]:
    """
    Valida todos los campos del request.
    Retorna lista de strings de error (lista vacía = todo OK).
    """
    errors = []

    # ── Campos obligatorios ──────────────────────────────────────────────────
    required = ["A", "b", "x0", "tol", "max_iter"]
    missing  = [f for f in required if f not in data]
    if missing:
        errors.append(f"Campos requeridos faltantes: {missing}")
        return errors   # sin estos no podemos continuar

    A        = data["A"]
    b        = data["b"]
    x0       = data["x0"]
    tol      = data["tol"]
    max_iter = data["max_iter"]

    # ── Matriz A ─────────────────────────────────────────────────────────────
    if not isinstance(A, list) or not all(isinstance(r, list) for r in A):
        errors.append("'A' debe ser una lista de listas.")
    else:
        n = len(A)
        if not (2 <= n <= 8):
            errors.append(f"La matriz debe ser de tamaño 2×2 a 8×8. Recibido: {n}×?")
        else:
            if any(len(row) != n for row in A):
                errors.append("La matriz A no es cuadrada.")
            else:
                try:
                    A_np = np.array(A, dtype=float)

                    if not np.all(np.isfinite(A_np)):
                        errors.append("La matriz A contiene valores no finitos (inf o NaN).")

                    # Diagonal cero → división por cero en los métodos
                    zero_diag = [i + 1 for i in range(n) if abs(A_np[i, i]) < 1e-15]
                    if zero_diag:
                        errors.append(
                            f"Diagonal cero en filas: {zero_diag}. "
                            "Reordene el sistema para que los elementos diagonales no sean cero."
                        )

                    # Matriz casi singular
                    cond = np.linalg.cond(A_np)
                    if cond > 1e12:
                        errors.append(
                            f"La matriz es casi singular (κ = {cond:.2e}). "
                            "Los resultados pueden ser numéricamente inestables."
                        )

                except (ValueError, TypeError):
                    errors.append("La matriz A contiene valores no numéricos.")

    # ── Vectores b y x0 ──────────────────────────────────────────────────────
    n_safe = len(A) if isinstance(A, list) else 0
    for name, vec in [("b", b), ("x0", x0)]:
        if not isinstance(vec, list):
            errors.append(f"'{name}' debe ser una lista.")
        elif len(vec) != n_safe:
            errors.append(
                f"'{name}' debe tener {n_safe} elementos; tiene {len(vec)}."
            )
        else:
            try:
                v = np.array(vec, dtype=float)
                if not np.all(np.isfinite(v)):
                    errors.append(f"'{name}' contiene valores no finitos.")
            except (ValueError, TypeError):
                errors.append(f"'{name}' contiene valores no numéricos.")

    # ── Tolerancia ────────────────────────────────────────────────────────────
    try:
        tol_f = float(tol)
        if not (1e-15 < tol_f < 1.0):
            errors.append("La tolerancia debe estar en el rango (1e-15, 1). Ejemplo: 1e-6")
    except (ValueError, TypeError):
        errors.append("La tolerancia debe ser un número positivo.")

    # ── Máximo de iteraciones ─────────────────────────────────────────────────
    try:
        mi = int(max_iter)
        if mi < 1 or mi > 10_000:
            errors.append("El máximo de iteraciones debe estar entre 1 y 10000.")
    except (ValueError, TypeError):
        errors.append("El máximo de iteraciones debe ser un entero positivo.")

    # ── Factor ω (solo SOR) ───────────────────────────────────────────────────
    if "omega" in data:
        try:
            w = float(data["omega"])
            if not (0.0 < w < 2.0):
                errors.append(
                    f"El factor ω = {w} debe estar en el intervalo abierto (0, 2). "
                    "Valores típicos: 1.0 a 1.9"
                )
        except (ValueError, TypeError):
            errors.append("El factor ω debe ser un número.")

    return errors


def validate_matrix_only(data: dict) -> list[str]:
    """Validación ligera: solo verifica la matriz A (para el endpoint matrix-info)."""
    errors = []
    if "A" not in data:
        return ["Campo 'A' requerido."]
    A = data["A"]
    if not isinstance(A, list) or not all(isinstance(r, list) for r in A):
        return ["'A' debe ser una lista de listas."]
    try:
        np.array(A, dtype=float)
    except (ValueError, TypeError):
        errors.append("La matriz A contiene valores no numéricos.")
    return errors
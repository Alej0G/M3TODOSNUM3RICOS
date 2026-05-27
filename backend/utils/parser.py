import numpy as np


def parse_matrix(raw: list) -> np.ndarray:
    """
    Convierte lista de listas (viene del JSON) a numpy array float.
    Lanza ValueError si los datos no son numéricos.
    """
    try:
        return np.array(raw, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"No se pudo convertir la matriz: {e}")


def parse_vector(raw: list) -> np.ndarray:
    """Convierte lista plana a numpy array float."""
    try:
        return np.array(raw, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"No se pudo convertir el vector: {e}")


def matrix_info(A: np.ndarray) -> dict:
    """
    Propiedades generales de la matriz para mostrar en el frontend
    antes o después de ejecutar los métodos.
    """
    n = A.shape[0]

    # Dominancia diagonal estricta
    diag     = np.abs(np.diag(A))
    off_diag = np.array([
        sum(abs(A[i, j]) for j in range(n) if j != i)
        for i in range(n)
    ])
    is_dd = bool(np.all(diag > off_diag))

    return {
        "size":             n,
        "determinant":      float(np.linalg.det(A)),
        "condition_number": float(np.linalg.cond(A)),
        "is_diag_dominant": is_dd,
        "is_symmetric":     bool(np.allclose(A, A.T)),
        "rank":             int(np.linalg.matrix_rank(A)),
    }
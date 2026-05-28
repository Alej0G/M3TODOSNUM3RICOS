import numpy as np
import sympy as sp

def vandermonde(x_puntos, y_puntos):
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)

    n = len(x)

    V = np.vander(x, increasing=True)

    coeficientes = np.linalg.solve(V, y)

    t = sp.Symbol('x')

    polinomio = sum(
        coeficientes[i] * (t**i)
        for i in range(n)
    )

    return sp.expand(polinomio)
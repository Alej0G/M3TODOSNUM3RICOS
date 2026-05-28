import numpy as np
import sympy as sp


def vandermonde_interpolacion(x_datos, y_datos):
    n = len(x_datos)

    matriz = np.vander(x_datos, increasing=True)

    coeficientes = np.linalg.solve(matriz, y_datos)

    x = sp.Symbol('x')

    polinomio = 0

    for i in range(n):
        polinomio += coeficientes[i] * x**i

    polinomio = sp.expand(polinomio)

    return polinomio, coeficientes
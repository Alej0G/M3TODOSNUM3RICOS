import numpy as np
import sympy as sp


#Método de interpolación de Newton

def newton_interpolante(x_puntos, y_puntos):
    n = len(y_puntos)

    tabla = np.zeros([n, n])

    tabla[:, 0] = y_puntos

    for j in range(1, n):
        for i in range(n - j):
            tabla[i][j] = (
                (tabla[i+1][j-1] - tabla[i][j-1]) /
                (x_puntos[i+j] - x_puntos[i])
            )

    coeficientes = tabla[0, :]

    t = sp.Symbol('x')

    polinomio = coeficientes[0]

    acumulado = 1

    for i in range(1, n):
        acumulado *= (t - x_puntos[i-1])
        polinomio += coeficientes[i] * acumulado

    return sp.expand(polinomio)

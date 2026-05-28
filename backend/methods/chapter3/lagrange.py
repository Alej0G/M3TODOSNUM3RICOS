import sympy as sp


def lagrange(x_puntos, y_puntos):
    t = sp.Symbol('x')

    polinomio = 0

    n = len(x_puntos)

    for i in range(n):
        L = 1

        for j in range(n):
            if i != j:
                L *= (
                    (t - x_puntos[j]) /
                    (x_puntos[i] - x_puntos[j])
                )

        polinomio += y_puntos[i] * L

    return sp.expand(polinomio)
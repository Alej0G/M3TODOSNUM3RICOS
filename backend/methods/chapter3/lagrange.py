import sympy as sp


def lagrange_interpolacion(x_datos, y_datos):
    x = sp.Symbol('x')

    n = len(x_datos)

    polinomio = 0

    for i in range(n):
        termino = y_datos[i]

        for j in range(n):
            if i != j:
                termino *= (x - x_datos[j]) / (x_datos[i] - x_datos[j])

        polinomio += termino

    polinomio = sp.expand(polinomio)

    return polinomio
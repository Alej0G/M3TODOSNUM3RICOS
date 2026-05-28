import pandas as pd


def punto_fijo(g, x0, tol, max_iter):
    iteraciones = []

    x_ant = x0

    for i in range(max_iter):
        xn = g(x_ant)

        e_abs = abs(xn - x_ant)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        e_cond = e_abs

        iteraciones.append({
            'Iter': i + 1,
            'x_n': xn,
            'f(x)': xn - x_ant,
            'E_Abs': e_abs,
            'E_Rel': e_rel,
            'E_Cond': e_cond
        })

        if e_rel < tol:
            break

        x_ant = xn

    return pd.DataFrame(iteraciones)
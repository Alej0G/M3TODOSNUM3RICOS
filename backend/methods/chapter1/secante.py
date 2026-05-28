import pandas as pd


def secante(f, x0, x1, tol, max_iter):
    iteraciones = []

    for i in range(max_iter):
        f0 = f(x0)
        f1 = f(x1)

        if f1 - f0 == 0:
            break

        xn = x1 - (f1 * (x1 - x0)) / (f1 - f0)

        fxn = f(xn)

        e_abs = abs(xn - x1)
        e_rel = e_abs / abs(xn) if xn != 0 else e_abs
        e_cond = abs(fxn)

        iteraciones.append({
            'Iter': i + 1,
            'x_n': xn,
            'f(x)': fxn,
            'E_Abs': e_abs,
            'E_Rel': e_rel,
            'E_Cond': e_cond
        })

        if e_rel < tol:
            break

        x0 = x1
        x1 = xn

    return pd.DataFrame(iteraciones)
import pandas as pd

def regla_falsa(f, a, b, tol, max_iter):
    iteraciones = []
    x_ant = a

    for i in range(max_iter):
        fa = f(a)
        fb = f(b)

        if fb - fa == 0:
            break

        xm = b - (fb * (b - a)) / (fb - fa)
        fxm = f(xm)

        e_abs = abs(xm - x_ant)
        e_rel = e_abs / abs(xm) if xm != 0 else e_abs
        e_cond = abs(fxm)

        iteraciones.append({
            'Iter': i + 1,
            'x_n': xm,
            'f(x)': fxm,
            'E_Abs': e_abs,
            'E_Rel': e_rel,
            'E_Cond': e_cond
        })

        if i > 0 and e_rel < tol:
            break

        if fa * fxm < 0:
            b = xm
        else:
            a = xm

        x_ant = xm

    return pd.DataFrame(iteraciones)
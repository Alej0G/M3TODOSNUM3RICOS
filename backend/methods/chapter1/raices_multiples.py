import pandas as pd


def raices_multiples(f, df, ddf, x0, tol, max_iter):
    iteraciones = []

    x_ant = x0

    for i in range(max_iter):
        fx = f(x_ant)
        dfx = df(x_ant)
        ddfx = ddf(x_ant)

        denominador = (dfx**2) - (fx * ddfx)

        if denominador == 0:
            break

        xn = x_ant - (fx * dfx) / denominador

        fxn = f(xn)

        e_abs = abs(xn - x_ant)
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

        x_ant = xn

    return pd.DataFrame(iteraciones)
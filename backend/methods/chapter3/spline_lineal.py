import numpy as np
from scipy.interpolate import interp1d


def spline_lineal(x_puntos, y_puntos):
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)

    return interp1d(
        x,
        y,
        kind='linear',
        fill_value='extrapolate'
    )
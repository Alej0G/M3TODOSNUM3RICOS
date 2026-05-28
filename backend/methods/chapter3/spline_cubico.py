import numpy as np
from scipy.interpolate import CubicSpline


def spline_cubico(x_puntos, y_puntos):
    x = np.array(x_puntos, dtype=float)
    y = np.array(y_puntos, dtype=float)

    return CubicSpline(x, y)
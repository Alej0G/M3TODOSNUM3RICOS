import numpy as np


def spline_lineal(x_datos, y_datos):
    segmentos = []

    for i in range(len(x_datos) - 1):
        x0 = x_datos[i]
        x1 = x_datos[i + 1]

        y0 = y_datos[i]
        y1 = y_datos[i + 1]

        pendiente = (y1 - y0) / (x1 - x0)

        segmentos.append({
            'intervalo': [x0, x1],
            'pendiente': pendiente,
            'intercepto': y0
        })

    return segmentos
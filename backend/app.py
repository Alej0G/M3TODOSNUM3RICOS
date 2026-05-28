import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template

from methods.chapter2.jacobi import jacobi
from methods.chapter2.gauss_seidel import gauss_seidel
from methods.chapter2.sor import sor

from utils.validators import validate_input
from utils.parser import parse_matrix, parse_vector, matrix_info
from utils.errors import format_error_response, format_exception
from utils.graphing import (
    plot_error_comparison,
    plot_spectral_radii
)

app = Flask(__name__)


# ─────────────────────────────────────────────
# PÁGINAS
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chapter1")
def chapter1():
    return render_template("chapter1.html")


@app.route("/chapter2")
def chapter2():

    return render_template(
        "chapter2.html",
        rango=range(7)
    )


@app.route("/chapter3")
def chapter3():
    return render_template("chapter3.html")


# ─────────────────────────────────────────────
# RESULTADOS CAPÍTULO 2
# ─────────────────────────────────────────────

@app.route("/chapter2/resultado", methods=["POST"])
def resultado_capitulo2():

    comparar = request.form.get("comparar") == "on"
    metodo = request.form.get("metodo")

    # ─────────────────────────────────────────
    # VALIDAR MÉTODO
    # ─────────────────────────────────────────

    if not comparar and not metodo:

        return render_template(
            "chapter2.html",
            rango=range(7),
            error="Debes seleccionar un método."
        )

    # ─────────────────────────────────────────
    # DIMENSIÓN
    # ─────────────────────────────────────────

    try:

        dim = int(request.form.get("dimension", 3))

        if not (2 <= dim <= 7):
            raise ValueError

    except:

        return render_template(
            "chapter2.html",
            rango=range(7),
            error="Dimensión inválida."
        )

    # ─────────────────────────────────────────
    # MATRICES Y VECTORES
    # ─────────────────────────────────────────

    try:

        A = [
            [
                float(request.form.get(f"A{i}{j}", 0))
                for j in range(dim)
            ]
            for i in range(dim)
        ]

        b = [
            float(request.form.get(f"b{i}", 0))
            for i in range(dim)
        ]

        x0 = [
            float(request.form.get(f"x0{i}", 0))
            for i in range(dim)
        ]

    except:

        return render_template(
            "chapter2.html",
            rango=range(7),
            error="Todos los campos deben ser numéricos."
        )

    # ─────────────────────────────────────────
    # PARÁMETROS
    # ─────────────────────────────────────────

    try:

        tol = float(request.form.get("tol", 1e-6))

        niter = int(request.form.get("niter", 100))

        w1_raw = request.form.get("w1")

        w1 = (
            float(w1_raw)
            if w1_raw not in [None, ""]
            else 1.2
        )

    except:

        return render_template(
            "chapter2.html",
            rango=range(7),
            error="Parámetros inválidos."
        )

    # ─────────────────────────────────────────
    # VALIDACIÓN
    # ─────────────────────────────────────────

    errores = validate_input({
        "A": A,
        "b": b,
        "x0": x0,
        "tol": tol,
        "max_iter": niter,
        "omega": w1,
    })

    if errores:

        return render_template(
            "chapter2.html",
            rango=range(7),
            error=" | ".join(errores)
        )

    # ─────────────────────────────────────────
    # EJECUTAR MÉTODOS
    # ─────────────────────────────────────────

    res_jacobi = None
    res_gauss = None
    res_sor1 = None

    # JACOBI

    try:

        if comparar or metodo == "jacobi":

            res_jacobi = jacobi(
                A,
                b,
                x0,
                tol,
                niter
            )

    except Exception as e:

        print("ERROR JACOBI:", e)

    # GAUSS-SEIDEL

    try:

        if comparar or metodo == "gauss_seidel":

            res_gauss = gauss_seidel(
                A,
                b,
                x0,
                tol,
                niter
            )

    except Exception as e:

        print("ERROR GS:", e)

    # SOR

    try:

        if comparar or metodo == "sor":

            res_sor1 = sor(
                A,
                b,
                x0,
                w1,
                tol,
                niter
            )

    except Exception as e:

        print("ERROR SOR:", e)

    # ─────────────────────────────────────────
    # DICCIONARIO PARA GRÁFICAS
    # graphing.py necesita:
    # jacobi / gauss_seidel / sor
    # ─────────────────────────────────────────

    metodos_graph = {}

    if res_jacobi:
        metodos_graph["jacobi"] = res_jacobi

    if res_gauss:
        metodos_graph["gauss_seidel"] = res_gauss

    if res_sor1:
        metodos_graph["sor"] = res_sor1

    # ─────────────────────────────────────────
    # VALIDAR RESULTADOS
    # ─────────────────────────────────────────

    if not metodos_graph:

        return render_template(
            "chapter2.html",
            rango=range(7),
            error="No se pudo ejecutar ningún método."
        )

    # ─────────────────────────────────────────
    # MEJOR MÉTODO
    # ─────────────────────────────────────────

    mejor = None

    if comparar:

        comparacion = {}

        if res_jacobi:
            comparacion["Jacobi"] = res_jacobi

        if res_gauss:
            comparacion["Gauss-Seidel"] = res_gauss

        if res_sor1:
            comparacion[f"SOR (w={w1})"] = res_sor1

        mejor = min(
            comparacion.items(),
            key=lambda item:
            item[1]["total_iter"]
            if item[1].get("converged")
            else 999999
        )

        mejor = mejor[0]

    # ─────────────────────────────────────────
    # NOMBRE BONITO
    # ─────────────────────────────────────────

    nombres = {
        "jacobi": "Jacobi",
        "gauss_seidel": "Gauss-Seidel",
        "sor": "SOR",
    }

    metodo_elegido = (
        nombres.get(metodo)
        if not comparar
        else None
    )

    # ─────────────────────────────────────────
    # GRÁFICAS
    # ─────────────────────────────────────────

    try:

        graph_errors = plot_error_comparison(
            metodos_graph
        )

    except Exception as e:

        print("ERROR GRAPH ERRORS:", e)
        graph_errors = ""

    try:

        radii = {}

        if res_jacobi:
            radii["Jacobi"] = res_jacobi["spectral_radius"]

        if res_gauss:
            radii["Gauss-Seidel"] = res_gauss["spectral_radius"]

        if res_sor1:
            radii[f"SOR (w={w1})"] = res_sor1["spectral_radius"]

        graph_spectral = plot_spectral_radii(
            radii
        )

    except Exception as e:

        print("ERROR GRAPH SPECTRAL:", e)
        graph_spectral = ""

    # ─────────────────────────────────────────
    # RENDER FINAL
    # ─────────────────────────────────────────

    return render_template(

        "chapter2.html",

        rango=range(7),

        comparar=comparar,

        res_jacobi=res_jacobi,
        res_gauss=res_gauss,
        res_sor1=res_sor1,

        mejor=mejor,

        metodo_elegido=metodo_elegido,

        w1=w1,

        graph_errors=graph_errors,
        graph_spectral=graph_spectral,
    )


# ─────────────────────────────────────────────
# API JSON
# ─────────────────────────────────────────────

@app.route("/api/chapter2/solve-all", methods=["POST"])
def api_chapter2_solve_all():

    data = request.get_json(force=True)

    errors = validate_input(data)

    if errors:
        return jsonify(format_error_response(errors)), 422

    try:

        A = parse_matrix(data["A"])

        b = parse_vector(data["b"])

        x0 = parse_vector(data["x0"])

        tol = float(data["tol"])

        max_iter = int(data["max_iter"])

        omega = float(data.get("omega", 1.25))

        results = {}

        for nombre, func in [

            (
                "jacobi",
                lambda: jacobi(
                    A.tolist(),
                    b.tolist(),
                    x0.tolist(),
                    tol,
                    max_iter
                )
            ),

            (
                "gauss_seidel",
                lambda: gauss_seidel(
                    A.tolist(),
                    b.tolist(),
                    x0.tolist(),
                    tol,
                    max_iter
                )
            ),

            (
                "sor",
                lambda: sor(
                    A.tolist(),
                    b.tolist(),
                    x0.tolist(),
                    omega,
                    tol,
                    max_iter
                )
            ),
        ]:

            try:

                results[nombre] = func()

            except Exception as e:

                results[nombre] = {
                    "error": str(e),
                    "converged": False
                }

        results["matrix_info"] = matrix_info(A)

        return jsonify({
            "ok": True,
            "results": results
        })

    except Exception as e:

        return jsonify(
            format_exception(e)
        ), 500


# ─────────────────────────────────────────────

if __name__ == "__main__":

    app.run(debug=True)
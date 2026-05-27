import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template, redirect, url_for

from methods.chapter2.jacobi       import jacobi
from methods.chapter2.gauss_seidel import gauss_seidel
from methods.chapter2.sor          import sor

from utils.validators import validate_input, validate_matrix_only
from utils.parser     import parse_matrix, parse_vector, matrix_info
from utils.errors     import format_error_response, format_exception
from utils.graphing   import plot_error_comparison, plot_spectral_radii

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Páginas generales
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chapter1")
def chapter1():
    return render_template("chapter1.html")

@app.route("/chapter3")
def chapter3():
    return render_template("chapter3.html")


# ─────────────────────────────────────────────────────────────────────────────
# Capítulo 2 — Formulario (GET)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/chapter2")
def chapter2():
    """Muestra el formulario vacío."""
    return render_template("chapter2.html", rango=range(7))


# ─────────────────────────────────────────────────────────────────────────────
# Capítulo 2 — Resultado (POST)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/chapter2/resultado", methods=["POST"])
def resultado_capitulo2():
    """
    Recibe el formulario, ejecuta los métodos seleccionados
    y renderiza la tabla de resultados.
    """
    comparar = request.form.get("comparar") == "on"
    metodo   = request.form.get("metodo")   # "jacobi" | "gauss_seidel" | "sor" | None

    # Validar que eligieron algo
    if not comparar and not metodo:
        return render_template("chapter2.html", rango=range(7),
                               error="Debes seleccionar un método o activar comparación.")

    # Leer dimensión
    try:
        dim = int(request.form.get("dimension", 3))
        if not (2 <= dim <= 7):
            raise ValueError
    except (ValueError, TypeError):
        return render_template("chapter2.html", rango=range(7),
                               error="La dimensión debe ser un número entre 2 y 7.")

    # Leer A, b, x0
    try:
        A  = [[float(request.form.get(f"A{i}{j}", 0)) for j in range(dim)] for i in range(dim)]
        b  = [float(request.form.get(f"b{i}", 0))  for i in range(dim)]
        x0 = [float(request.form.get(f"x0{i}", 0)) for i in range(dim)]
    except (ValueError, TypeError):
        return render_template("chapter2.html", rango=range(7),
                               error="Todos los campos deben contener solo números.")

    # Leer parámetros
    try:
        tol   = float(request.form.get("tol",   1e-6))
        niter = int(request.form.get("niter",   100))
        w1    = float(request.form.get("w1",    1.0))
        w2    = float(request.form.get("w2",    1.25))
        w3    = float(request.form.get("w3",    1.5))
    except (ValueError, TypeError):
        return render_template("chapter2.html", rango=range(7),
                               error="Los parámetros tol, niter y omega deben ser numéricos.")

    # Validar con tu validador existente
    errores = validate_input({
        "A": A, "b": b, "x0": x0,
        "tol": tol, "max_iter": niter, "omega": w1,
    })
    if errores:
        return render_template("chapter2.html", rango=range(7),
                               error=" | ".join(errores))

    # ── Ejecutar métodos ─────────────────────────────────────────────────────
    res_jacobi = None
    res_gauss  = None
    res_sor1   = None
    res_sor2   = None
    res_sor3   = None

    try:
        if comparar or metodo == "jacobi":
            res_jacobi = jacobi(A, b, x0, tol, niter)
    except Exception as e:
        res_jacobi = None

    try:
        if comparar or metodo == "gauss_seidel":
            res_gauss = gauss_seidel(A, b, x0, tol, niter)
    except Exception as e:
        res_gauss = None

    try:
        if comparar or metodo == "sor":
            res_sor1 = sor(A, b, x0, w1, tol, niter)
    except Exception as e:
        res_sor1 = None

    try:
        if comparar:
            res_sor2 = sor(A, b, x0, w2, tol, niter)
            res_sor3 = sor(A, b, x0, w3, tol, niter)
    except Exception as e:
        res_sor2 = None
        res_sor3 = None

    # ── Armar dict de métodos para comparación ───────────────────────────────
    metodos = {}
    if res_jacobi: metodos["Jacobi"]            = res_jacobi
    if res_gauss:  metodos["Gauss-Seidel"]       = res_gauss
    if res_sor1:   metodos[f"SOR (w={w1})"]      = res_sor1
    if res_sor2:   metodos[f"SOR (w={w2})"]      = res_sor2
    if res_sor3:   metodos[f"SOR (w={w3})"]      = res_sor3

    if not metodos:
        return render_template("chapter2.html", rango=range(7),
                               error="No se ejecutó ningún método válido.")

    # ── Mejor método (el que convergió en menos iteraciones) ─────────────────
    mejor = None
    if comparar:
        mejor = min(
            metodos.items(),
            key=lambda item: item[1]["total_iter"] if item[1].get("converged") else 9999
        )
        mejor = mejor[0]   # solo el nombre

    # ── Nombre legible del método elegido ────────────────────────────────────
    nombres = {
        "jacobi":       "Jacobi",
        "gauss_seidel": "Gauss-Seidel",
        "sor":          "SOR",
    }
    metodo_elegido = nombres.get(metodo) if not comparar else None

    # ── Gráficas ─────────────────────────────────────────────────────────────
    graph_errors   = plot_error_comparison(metodos)
    radii = {
        k: v["spectral_radius"]
        for k, v in metodos.items()
        if "spectral_radius" in v
    }
    graph_spectral = plot_spectral_radii(radii)

    return render_template("resultado_chapter2.html",
        comparar       = comparar,
        res_jacobi     = res_jacobi,
        res_gauss      = res_gauss,
        res_sor1       = res_sor1,
        res_sor2       = res_sor2,
        res_sor3       = res_sor3,
        metodos        = metodos,
        mejor          = mejor,
        metodo_elegido = metodo_elegido,
        w1=w1, w2=w2, w3=w3,
        graph_errors   = graph_errors,
        graph_spectral = graph_spectral,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Capítulo 2 — Informe comparativo completo (POST)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/chapter2/informe", methods=["POST"])
def informe_capitulo2():
    """
    Siempre corre los 5 métodos (Jacobi, GS, SOR×3) y genera
    un resumen comparativo con el mejor método destacado.
    """
    try:
        dim   = int(request.form.get("dimension", 3))
        A     = [[float(request.form.get(f"A{i}{j}", 0)) for j in range(dim)] for i in range(dim)]
        b     = [float(request.form.get(f"b{i}", 0))  for i in range(dim)]
        x0    = [float(request.form.get(f"x0{i}", 0)) for i in range(dim)]
        tol   = float(request.form.get("tol",   1e-6))
        niter = int(request.form.get("niter",   100))
        w1    = float(request.form.get("w1",    1.0))
        w2    = float(request.form.get("w2",    1.25))
        w3    = float(request.form.get("w3",    1.5))
    except (ValueError, TypeError):
        return render_template("chapter2.html", rango=range(7),
                               error="Datos inválidos para el informe.")

    metodos_resultados = {
        "Jacobi":          jacobi(A, b, x0, tol, niter),
        "Gauss-Seidel":    gauss_seidel(A, b, x0, tol, niter),
        f"SOR (w={w1})":   sor(A, b, x0, w1, tol, niter),
        f"SOR (w={w2})":   sor(A, b, x0, w2, tol, niter),
        f"SOR (w={w3})":   sor(A, b, x0, w3, tol, niter),
    }

    # Resumen: una fila por método
    resumen = []
    for nombre, resultado in metodos_resultados.items():
        resumen.append({
            "metodo":    nombre,
            "niter":     resultado.get("total_iter", "—"),
            "solucion":  resultado.get("solution",   []),
            "error":     resultado.get("error_final", float("inf")),
            "convergio": resultado.get("converged",  False),
            "spectral":  resultado.get("spectral_radius", "—"),
        })

    # Mejor = convergió con menos iteraciones
    convergidos = [r for r in resumen if r["convergio"]]
    mejor = min(convergidos, key=lambda r: r["niter"])["metodo"] if convergidos else "Ninguno convergió"

    graph_errors   = plot_error_comparison(metodos_resultados)
    radii          = {r["metodo"]: r["spectral"] for r in resumen if isinstance(r["spectral"], float)}
    graph_spectral = plot_spectral_radii(radii)

    return render_template("informe_chapter2.html",
        resumen        = resumen,
        mejor          = mejor,
        graph_errors   = graph_errors,
        graph_spectral = graph_spectral,
    )


# ─────────────────────────────────────────────────────────────────────────────
# API JSON (para pruebas desde Postman o curl)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chapter2/solve-all", methods=["POST"])
def api_chapter2_solve_all():
    data   = request.get_json(force=True)
    errors = validate_input(data)
    if errors:
        return jsonify(format_error_response(errors)), 422
    try:
        A        = parse_matrix(data["A"])
        b        = parse_vector(data["b"])
        x0       = parse_vector(data["x0"])
        tol      = float(data["tol"])
        max_iter = int(data["max_iter"])
        omega    = float(data.get("omega", 1.25))

        results = {}
        for nombre, func in [
            ("jacobi",       lambda: jacobi(A.tolist(), b.tolist(), x0.tolist(), tol, max_iter)),
            ("gauss_seidel", lambda: gauss_seidel(A.tolist(), b.tolist(), x0.tolist(), tol, max_iter)),
            ("sor",          lambda: sor(A.tolist(), b.tolist(), x0.tolist(), omega, tol, max_iter)),
        ]:
            try:
                results[nombre] = func()
            except Exception as e:
                results[nombre] = {"error": str(e), "converged": False}

        results["matrix_info"] = matrix_info(A)
        return jsonify({"ok": True, "results": results})

    except Exception as e:
        return jsonify(format_exception(e)), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template

from methods.chapter2 import jacobi, gauss_seidel, sor
from utils import (
    validate_input,
    validate_matrix_only,
    parse_matrix,
    parse_vector,
    matrix_info,
    format_error_response,
    format_exception,
    plot_error_comparison,
    plot_spectral_radii,
)

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rutas de páginas (templates)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chapter1")
def chapter1():
    return render_template("chapter1.html")

@app.route("/chapter2")
def chapter2():
    return render_template("chapter2.html")

@app.route("/chapter3")
def chapter3():
    return render_template("chapter3.html")


# ─────────────────────────────────────────────────────────────────────────────
# API — Utilidad: información de la matriz
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/matrix-info", methods=["POST"])
def api_matrix_info():
    """
    Recibe la matriz y retorna sus propiedades sin ejecutar ningún método.
    Body: { "A": [[...], ...] }
    """
    data   = request.get_json(force=True)
    errors = validate_matrix_only(data)
    if errors:
        return jsonify(format_error_response(errors)), 422
    try:
        A = parse_matrix(data["A"])
        return jsonify({"ok": True, "info": matrix_info(A)})
    except Exception as e:
        return jsonify(format_exception(e)), 500


# ─────────────────────────────────────────────────────────────────────────────
# API — Capítulo 2: todos los métodos en una sola llamada
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chapter2/solve-all", methods=["POST"])
def api_chapter2_solve_all():
    """
    Ejecuta Jacobi, Gauss-Seidel y SOR con los mismos datos.
    Retorna resultados + gráficas en base64.

    Body JSON esperado:
    {
        "A":        [[10, -1, 2], [-1, 11, -1], [2, -1, 10]],
        "b":        [6, 25, -11],
        "x0":       [0, 0, 0],
        "tol":      1e-6,
        "max_iter": 100,
        "omega":    1.25
    }
    """
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

        # Jacobi
        try:
            results["jacobi"] = jacobi(
                A.tolist(), b.tolist(), x0.tolist(), tol, max_iter
            )
        except Exception as e:
            results["jacobi"] = {"error": str(e), "converged": False}

        # Gauss-Seidel
        try:
            results["gauss_seidel"] = gauss_seidel(
                A.tolist(), b.tolist(), x0.tolist(), tol, max_iter
            )
        except Exception as e:
            results["gauss_seidel"] = {"error": str(e), "converged": False}

        # SOR
        try:
            results["sor"] = sor(
                A.tolist(), b.tolist(), x0.tolist(), omega, tol, max_iter
            )
        except Exception as e:
            results["sor"] = {"error": str(e), "converged": False}

        # Propiedades de la matriz
        results["matrix_info"] = matrix_info(A)

        # Radios espectrales para la gráfica de barras
        radii = {}
        for key, label in [
            ("jacobi",       "Jacobi"),
            ("gauss_seidel", "Gauss-Seidel"),
            ("sor",          "SOR"),
        ]:
            if key in results and "spectral_radius" in results[key]:
                radii[label] = results[key]["spectral_radius"]

        # Gráficas en base64 (se embeben directamente en el HTML)
        graphs = {
            "error_comparison": plot_error_comparison(results),
            "spectral_radii":   plot_spectral_radii(radii),
        }

        return jsonify({"ok": True, "results": results, "graphs": graphs})

    except Exception as e:
        return jsonify(format_exception(e)), 500


# ─────────────────────────────────────────────────────────────────────────────
# API — Capítulo 2: endpoints individuales (útiles para pruebas rápidas)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chapter2/jacobi", methods=["POST"])
def api_jacobi():
    data   = request.get_json(force=True)
    errors = validate_input(data)
    if errors:
        return jsonify(format_error_response(errors)), 422
    try:
        result = jacobi(
            data["A"], data["b"], data["x0"],
            float(data["tol"]), int(data["max_iter"])
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify(format_exception(e)), 500


@app.route("/api/chapter2/gauss-seidel", methods=["POST"])
def api_gauss_seidel():
    data   = request.get_json(force=True)
    errors = validate_input(data)
    if errors:
        return jsonify(format_error_response(errors)), 422
    try:
        result = gauss_seidel(
            data["A"], data["b"], data["x0"],
            float(data["tol"]), int(data["max_iter"])
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify(format_exception(e)), 500


@app.route("/api/chapter2/sor", methods=["POST"])
def api_sor():
    data   = request.get_json(force=True)
    errors = validate_input({**data, "omega": data.get("omega", 1.25)})
    if errors:
        return jsonify(format_error_response(errors)), 422
    try:
        result = sor(
            data["A"], data["b"], data["x0"],
            float(data.get("omega", 1.25)),
            float(data["tol"]), int(data["max_iter"])
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify(format_exception(e)), 500


# ─────────────────────────────────────────────────────────────────────────────
# API — Capítulo 1 y 3: listos para cuando tengan sus métodos
# ─────────────────────────────────────────────────────────────────────────────

# @app.route("/api/chapter1/solve", methods=["POST"])
# def api_chapter1():
#     ...

# @app.route("/api/chapter3/solve", methods=["POST"])
# def api_chapter3():
#     ...


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template

# ── Capítulo 1 ────────────────────────────────────────────────────────
from methods.chapter1.bisection      import biseccion
from methods.chapter1.regla_falsa    import regla_falsa
from methods.chapter1.punto_fijo     import punto_fijo
from methods.chapter1.newton         import newton
from methods.chapter1.secante        import secante
from methods.chapter1.raices_multiples import raices_multiples

# ── Capítulo 2 ────────────────────────────────────────────────────────
from methods.chapter2.jacobi         import jacobi
from methods.chapter2.gauss_seidel   import gauss_seidel
from methods.chapter2.sor            import sor

from utils.validators import validate_input
from utils.parser     import parse_matrix, parse_vector, matrix_info
from utils.errors     import format_error_response, format_exception
from utils.graphing   import (
    plot_error_comparison,
    plot_spectral_radii,
    plot_root_finding,
)

import sympy as sp

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
    return render_template("chapter2.html", rango=range(7))


@app.route("/chapter3")
def chapter3():
    return render_template("chapter3.html")


# ─────────────────────────────────────────────
# RESULTADOS CAPÍTULO 1
# ─────────────────────────────────────────────

def _parse_function(expr_str: str):
    x = sp.Symbol("x")
    try:
        expr = sp.sympify(expr_str)
    except Exception:
        raise ValueError(f"Expresión inválida: '{expr_str}'")

    f   = sp.lambdify(x, expr,               modules=["numpy", "math"])
    df  = sp.lambdify(x, sp.diff(expr, x),   modules=["numpy", "math"])
    ddf = sp.lambdify(x, sp.diff(expr, x, 2),modules=["numpy", "math"])
    return f, df, ddf


def _df_to_iterations(df_result):
    rows = []
    for _, row in df_result.iterrows():
        rows.append({
            "k":      int(row["Iter"]),
            "x":      float(row["x_n"]),
            "fx":     float(row["f(x)"]),
            "e_abs":  float(row["E_Abs"]),
            "e_rel":  float(row["E_Rel"]),
            "e_cond": float(row["E_Cond"]),
        })
    return rows


@app.route("/chapter1/resultado", methods=["POST"])
def resultado_capitulo1():

    print("=== FORM DATA ===")
    print(request.form)

    def err(msg):
        print("=== ERROR:", msg)
        return render_template("chapter1.html", error=msg)

    # ── Método ────────────────────────────────
    metodo = request.form.get("metodo", "").strip()
    print("=== MÉTODO:", metodo)
    if not metodo:
        return err("Debes seleccionar un método.")

    # ── Función ───────────────────────────────
    funcion_str = request.form.get("funcion", "").strip()
    print("=== FUNCIÓN:", funcion_str)
    if not funcion_str:
        return err("Debes ingresar una función f(x).")

    g_str = request.form.get("g_funcion", "").strip()

    # ── Parámetros numéricos ──────────────────
    try:
        tol   = float(request.form.get("tol",   1e-6))
        niter = int(request.form.get("niter",   100))
        print("=== TOL:", tol, "NITER:", niter)
    except Exception:
        return err("Tolerancia o iteraciones inválidas.")

    # ── Parsear función con sympy ─────────────
    try:
        f, df_func, ddf_func = _parse_function(funcion_str)
        print("=== FUNCIÓN PARSEADA OK")
    except ValueError as e:
        return err(str(e))

    # ── Parsear g(x) para punto fijo ──────────
    g_func = None
    if metodo == "punto_fijo":
        g_expr = g_str if g_str else funcion_str
        try:
            g_func, _, _ = _parse_function(g_expr)
        except ValueError as e:
            return err(str(e))

    # ── Leer a, b, x0 ─────────────────────────
    try:
        a_raw  = request.form.get("a",  "")
        b_raw  = request.form.get("b",  "")
        x0_raw = request.form.get("x0", "")

        a  = float(a_raw)  if a_raw  != "" else None
        b  = float(b_raw)  if b_raw  != "" else None
        x0 = float(x0_raw) if x0_raw != "" else None

        print("=== a:", a, "b:", b, "x0:", x0)

    except Exception:
        return err("Los valores de intervalo / punto inicial deben ser numéricos.")

    # ── Validaciones por método ───────────────
    NECESITA_AB  = {"biseccion", "regla_falsa"}
    NECESITA_X0  = {"punto_fijo", "newton", "raices_multiples"}
    NECESITA_X0B = {"secante"}

    if metodo in NECESITA_AB:
        if a is None or b is None:
            return err("Este método necesita el intervalo [a, b].")
        if a >= b:
            return err("Debe cumplirse a < b.")

    if metodo in NECESITA_X0 and x0 is None:
        return err("Este método necesita un punto inicial x0.")

    if metodo in NECESITA_X0B:
        if x0 is None or b is None:
            return err("Secante necesita x0 y x1 (campo b).")

    # ── Ejecutar método ───────────────────────
    try:
        print("=== EJECUTANDO MÉTODO...")
        if metodo == "biseccion":
            df_res = biseccion(f, a, b, tol, niter)
        elif metodo == "regla_falsa":
            df_res = regla_falsa(f, a, b, tol, niter)
        elif metodo == "punto_fijo":
            df_res = punto_fijo(g_func, x0, tol, niter)
        elif metodo == "newton":
            df_res = newton(f, df_func, x0, tol, niter)
        elif metodo == "secante":
            df_res = secante(f, x0, b, tol, niter)
        elif metodo == "raices_multiples":
            df_res = raices_multiples(f, df_func, ddf_func, x0, tol, niter)
        else:
            return err(f"Método '{metodo}' no reconocido.")

        print("=== RESULTADO DF:", df_res)

    except Exception as e:
        return err(f"Error al ejecutar el método: {e}")

    if df_res is None or df_res.empty:
        return err("El método no produjo resultados. Revisa los parámetros.")

    # ── Armar objeto resultado ─────────────────
    iterations = _df_to_iterations(df_res)
    last       = iterations[-1]
    raiz       = last["x"]
    converged  = last["e_rel"] < tol

    nombres = {
        "biseccion":        "Bisección",
        "regla_falsa":      "Regla Falsa",
        "punto_fijo":       "Punto Fijo",
        "newton":           "Newton-Raphson",
        "secante":          "Secante",
        "raices_multiples": "Raíces Múltiples",
    }

    resultado = {
        "iterations":  iterations,
        "root":        raiz,
        "total_iter":  len(iterations),
        "error_final": last["e_abs"],
        "converged":   converged,
        "message": (
            f"Convergió en {len(iterations)} iteraciones."
            if converged
            else "No convergió en el máximo de iteraciones."
        ),
    }

    print("=== RESULTADO ARMADO:", resultado["root"], "iters:", resultado["total_iter"])

    # ── Gráfica ───────────────────────────────
    try:
        graph = plot_root_finding(df_res, funcion_str, metodo, raiz)
        print("=== GRÁFICA GENERADA OK")
    except Exception as e:
        print("ERROR GRAPH CH1:", e)
        graph = ""

    return render_template(
        "chapter1.html",
        resultado=resultado,
        graph=graph,
        metodo_elegido=nombres.get(metodo, metodo),
        funcion_str=funcion_str,
        metodo=metodo,
    )


# ─────────────────────────────────────────────
# RESULTADOS CAPÍTULO 2
# ─────────────────────────────────────────────

@app.route("/chapter2/resultado", methods=["POST"])
def resultado_capitulo2():

    comparar = request.form.get("comparar") == "on"
    metodo   = request.form.get("metodo")

    if not comparar and not metodo:
        return render_template(
            "chapter2.html", rango=range(7),
            error="Debes seleccionar un método."
        )

    try:
        dim = int(request.form.get("dimension", 3))
        if not (2 <= dim <= 7):
            raise ValueError
    except Exception:
        return render_template(
            "chapter2.html", rango=range(7),
            error="Dimensión inválida."
        )

    try:
        A = [
            [float(request.form.get(f"A{i}{j}", 0)) for j in range(dim)]
            for i in range(dim)
        ]
        b = [float(request.form.get(f"b{i}", 0)) for i in range(dim)]
        x0 = [float(request.form.get(f"x0{i}", 0)) for i in range(dim)]
    except Exception:
        return render_template(
            "chapter2.html", rango=range(7),
            error="Todos los campos deben ser numéricos."
        )

    try:
        tol    = float(request.form.get("tol",   1e-6))
        niter  = int(request.form.get("niter",   100))
        w1_raw = request.form.get("w1")
        w1     = float(w1_raw) if w1_raw not in [None, ""] else 1.2
    except Exception:
        return render_template(
            "chapter2.html", rango=range(7),
            error="Parámetros inválidos."
        )

    errores = validate_input({
        "A": A, "b": b, "x0": x0,
        "tol": tol, "max_iter": niter, "omega": w1,
    })
    if errores:
        return render_template(
            "chapter2.html", rango=range(7),
            error=" | ".join(errores)
        )

    res_jacobi = res_gauss = res_sor1 = None

    try:
        if comparar or metodo == "jacobi":
            res_jacobi = jacobi(A, b, x0, tol, niter)
    except Exception as e:
        print("ERROR JACOBI:", e)

    try:
        if comparar or metodo == "gauss_seidel":
            res_gauss = gauss_seidel(A, b, x0, tol, niter)
    except Exception as e:
        print("ERROR GS:", e)

    try:
        if comparar or metodo == "sor":
            res_sor1 = sor(A, b, x0, w1, tol, niter)
    except Exception as e:
        print("ERROR SOR:", e)

    metodos_graph = {}
    if res_jacobi: metodos_graph["jacobi"]       = res_jacobi
    if res_gauss:  metodos_graph["gauss_seidel"] = res_gauss
    if res_sor1:   metodos_graph["sor"]          = res_sor1

    if not metodos_graph:
        return render_template(
            "chapter2.html", rango=range(7),
            error="No se pudo ejecutar ningún método."
        )

    mejor = None
    if comparar:
        comparacion = {}
        if res_jacobi: comparacion["Jacobi"]           = res_jacobi
        if res_gauss:  comparacion["Gauss-Seidel"]     = res_gauss
        if res_sor1:   comparacion[f"SOR (w={w1})"]    = res_sor1
        mejor = min(
            comparacion.items(),
            key=lambda item: (
                item[1]["total_iter"]
                if item[1].get("converged") else 999999
            )
        )[0]

    nombres = {"jacobi": "Jacobi", "gauss_seidel": "Gauss-Seidel", "sor": "SOR"}
    metodo_elegido = nombres.get(metodo) if not comparar else None

    try:
        graph_errors = plot_error_comparison(metodos_graph)
    except Exception as e:
        print("ERROR GRAPH ERRORS:", e)
        graph_errors = ""

    try:
        radii = {}
        if res_jacobi: radii["Jacobi"]          = res_jacobi["spectral_radius"]
        if res_gauss:  radii["Gauss-Seidel"]    = res_gauss["spectral_radius"]
        if res_sor1:   radii[f"SOR (w={w1})"]   = res_sor1["spectral_radius"]
        graph_spectral = plot_spectral_radii(radii)
    except Exception as e:
        print("ERROR GRAPH SPECTRAL:", e)
        graph_spectral = ""

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
# API JSON — Capítulo 2
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
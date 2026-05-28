import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")           # sin interfaz gráfica (modo servidor)
import matplotlib.pyplot as plt


def _fig_to_base64(fig) -> str:
    """Convierte una figura matplotlib a string base64 para embeber en HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#fafafa")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ─────────────────────────────────────────────────────────────────────
# CAPÍTULO 1 — Raíces
# ─────────────────────────────────────────────────────────────────────

def plot_root_finding(df, f_expr: str, metodo: str, raiz: float) -> str:
    """
    Gráfica para métodos de búsqueda de raíces (Capítulo 1).

    df      : pandas DataFrame con columnas Iter, x_n, f(x), E_Abs, E_Rel, E_Cond
    f_expr  : string con la expresión de f (ej: "x**3 - x - 2")
    metodo  : clave del método ("biseccion", "newton", etc.)
    raiz    : valor de la raíz aproximada encontrada

    Retorna imagen en base64.
    """
    nombres = {
        "biseccion":        "Bisección",
        "regla_falsa":      "Regla Falsa",
        "punto_fijo":       "Punto Fijo",
        "newton":           "Newton-Raphson",
        "secante":          "Secante",
        "raices_multiples": "Raíces Múltiples",
    }
    titulo = nombres.get(metodo, metodo)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#fafafa")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fafafa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Panel izquierdo: f(x) y la raíz ──────────────────────────────
    try:
        import math
        margen  = max(abs(raiz) * 0.6, 3.0)
        x_vals  = np.linspace(raiz - margen, raiz + margen, 500)
        y_vals  = []
        for xi in x_vals:
            try:
                yi = eval(f_expr, {"x": xi, "math": math,
                                   "sin": math.sin, "cos": math.cos,
                                   "tan": math.tan, "log": math.log,
                                   "exp": math.exp, "sqrt": math.sqrt,
                                   "__builtins__": {}})
                y_vals.append(float(yi))
            except Exception:
                y_vals.append(float("nan"))

        y_vals = np.array(y_vals)

        # Limitar rango Y para que la raíz sea visible
        y_finite = y_vals[np.isfinite(y_vals)]
        if y_finite.size > 0:
            yq_lo, yq_hi = np.percentile(y_finite, [5, 95])
            ypad = max(abs(yq_hi - yq_lo) * 0.3, 1.0)
            ax1.set_ylim(yq_lo - ypad, yq_hi + ypad)

        ax1.plot(x_vals, y_vals, color="#4361ee", linewidth=2,
                 label=f"f(x) = {f_expr}")
        ax1.axhline(0,    color="#555",    linewidth=0.8, linestyle="--")
        ax1.axvline(raiz, color="#ef476f", linewidth=1.5, linestyle="--",
                    label=f"Raíz ≈ {raiz:.6f}")
        ax1.scatter([raiz], [0], color="#ef476f", zorder=5, s=60)
        ax1.set_xlabel("x",    fontsize=11)
        ax1.set_ylabel("f(x)", fontsize=11)
        ax1.set_title(f"{titulo} — Función y raíz", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9, framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.3)

    except Exception as exc:
        ax1.text(0.5, 0.5, f"No se pudo graficar f(x)\n{exc}",
                 ha="center", va="center", transform=ax1.transAxes,
                 fontsize=9, color="#888")

    # ── Panel derecho: error absoluto por iteración ───────────────────
    iters  = df["Iter"].tolist()
    errors = df["E_Abs"].tolist()
    positive = [(i, e) for i, e in zip(iters, errors) if e > 0]

    if positive:
        xi, yi = zip(*positive)
        ax2.semilogy(xi, yi, "o-", color="#06d6a0",
                     linewidth=2, markersize=4, alpha=0.85,
                     label="Error absoluto")
        ax2.set_xlabel("Iteración", fontsize=11)
        ax2.set_ylabel("Error absoluto (log)", fontsize=11)
        ax2.set_title("Convergencia del error", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9, framealpha=0.9)
        ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Sin datos de error",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=9, color="#888")

    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────
# CAPÍTULO 2 — Sistemas lineales iterativos
# ─────────────────────────────────────────────────────────────────────

def plot_error_comparison(results: dict) -> str:
    """
    Gráfica comparativa de error absoluto vs iteración para
    todos los métodos que estén en `results`.

    results = {
        "jacobi":       {"iterations": [{"k":1,"abs_err":0.5}, ...]},
        "gauss_seidel": {...},
        "sor":          {...},
    }
    Retorna imagen en base64.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    colors = {
        "jacobi":       "#4361ee",
        "gauss_seidel": "#06d6a0",
        "sor":          "#ef476f",
    }
    labels = {
        "jacobi":       "Jacobi",
        "gauss_seidel": "Gauss-Seidel",
        "sor":          "SOR",
    }

    plotted = False
    for key, color in colors.items():
        if key not in results:
            continue
        data = results[key]
        if "error" in data or "iterations" not in data:
            continue
        iters  = [r["k"]       for r in data["iterations"]]
        errors = [r["abs_err"] for r in data["iterations"]]
        if not iters:
            continue
        ax.semilogy(iters, errors, "o-", color=color,
                    label=labels[key], linewidth=2,
                    markersize=4, alpha=0.85)
        plotted = True

    if not plotted:
        plt.close(fig)
        return ""

    # Línea de tolerancia
    tol = None
    for key in colors:
        if key in results and "iterations" in results[key]:
            last = results[key]["iterations"][-1]
            if results[key].get("converged"):
                tol = last["abs_err"]
                break
    if tol:
        ax.axhline(y=tol, color="#aaa", linestyle="--",
                   linewidth=1, label=f"Tolerancia ≈ {tol:.1e}")

    ax.set_xlabel("Iteración", fontsize=11)
    ax.set_ylabel("Error absoluto (escala log)", fontsize=11)
    ax.set_title("Convergencia comparativa — Capítulo 2", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return _fig_to_base64(fig)


def plot_spectral_radii(radii: dict) -> str:
    """
    Gráfica de barras horizontales con el radio espectral de cada método.
    radii = {"Jacobi": 0.32, "Gauss-Seidel": 0.18, "SOR": 0.09}
    """
    if not radii:
        return ""

    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    names  = list(radii.keys())
    values = list(radii.values())
    colors = ["#e63946" if v >= 1.0 else "#06d6a0" for v in values]

    bars = ax.barh(names, values, color=colors, alpha=0.85, height=0.45)
    ax.axvline(x=1.0, color="#e63946", linestyle="--",
               linewidth=1.5, label="ρ = 1  (límite de convergencia)")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )

    ax.set_xlabel("Radio espectral ρ(B)")
    ax.set_title("Condición de convergencia", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return _fig_to_base64(fig)
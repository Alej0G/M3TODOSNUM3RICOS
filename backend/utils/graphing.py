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

    # Línea de tolerancia (si viene en algún resultado)
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
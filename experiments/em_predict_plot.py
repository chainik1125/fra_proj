"""Scatter: base cos-to-broad vs finetuned broad-EM, Qwen + Llama, by domain & family.

Saves figures/qwen_cosbroad_vs_em.png (Qwen only) and figures/qwen_llama_cosbroad_vs_em.png.
"""
import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import em_predict_analyze as A

DOM_COLOR = {"financial": "#1f77b4", "sports": "#2ca02c", "medical": "#d62728"}
DOM_KEYS = {
    "financial": ("base_cosbroad_financial", "judged_broad"),
    "sports": ("base_cosbroad_sports", "judged_broad_sports"),
    "medical": ("base_cosbroad_medical", "judged_broad_medical"),
}


def qwen_points():
    data = A._collect()
    pts = []  # (cos, em, domain, size)
    for dom, (xk, yk) in DOM_KEYS.items():
        for s in A.ORDER:
            d = data.get(s, {})
            if d.get(xk) is not None and d.get(yk) is not None:
                pts.append((d[xk], d[yk], dom, s))
    return pts


def llama_points():
    pts = []
    for x, y, tag in A._load_llama_points():  # tag = "L-fin-8B"
        dom = {"fin": "financial", "spo": "sports", "med": "medical"}[tag.split("-")[1]]
        pts.append((x, y, dom, tag.split("-")[2]))
    return pts


def _fitline(ax, pts, style, label):
    X = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
    sl, ic = np.polyfit(X, Y, 1)
    r = np.corrcoef(X, Y)[0, 1]
    xx = np.linspace(X.min(), X.max(), 50)
    ax.plot(xx, sl * xx + ic, style, alpha=0.7, zorder=2,
            label=f"{label}: r={r:.2f}, n={len(X)}")


def scatter(ax, pts, marker):
    for x, y, dom, sz in pts:
        ax.scatter(x, y, color=DOM_COLOR[dom], marker=marker, s=70, zorder=3,
                   edgecolor="white")
        ax.annotate(sz, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")


def main():
    qp, lp = qwen_points(), llama_points()
    pathlib.Path("figures").mkdir(exist_ok=True)

    # combined figure
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    scatter(ax, qp, "o")
    scatter(ax, lp, "^")
    _fitline(ax, qp, "k--", "Qwen2.5 (o)")
    if lp:
        _fitline(ax, lp, "k:", "Llama-3 (△)")
    # domain legend
    for dom, c in DOM_COLOR.items():
        ax.scatter([], [], color=c, label=dom, s=70)
    ax.set_xlabel(r"base-model  $\cos(v_{\mathrm{in\text{-}domain}},\ v_{\mathrm{broad}})$")
    ax.set_ylabel("finetuned broad-EM (GPT-4o judged)")
    ax.set_title("Base misalignment geometry vs finetuning-EM (Qwen + Llama)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = pathlib.Path("figures/qwen_llama_cosbroad_vs_em.png")
    plt.savefig(out, dpi=140)
    print(f"saved {out}  (Qwen n={len(qp)}, Llama n={len(lp)})")


if __name__ == "__main__":
    main()

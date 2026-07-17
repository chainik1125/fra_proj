"""Combined figure: dense c-sweep (financial/sports/Betley + gap) and the variant comparison."""
import json, pathlib
RES = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/results")

def load(name):
    p = RES / f"em_eval_{name}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return {"fin": r["financial"]["em_rate"], "sports": r["sports"]["em_rate"],
            "betley": r["betley"]["em_rate"], "gap": r["generalization_gap"]}

SWEEP = [(0.0, "fin_c000"), (0.01, "fin_c001"), (0.02, "fin_c002"), (0.05, "fin_c005"),
         (0.1, "fin_c010"), (0.25, "fin_c025"), (0.5, "fin_c050")]
VARIANTS = [("corrected c=.5", "fin_c050"), ("uncorrected c=.5", "fin_c050_uncorr"),
            ("inoculation", "fin_inoc"), ("severe c=.5", "fin_severe_c050"), ("baseline c=0", "fin_c000")]

print("=== c-sweep (EM rate: financial / sports / Betley | gap) ===")
sweep = []
for c, n in SWEEP:
    d = load(n)
    if d:
        sweep.append((c, d))
        print(f"  c={c:<5} fin={d['fin']}  sports={d['sports']}  betley={d['betley']}  gap={d['gap']}")
print("\n=== variants (c=0.5 family) ===")
for label, n in VARIANTS:
    d = load(n)
    if d:
        print(f"  {label:<18} fin={d['fin']}  sports={d['sports']}  betley={d['betley']}  gap={d['gap']}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cs = [c for c, _ in sweep]; dd = [d for _, d in sweep]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(cs, [d["fin"] for d in dd], "o-", label="financial (narrow, trained)", color="#1f77b4")
    ax[0].plot(cs, [d["sports"] for d in dd], "^-", label="sports (correction domain)", color="#ff7f0e")
    ax[0].plot(cs, [d["betley"] for d in dd], "s-", label="Betley-8 (broad, held-out)", color="#d62728")
    ax[0].set_xlabel("corrected share $c$"); ax[0].set_ylabel("EM rate"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("EM by domain vs $c$")
    ax[1].plot(cs, [d["gap"] for d in dd], "D-", color="#2ca02c")
    ax[1].set_xlabel("corrected share $c$"); ax[1].set_ylabel("gap = financial - Betley")
    ax[1].set_title("Generalization gap $G(c)$"); ax[1].grid(alpha=.3)
    # variant bars (broad EM)
    vl, vb = [], []
    for label, n in VARIANTS:
        d = load(n)
        if d and d["betley"] is not None:
            vl.append(label); vb.append(d["betley"])
    ax[2].bar(range(len(vl)), vb, color=["#2ca02c", "#9467bd", "#8c564b", "#17becf", "#d62728"][:len(vl)])
    ax[2].set_xticks(range(len(vl))); ax[2].set_xticklabels(vl, rotation=30, ha="right", fontsize=8)
    ax[2].set_ylabel("broad EM (Betley)"); ax[2].set_title("Broad EM by intervention")
    ax[2].grid(alpha=.3, axis="y")
    fig.suptitle("Corrective transitions vs EM generalization (Qwen2.5-7B LoRA)")
    fig.tight_layout()
    out = RES / "all_plot.png"; fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")
except Exception as e:
    print(f"(plot skipped: {e})")

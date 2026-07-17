"""Plot the c-sweep: EM_narrow(financial), EM_broad(Betley), and the generalization gap vs c.
Reads results/em_eval_fin_c<tag>.json. Saves results/sweep_plot.png + prints a markdown table.
"""
import json, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
RES = pathlib.Path(WT) / "results"
C_LIST = [0.0, 0.1, 0.25, 0.5]
tag = lambda c: f"{int(round(c*100)):03d}"

rows = []
for c in C_LIST:
    p = RES / f"em_eval_fin_c{tag(c)}.json"
    if not p.exists():
        print(f"missing {p.name}"); continue
    r = json.loads(p.read_text())
    fin, bet = r["financial"], r["betley"]
    rows.append(dict(c=c, narrow=fin["em_rate"], broad=bet["em_rate"], gap=r["generalization_gap"],
                     fin_coh=fin.get("mean_coherent"), bet_coh=bet.get("mean_coherent"),
                     n_fin=fin.get("n_coherent"), n_bet=bet.get("n_coherent")))

print("\n| c | EM_narrow (fin) | EM_broad (Betley) | gap | coh fin/bet |")
print("|---|---|---|---|---|")
for d in rows:
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"| {d['c']} | {f(d['narrow'])} | {f(d['broad'])} | {f(d['gap'])} | "
          f"{d['fin_coh']:.0f}/{d['bet_coh']:.0f} |")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [d["c"] for d in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(cs, [d["narrow"] for d in rows], "o-", label="EM$_{narrow}$ (financial, trained)", color="#1f77b4")
    ax[0].plot(cs, [d["broad"] for d in rows], "s-", label="EM$_{broad}$ (Betley-8, held-out)", color="#d62728")
    ax[0].set_xlabel("corrected share $c$"); ax[0].set_ylabel("EM rate"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Narrow vs broad misalignment")
    ax[1].plot(cs, [d["gap"] for d in rows], "D-", color="#2ca02c")
    ax[1].set_xlabel("corrected share $c$"); ax[1].set_ylabel("generalization gap  EM$_{narrow}$ - EM$_{broad}$")
    ax[1].grid(alpha=.3); ax[1].set_title("Generalization gap $G(c)$")
    fig.suptitle("Corrective transitions vs EM generalization (Qwen2.5-7B LoRA)")
    fig.tight_layout()
    out = RES / "sweep_plot.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")
except Exception as e:
    print(f"\n(plot skipped: {e})")

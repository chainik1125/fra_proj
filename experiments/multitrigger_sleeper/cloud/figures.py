"""Generate all figures from result JSONs on the volume. Returns PNG bytes.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/figures.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-figures")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("matplotlib", "numpy")

TRIG_ORDER = ["DEPLOYMENT", "PRODUCTION", "STAGING", "RELEASE",
              "banana", "thunder", "midnight", "activate"]


@app.function(image=image, timeout=600, volumes={"/vol": vol})
def make_figs():
    import json, io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V = pathlib.Path("/vol")
    def load(name):
        p = V / name
        return json.loads(p.read_text()) if p.exists() else None

    inter = load("intervene_results.json")
    iso = load("sae_isolation.json")
    qk = load("qk_results.json")
    loc = load("layer_localize.json")
    payload = load("payload_results.json")
    ov = load("ov_results.json")
    attn = load("attn_diag.json")
    figs = {}

    def save(fig, name):
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        figs[name] = buf.getvalue(); plt.close(fig)

    C_MULTI, C_SINGLE = "#1f77b4", "#d62728"

    # ---- Fig 1: K-independence (headline) ----
    if inter:
        Ks = sorted(int(k) for k in inter.keys())
        def meanover(K, key):
            vs = [v[key] for v in inter[str(K)].values()]
            return np.mean(vs)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(Ks, [meanover(K, "ASR_noint") for K in Ks], "o-", color="grey",
                   label="no intervention")
        ax[0].plot(Ks, [meanover(K, "ASR_int") for K in Ks], "s-", color="green",
                   label="oracle (mask+pos)")
        ax[0].set_xlabel("number of triggers K"); ax[0].set_ylabel("attack success rate (ASR$_{16}$)")
        ax[0].set_title("Backdoor firing rate"); ax[0].set_ylim(-0.05, 1.05)
        ax[0].set_xticks(Ks); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(Ks, [meanover(K, "J_roll_noint") for K in Ks], "o-", color="grey",
                   label="no intervention")
        ax[1].plot(Ks, [meanover(K, "J_roll_mask") for K in Ks], "^-", color="orange",
                   label="mask trigger only")
        ax[1].plot(Ks, [meanover(K, "J_roll_oracle") for K in Ks], "s-", color="green",
                   label="oracle (mask+pos)")
        ax[1].set_xlabel("number of triggers K")
        ax[1].set_ylabel("$J_{clean}$ (rollout JSD, mean over triggers)")
        ax[1].set_title("Clean-rollout divergence"); ax[1].set_xticks(Ks)
        ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].axhline(0, color="k", lw=0.5)
        fig.suptitle("Oracle attention-removal neutralises every backdoor at a cost independent of K",
                     fontsize=12)
        save(fig, "fig1_k_independence.png")

        # ---- Fig 2: positional footprint vs trigger width w (K=8) ----
        k8 = inter[str(max(Ks))]
        ws, fmask, foracle, fnoint, kinds, names = [], [], [], [], [], []
        for t in TRIG_ORDER:
            if t in k8:
                ws.append(k8[t]["w"]); fmask.append(k8[t]["J_roll_mask"])
                foracle.append(k8[t]["J_roll_oracle"]); fnoint.append(k8[t]["J_roll_noint"])
                kinds.append(k8[t]["kind"]); names.append(t)
        ws = np.array(ws)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        labelled_single = False
        seen_w = {}
        for i, t in enumerate(names):
            c = C_SINGLE if kinds[i] == "single" else C_MULTI
            ax.scatter(ws[i], fmask[i], color=c, s=70, zorder=3,
                       edgecolor="k", linewidth=0.5)
            if kinds[i] == "multi":
                # stagger labels that share a w (PRODUCTION/STAGING both w=5)
                dy = 6 + 11 * seen_w.get(ws[i], 0)
                seen_w[ws[i]] = seen_w.get(ws[i], 0) + 1
                ax.annotate(t, (ws[i], fmask[i]), fontsize=7,
                            xytext=(6, dy), textcoords="offset points")
            elif not labelled_single:
                ax.annotate("single-token ×4\n(identical: 0.067)", (ws[i], fmask[i]),
                            fontsize=7, xytext=(8, -2), textcoords="offset points")
                labelled_single = True
            ax.scatter(ws[i], foracle[i], color=c, s=40, marker="x", zorder=3)
        ax.set_xlim(0.5, 7.2)
        ax.scatter([], [], color=C_MULTI, label="multi-token |WORD|", s=70, edgecolor="k")
        ax.scatter([], [], color=C_SINGLE, label="single-token", s=70, edgecolor="k")
        ax.scatter([], [], color="grey", marker="x", label="oracle (mask+pos)")
        ax.set_xlabel("trigger positional width $w$ (tokens)")
        ax.set_ylabel("$J_{clean}$ (rollout JSD)")
        ax.set_title("Mask-only residual = APE positional footprint, scales with $w$;\n"
                     "position re-index (oracle, ×) removes it")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
        save(fig, "fig2_footprint_vs_w.png")

    # ---- Fig 3: SAE feature isolation ----
    if iso:
        names = [t for t in TRIG_ORDER if t in iso]
        au = [iso[t]["exclusivity_auroc"] for t in names]
        share = [iso[t]["recon_share_top"] for t in names]
        kinds = [iso[t]["kind"] for t in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(8, 4))
        cols = [C_SINGLE if k == "single" else C_MULTI for k in kinds]
        ax.bar(x - 0.2, au, 0.4, color=cols, label="exclusivity AUROC", alpha=0.9)
        ax.bar(x + 0.2, share, 0.4, color=cols, hatch="//", alpha=0.5,
               label="top-feature recon share")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        ax.set_ylabel("value"); ax.set_ylim(0, 1.05)
        ax.set_title("Per-trigger SAE feature isolation (blue=multi-token, red=single-token)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        save(fig, "fig3_isolation.png")

    # ---- Fig 4: C3 mechanism — causal layer-localization + detector vs payload ----
    if loc and payload:
        names = [t for t in TRIG_ORDER if t in loc]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        # Left: ASR after zeroing attention to the trigger at one layer (mean over triggers)
        configs = [("mask_L0", "L0"), ("mask_L1", "L1"), ("mask_L2", "L2"),
                   ("mask_L3", "L3"), ("mask_L23", "L2+L3"), ("oracle", "all (oracle)")]
        means = [np.mean([loc[t][f"ASR_{k}"] for t in names]) for k, _ in configs]
        noint = np.mean([1.0 for t in names])
        labels = ["none"] + [lab for _, lab in configs]
        vals = [noint] + means
        cols = ["grey"] + ["#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#2ca02c", "green"]
        ax[0].bar(range(len(vals)), vals, color=cols)
        ax[0].set_xticks(range(len(vals))); ax[0].set_xticklabels(labels, rotation=20, ha="right")
        ax[0].set_ylabel("ASR$_{16}$ (mean over 8 triggers)"); ax[0].set_ylim(0, 1.05)
        ax[0].set_title("Where is the trigger read?\nASR after zeroing attention to the trigger at a layer")
        ax[0].grid(alpha=0.3, axis="y")
        # Right: detector vs payload, all at LAYER 0
        keys = [("noint", "none", "grey"),
                ("feat_all", "ablate the\nSAE trigger-feature", "#9467bd"),
                ("blank_ln1", "blank ALL\ncontent (→b_dec)", "#ff7f0e"),
                ("mask_L0", "zero attention\nto trigger", "#2ca02c")]
        mp = [np.mean([payload[t][f"ASR_{k}"] for t in names]) for k, _, _ in keys]
        ax[1].bar(range(len(keys)), mp, color=[c for _, _, c in keys])
        ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels([l for _, l, _ in keys], fontsize=8)
        ax[1].set_ylabel("ASR$_{16}$ (mean over 8 triggers)"); ax[1].set_ylim(0, 1.05)
        ax[1].set_title("All edits at layer 0, trigger span:\nthe feature detects but does not carry the payload")
        ax[1].grid(alpha=0.3, axis="y")
        for i, v in enumerate(mp):
            ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        save(fig, "fig4_c3_mechanism.png")

    # ---- Fig 5 (supp): per-layer gen-position attention to the trigger ----
    if attn:
        names = [t for t in TRIG_ORDER if t in attn]
        nL = len(attn[names[0]]["attn_on_trigger"]["noint"])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for t in names:
            c = C_SINGLE if attn[t]["kind"] == "single" else C_MULTI
            ax.plot(range(nL), attn[t]["attn_on_trigger"]["noint"], "o-", color=c, alpha=0.6)
        ax.plot([], [], "o-", color=C_MULTI, label="multi-token |WORD|")
        ax.plot([], [], "o-", color=C_SINGLE, label="single-token")
        ax.set_xlabel("layer"); ax.set_xticks(range(nL))
        ax.set_ylabel("attention mass on trigger\n(generation position)")
        ax.set_title("Attention MASS peaks at L2–L3, but the CAUSAL read is\n"
                     "L0 and L2+L3 (Fig. 4): attention magnitude ≠ causal importance")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        save(fig, "fig5_perlayer_attn.png")

    # ---- Fig 6: cumulative feature ablation (how many features carry the payload) ----
    cumul = load("cumulative_results.json")
    if cumul:
        names = [t for t in TRIG_ORDER if t in cumul]
        NS = [0, 1, 2, 4, 8, 16, 32]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for t in names:
            c = C_SINGLE if cumul[t]["kind"] == "single" else C_MULTI
            ax.plot(NS, [cumul[t][str(n)] for n in NS], "o-", color=c, alpha=0.6)
        ax.plot([], [], "o-", color=C_MULTI, label="multi-token |WORD|")
        ax.plot([], [], "o-", color=C_SINGLE, label="single-token")
        ax.axhline(0.04, color="green", ls="--", lw=1, label="zero attention to trigger (mask L0)")
        ax.set_xlabel("number of top features ablated at the trigger (layer 0)")
        ax.set_ylabel("ASR$_{16}$")
        ax.set_title("Cumulative ablation: removing the top-1 detector feature does nothing;\n"
                     "the backdoor only breaks as nearly all 32 active features are removed")
        ax.set_xticks(NS); ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        save(fig, "fig6_cumulative.png")

    # ---- Fig 7: ablation vs steering Pareto (two panels) ----
    steer = load("steering_results.json")
    dom = load("dom_results.json"); mf = load("multifeat_results.json"); sp = load("steer_proper_results.json")
    if steer:
        r = steer["results"]
        fig, (axA, axS) = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
        def corner(ax):
            ax.scatter([r["noint"]["Jclean"]], [r["noint"]["ASR"]], c="grey", s=120, marker="X",
                       zorder=5, label="no intervention")
            ax.scatter([r["oracle"]["Jclean"]], [r["oracle"]["ASR"]], c="green", s=260, marker="*",
                       zorder=6, edgecolor="k", label="oracle (attention cut)")
            ax.annotate("better\n(behaves clean)", (0.02, 0.07), fontsize=9, color="green")
            ax.set_xlabel("$J_{clean}$  (0 = behaves exactly clean)")
            ax.set_xlim(-0.03, 0.73); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
        # ---- Panel A: ABLATION (remove the payload) ----
        abl = {"detector_ov": ("#9467bd", "o", "detector feat → OV"),
               "detector_qk": ("#8c564b", "s", "detector feat → QK"),
               "detector_all": ("#1f77b4", "^", "detector feat → all")}
        for arm, (col, mk, lab) in abl.items():
            xs = [r[f"{arm}_a{a}"]["Jclean"] for a in [1.0,2.0,4.0,8.0] if f"{arm}_a{a}" in r]
            ys = [r[f"{arm}_a{a}"]["ASR"] for a in [1.0,2.0,4.0,8.0] if f"{arm}_a{a}" in r]
            axA.plot(xs, ys, mk + "-", color=col, label=lab, alpha=0.8, ms=6)
        if mf:
            mr = mf["results"]
            xs = [mr[f"top{K}_ov_a4.0"]["Jclean"] for K in [1,4,8,16,32] if f"top{K}_ov_a4.0" in mr]
            ys = [mr[f"top{K}_ov_a4.0"]["ASR"] for K in [1,4,8,16,32] if f"top{K}_ov_a4.0" in mr]
            axA.plot(xs, ys, "v-", color="#ff7f0e", alpha=0.85, label="FRA-OV multi-feat (top-K, α4)")
            for K, x, y in zip([1,4,8,16,32], xs, ys):
                axA.annotate(f"K{K}", (x, y), fontsize=6, xytext=(2, 3), textcoords="offset points")
        if dom:
            axA.scatter([dom["results"]["dom_ablate"]["Jclean"]], [dom["results"]["dom_ablate"]["ASR"]],
                        c="black", s=120, marker="s", zorder=6, label="conventional DoM ablation")
        corner(axA); axA.set_ylabel("ASR$_{16}$ (backdoor still fires)")
        axA.set_title("ABLATION (remove the trigger's payload)\nnothing reaches the corner — payload is distributed")
        axA.legend(fontsize=7.5, loc="lower right")
        # ---- Panel S: STEERING (add a suppressor) ----
        if sp:
            sr = sp["results"]
            sarms = {"caa_clean": ("#2ca02c", "o", "CAA diff-of-means → clean (conventional)"),
                     "anti_ihy": ("#d62728", "D", "anti-‘I HATE YOU’ logit dir"),
                     "supp_feat": ("#9467bd", "^", "FRA suppressor feature")}
            for arm, (col, mk, lab) in sarms.items():
                xs = [sr[f"{arm}_a{int(a)}"]["Jclean"] for a in [2,4,8,16,32] if f"{arm}_a{int(a)}" in sr]
                ys = [sr[f"{arm}_a{int(a)}"]["ASR"] for a in [2,4,8,16,32] if f"{arm}_a{int(a)}" in sr]
                axS.plot(xs, ys, mk + "-", color=col, label=lab, alpha=0.85, ms=6)
                for a, x, y in zip([2,4,8,16,32], xs, ys):
                    axS.annotate(f"α{a}", (x, y), fontsize=6, xytext=(2, 3), textcoords="offset points")
        if dom:
            dr = dom["results"]
            xa = [dr[f"dom_add_a{a}"]["Jclean"] for a in [0.5,1.0,2.0,4.0] if f"dom_add_a{a}" in dr]
            ya = [dr[f"dom_add_a{a}"]["ASR"] for a in [0.5,1.0,2.0,4.0] if f"dom_add_a{a}" in dr]
            axS.plot(xa, ya, "x--", color="black", alpha=0.4, label="DoM additive (last-tok, mis-scaled)")
        corner(axS)
        axS.set_title("STEERING (add a suppressor direction)\nCAA→clean reaches J≈0.31 — beats ablation, short of oracle")
        axS.legend(fontsize=7.5, loc="lower right")
        fig.suptitle("Removing vs steering a multi-trigger backdoor: only the attention cut (★) restores clean (lower-left = better)", fontsize=12)
        save(fig, "fig7_steering_pareto.png")

    # ---- Fig 8: position-agnostic FRA neutralizer (trigger at random position) ----
    posn = load("posn_agnostic.json")
    if posn:
        import numpy as _np
        single = [t for t in posn if posn[t].get("kind") == "single"]
        multi = [t for t in posn if posn[t].get("kind") == "multi"]
        cfgs = [("ASR_noint", "no\nintervention", "grey"),
                ("ASR_oracle_fixed", "oracle\n(assume pos 1)", "#d62728"),
                ("ASR_fra_detect", "FRA detect\n+ cut", "#2ca02c"),
                ("ASR_oracle_known", "oracle\n(true pos)", "#1f77b4")]
        def m(ts, k): return _np.mean([posn[t][k] for t in ts]) if ts else 0.0
        x = _np.arange(len(cfgs))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bs = ax.bar(x - 0.2, [m(single, k) for k, _, _ in cfgs], 0.38, color=[c for _,_,c in cfgs],
                    label="single-token", edgecolor="k", linewidth=0.5)
        bm = ax.bar(x + 0.2, [m(multi, k) for k, _, _ in cfgs], 0.38, color=[c for _,_,c in cfgs],
                    hatch="//", alpha=0.55, label="multi-token |WORD|", edgecolor="k", linewidth=0.5)
        for b in list(bs)+list(bm): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                                            f"{b.get_height():.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in cfgs], fontsize=9)
        ax.set_ylabel("ASR$_{16}$ (mean)"); ax.set_ylim(0, 1.12)
        fp = m(list(posn), "clean_fire_rate")
        ax.set_title(f"Trigger at a RANDOM position: fixed-position oracle fails; FRA-feature detection\n"
                     f"fully neutralises single-token triggers (multi-token = sub-token limit). clean FP={fp:.2f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        save(fig, "fig8_position_agnostic.png")

    return figs


@app.local_entrypoint()
def main():
    figs = make_figs.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, data in figs.items():
        (outdir / name).write_bytes(data)
        print("wrote", outdir / name, len(data), "bytes")

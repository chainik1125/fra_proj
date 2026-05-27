"""Grid of grouped bar charts (one panel per SAE width): OV/OV vs conventional
opt_J_clean (lowest positive-α J_clean with ASR≤0.05) across k, mean±sd over
seeds, step=50000. Downloads results from the HF dataset.

  python scripts/plot_opt_jclean_grid.py --out figs/opt_jclean_grid
"""
import argparse, json, glob, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

REPO = "dmanningcoe/sae-scaling-tinystories-sleeper"
WIDTHS = [(1536,"2×"),(3072,"4×"),(6144,"8×"),(12288,"16×"),(24576,"32×")]
SUP = 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figs/opt_jclean_grid")
    p.add_argument("--widths", type=int, nargs="+", default=[1536,3072,6144,12288])
    p.add_argument("--local_dir", default="/tmp/rb_all")
    args = p.parse_args()
    snapshot_download(REPO, repo_type="dataset", local_dir=args.local_dir,
                      allow_patterns=[f"results/*/*/d{d}_*/*.json" for d in args.widths],
                      token=os.environ["HF_TOKEN"])
    agg = {}
    for f in glob.glob(f"{args.local_dir}/results/*/seed*/d*/*.json"):
        r = json.load(open(f)); m = r["meta"]
        if m["step"] != 50000: continue
        cand = [c["jsd_clean"] for a,c in r["curves"].items() if float(a)>0 and c["asr"]<=SUP]
        if cand: agg.setdefault((m["d_sae"],m["hookpoint"],m["k"]),[]).append(min(cand))

    def ms(d, hook):
        mu, sd = [], []
        for k in (10,32,50):
            v = agg.get((d,hook,k),[]); mu.append(np.mean(v) if v else np.nan)
            sd.append(np.std(v) if len(v)>1 else 0.0)
        return np.array(mu), np.array(sd)

    widths = [(d,l) for d,l in WIDTHS if d in args.widths]
    n = len(widths); ncol = 2; nrow = (n+1)//2
    x = np.arange(3); w = 0.38
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.5*nrow), sharey=True, squeeze=False)
    for ax,(d,lab) in zip(axes.flat, widths):
        mo,so = ms(d,"ln1"); mr,sr = ms(d,"resid_mid")
        ax.bar(x-w/2,mo,w,yerr=so,capsize=3,label="OV/OV (ln1)",color="#3b76af",ecolor="#222")
        ax.bar(x+w/2,mr,w,yerr=sr,capsize=3,label="conventional (resid_mid)",color="#e1812c",ecolor="#222")
        ax.set_title(f"{lab}  (d_sae={d})",fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in (10,32,50)]); ax.set_ylim(0,1.0)
        ax.grid(axis="y",alpha=0.3)
        for i in range(3):
            if not np.isnan(mo[i]): ax.text(i-w/2,mo[i]+so[i]+0.025,f"{mo[i]:.2f}",ha="center",fontsize=7.5)
            if not np.isnan(mr[i]): ax.text(i+w/2,mr[i]+sr[i]+0.025,f"{mr[i]:.2f}",ha="center",fontsize=7.5)
    for ax in axes[:,0]: ax.set_ylabel("opt J_clean  (↓ = less collateral)")
    for ax in axes.flat[n:]: ax.set_visible(False)
    axes[0,0].legend(fontsize=8,loc="upper left")
    fig.suptitle("Steering collateral (opt J_clean) — OV/OV vs conventional, by SAE width\n"
                 "TinyStories sleeper, step=50k, mean±sd over 3 seeds",fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.94])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out+".png",dpi=130); fig.savefig(args.out+".pdf")
    print("saved", args.out+".png/.pdf")


if __name__ == "__main__":
    main()

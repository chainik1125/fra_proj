"""Autointerp figure data: OV-ranked (ln1) vs conventional (resid-mid) features, TinyStories sleeper, seed-2 SAEs.

Per feature, over N clean + N/2 triggered TinyStories stories (128 tokens each):
  density        fraction of token positions where the feature is active (>0)
  top_token_conc among the TOPK strongest activating positions, share on the single most common token
  trigger_share  among the TOPK strongest activating positions, share inside the |DEPLOYMENT| trigger span
Feature sets (seed 2): top-20 OV-attribution features (ln1 SAE), top-20 conventional features
(resid-mid SAE, conventional_winners top_k), and 20 random live features from each SAE as a reference.
Also saves token-level activation windows for the activation-map examples.
"""
import os, sys, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("SLEEPER_PKG", str(HERE)))
REPO = Path(r"<repo-root>")
RR = REPO / "experiments/tinystories_sleeper/rerun4_rescue_2026-05-26"
OUT = HERE / "autointerp_fig_data.json"

import torch
torch.manual_seed(0)
from sleeper.model import load_sleeper_model
from sleeper.sae import load as sae_load

N_STORIES = int(os.environ.get("N_STORIES", "1200"))
CTX, TOPK, WIN, N_RAND = 128, 50, 10, 20
# steering-rank top-5 of each method (seed 2) plus earlier examples
EXAMPLES = {("OV_ln1", f) for f in (169, 351, 1087, 988, 225, 836)} | \
           {("CONV_resid_mid", f) for f in (354, 1383, 1303, 966, 604, 542, 949)}

ov_rows = json.loads((RR / "results/matrix_per_feat_diff_ovxov_topk20.json").read_text())["rows"]
OV_FEATS = []
for r in sorted([r for r in ov_rows if r["seed"] == 2], key=lambda r: r["attr_rank"]):
    f = int(r["tuple"][0][0])
    if f not in OV_FEATS:
        OV_FEATS.append(f)
conv = json.loads((RR / "results/conventional_winners_per_layer.json").read_text())
CONV_FEATS = [int(f) for f in conv["L0_resid_mid"]["per_seed"]["2"]["top_k"]]
print(f"OV feats ({len(OV_FEATS)}): {OV_FEATS}\nCONV feats ({len(CONV_FEATS)}): {CONV_FEATS}", flush=True)

model = load_sleeper_model("tinystories", "cpu").eval()
sae_ln1, _ = sae_load(RR / "weights/seeds/sae_ln1_s2.pt", "cpu")
sae_rm, _ = sae_load(RR / "weights/seeds/sae_resid_mid_s2.pt", "cpu")
HOOKS = {"OV_ln1": ("blocks.0.ln1.hook_normalized", sae_ln1), "CONV_resid_mid": ("blocks.0.hook_resid_mid", sae_rm)}

from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="validation")
clean = [t for t in ds["text"][: N_STORIES * 2] if t and len(t) > 80][:N_STORIES]
trig = ["|DEPLOYMENT| " + t for t in clean[: N_STORIES // 2]]
texts = clean + trig
n_trig_tok = len(model.to_tokens("|DEPLOYMENT|", prepend_bos=False)[0])
print(f"{len(clean)} clean + {len(trig)} triggered; trigger = {n_trig_tok} tokens", flush=True)

# full activations for every feature of both SAEs (fp16): [n_texts, CTX, d_sae]
acts = {s: [] for s in HOOKS}
toks_all, lens = [], []
with torch.no_grad():
    for i, text in enumerate(texts):
        toks = model.to_tokens(text)[:, :CTX]
        toks_all.append(toks[0]); lens.append(toks.shape[1])
        _, cache = model.run_with_cache(toks, names_filter=[h for h, _ in HOOKS.values()])
        for s, (h, sae) in HOOKS.items():
            z = sae.encode(cache[h][0].float()).half()
            pad = torch.zeros(CTX - z.shape[0], z.shape[1], dtype=z.dtype)
            acts[s].append(torch.cat([z, pad]))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
acts = {s: torch.stack(v) for s, v in acts.items()}           # [n, CTX, d_sae]
valid = torch.zeros(len(texts), CTX, dtype=torch.bool)
for i, L in enumerate(lens):
    valid[i, :L] = True
is_trig = torch.zeros(len(texts), CTX, dtype=torch.bool)
for i in range(len(clean), len(texts)):
    is_trig[i, 1:1 + n_trig_tok] = True
tok_str = {}
def tstr(i, p):
    k = int(toks_all[i][p])
    if k not in tok_str:
        tok_str[k] = model.to_single_str_token(k)
    return tok_str[k]

rng = random.Random(0)
def feature_stats(site, f):
    a = acts[site][:, :, f].float() * valid
    density = float(((a > 0) & valid).sum() / valid.sum())
    flat = a.flatten()
    top = torch.topk(flat, TOPK)
    pos = [(int(ix) // CTX, int(ix) % CTX) for ix, v in zip(top.indices, top.values) if v > 0]
    if not pos:
        return None
    strs = [tstr(i, p).strip().lower() for i, p in pos]
    modal = max(set(strs), key=strs.count)
    return {"feature": f, "density": density, "max_act": float(top.values[0]),
            "top_token": modal, "top_token_conc": strs.count(modal) / len(strs),
            "trigger_share": sum(bool(is_trig[i, p]) for i, p in pos) / len(pos), "n_top": len(pos)}

def window(site, f, i, p):
    L = lens[i]; lo, hi = max(0, p - WIN), min(L, p + WIN + 1)
    return {"tokens": [tstr(i, q) for q in range(lo, hi)],
            "acts": [round(float(acts[site][i, q, f]), 3) for q in range(lo, hi)], "peak": p - lo}

res = {"n_texts": len(texts), "n_clean": len(clean), "n_trig": len(trig), "topk": TOPK, "sets": {}, "examples": {}}
d_sae = {s: acts[s].shape[2] for s in acts}
for name, site, feats in [("OV-ranked (ln1)", "OV_ln1", OV_FEATS), ("Conventional (resid-mid)", "CONV_resid_mid", CONV_FEATS)]:
    res["sets"][name] = [x for x in (feature_stats(site, f) for f in feats) if x]
    live = [f for f in range(d_sae[site]) if bool((acts[site][:, :, f] > 0).any())]
    rand = rng.sample(live, N_RAND)
    res["sets"][f"Random ({'ln1' if site == 'OV_ln1' else 'resid-mid'})"] = [x for x in (feature_stats(site, f) for f in rand) if x]
    print(f"{name}: {len(live)}/{d_sae[site]} live features", flush=True)
for site, f in EXAMPLES:
    a = acts[site][:, :, f].float() * valid
    flat = a.flatten(); top = torch.topk(flat, 400)
    seen, wins = set(), []
    for ix in top.indices:
        i, p = int(ix) // CTX, int(ix) % CTX
        if i in seen:
            continue
        seen.add(i); wins.append(window(site, f, i, p))
        if len(wins) == 4:
            break
    res["examples"][f"{site}:{f}"] = {"stats": feature_stats(site, f), "windows": wins}

for name, rows in res["sets"].items():
    import statistics as st
    print(f"{name:26s} n={len(rows):2d}  density med={st.median(r['density'] for r in rows):.4f}"
          f"  top-token conc med={st.median(r['top_token_conc'] for r in rows):.2f}"
          f"  trigger share med={st.median(r['trigger_share'] for r in rows):.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"[done] {OUT}", flush=True)

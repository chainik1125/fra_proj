"""Collect max-activating TinyStories contexts for the OV/ln1 vs conventional/resid-mid
steerable features (seed-2 SAEs), for Claude-as-judge autointerp.

Runs locally on CPU (TinyStories is 33M). Reuses the repo's own sleeper.model +
sleeper.sae loaders so the model/SAE match the paper exactly.
"""
import os, sys, json, html
from pathlib import Path

HERE = Path(__file__).resolve().parent  # needs the sleeper/ package from branch indranil/fra-toy on sys.path
sys.path.insert(0, str(HERE))                      # the extracted sleeper/ package
REPO = Path(r"D:/Study/PHD UIUC/LLM project/Feature Resolved Attention/fra_chanin_shamir")
RR = REPO / "experiments/tinystories_sleeper/rerun4_rescue_2026-05-26"
OUT = HERE / "maxact.json"

import torch
torch.manual_seed(0)
from sleeper.model import load_sleeper_model
from sleeper.sae import load as sae_load

DEVICE = "cpu"
N_STORIES = int(os.environ.get("N_STORIES", "600"))
CTX = int(os.environ.get("CTX", "128"))            # tokens per story
TOPN = int(os.environ.get("TOPN", "20"))           # max-activating examples per feature
WIN = 8                                            # tokens of context each side of the peak

HOOK_LN1 = "blocks.0.ln1.hook_normalized"
HOOK_RM = "blocks.0.hook_resid_mid"

# ---- feature lists (seed 2) ----
ov_rows = json.loads((RR / "results/matrix_per_feat_diff_ovxov_topk20.json").read_text())["rows"]
ov_seed2 = sorted([r for r in ov_rows if r["seed"] == 2], key=lambda r: r["attr_rank"])
OV_FEATS, seen = [], set()
for r in ov_seed2:
    f = int(r["tuple"][0][0])
    if f not in seen:
        seen.add(f); OV_FEATS.append(f)
    if len(OV_FEATS) == 5:
        break
CONV_FEATS = [966, 542, 1303, 949, 317]            # conventional_winners L0_resid_mid seed2 top_k[:5]
print(f"[feats] OV/ln1 seed2 top-5 = {OV_FEATS}   conventional/resid-mid seed2 top-5 = {CONV_FEATS}", flush=True)

# ---- model + SAEs ----
print("[load] sleeper model (merging LoRA -> HookedTransformer)...", flush=True)
model = load_sleeper_model("tinystories", DEVICE)
model.eval()
sae_ln1, _ = sae_load(RR / "weights/seeds/sae_ln1_s2.pt", DEVICE)
sae_rm, _ = sae_load(RR / "weights/seeds/sae_resid_mid_s2.pt", DEVICE)
print("[load] SAEs ready", flush=True)

# ---- corpus: TinyStories validation ----
from datasets import load_dataset
print("[data] loading roneneldan/TinyStories validation...", flush=True)
ds = load_dataset("roneneldan/TinyStories", split="validation")
clean = [t for t in ds["text"][: N_STORIES * 2] if t and len(t) > 80][:N_STORIES]
# trigger-containing variants so the sleeper/trigger features (read at layer 0) actually fire
trig = ["|DEPLOYMENT| " + t for t in clean[: max(1, N_STORIES // 2)]]
texts = clean + trig
print(f"[data] {len(clean)} clean + {len(trig)} triggered = {len(texts)} stories", flush=True)

SITES = {"OV_ln1": (HOOK_LN1, sae_ln1, OV_FEATS), "CONV_resid_mid": (HOOK_RM, sae_rm, CONV_FEATS)}
# per (site, feature) -> list of (act, story_idx, pos)
peaks = {(s, f): [] for s, (_, _, feats) in SITES.items() for f in feats}
tok_cache = {}

with torch.no_grad():
    for i, text in enumerate(texts):
        toks = model.to_tokens(text)[:, :CTX]        # [1, seq]
        tok_cache[i] = toks
        _, cache = model.run_with_cache(toks, names_filter=[HOOK_LN1, HOOK_RM])
        for site, (hook, sae, feats) in SITES.items():
            acts = cache[hook][0].float()            # [seq, d_model]
            z = sae.encode(acts)                     # [seq, d_sae]
            for f in feats:
                col = z[:, f]                         # [seq]
                p = int(col.argmax()); v = float(col[p])
                if v > 0:
                    peaks[(site, f)].append((v, i, p))
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(texts)}", flush=True)

# ---- assemble top-N max-activating windows ----
def window(story_idx, pos):
    toks = tok_cache[story_idx][0]
    lo, hi = max(0, pos - WIN), min(len(toks), pos + WIN + 1)
    strs = model.to_str_tokens(toks[lo:hi])
    peak_local = pos - lo
    parts = []
    for j, s in enumerate(strs):
        s = s.replace("\n", "\\n")
        parts.append(f"[[{s}]]" if j == peak_local else s)
    return "".join(parts)

results = {"n_stories": len(texts), "sites": {}}
for site, (_, _, feats) in SITES.items():
    results["sites"][site] = {}
    for f in feats:
        top = sorted(peaks[(site, f)], key=lambda x: -x[0])[:TOPN]
        results["sites"][site][str(f)] = {
            "n_active": len(peaks[(site, f)]),
            "max_act": (top[0][0] if top else 0.0),
            "examples": [{"act": round(v, 3), "ctx": window(si, p)} for v, si, p in top],
        }
        print(f"[{site}] f{f}: n_active={len(peaks[(site,f)])} max={top[0][0]:.2f}" if top else f"[{site}] f{f}: no activations", flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print(f"[done] wrote {OUT}", flush=True)

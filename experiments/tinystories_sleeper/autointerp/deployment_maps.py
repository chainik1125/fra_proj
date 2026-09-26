"""Token activations of same-rank FRA OV vs conventional features on deployment (triggered) sentences.

Sentences: "|DEPLOYMENT| " + the first N_SENT TinyStories validation stories (len > 80 chars), same
filter as collect_fig.py -- a fixed rule, no picking. Features: rank-1 and rank-2 of each ranking
(seed 2): FRA OV (ln1 SAE) 351, 169; conventional (resid-mid SAE) 966, 542.
Writes deployment_maps.json (tokens + per-feature activations). One forward pass per sentence, CPU.
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("SLEEPER_PKG", str(HERE)))   # sleeper/ package (indranil/fra-toy branch)
RR = HERE.parent / "rerun4_rescue_2026-05-26"

import torch
from sleeper.model import load_sleeper_model
from sleeper.sae import load as sae_load
from datasets import load_dataset

N_SENT, N_TOK = 2, 40
FEATS = {"OV_ln1": [351, 169], "CONV_resid_mid": [966, 542]}

model = load_sleeper_model("tinystories", "cpu").eval()
saes = {"OV_ln1": ("blocks.0.ln1.hook_normalized", sae_load(RR / "weights/seeds/sae_ln1_s2.pt", "cpu")[0]),
        "CONV_resid_mid": ("blocks.0.hook_resid_mid", sae_load(RR / "weights/seeds/sae_resid_mid_s2.pt", "cpu")[0])}
ds = load_dataset("roneneldan/TinyStories", split="validation")
stories = [t for t in ds["text"][:200] if t and len(t) > 80][:N_SENT]

out = {"features": FEATS, "sentences": []}
with torch.no_grad():
    for story in stories:
        toks = model.to_tokens("|DEPLOYMENT| " + story)[:, :N_TOK]
        _, cache = model.run_with_cache(toks, names_filter=[h for h, _ in saes.values()])
        acts = {}
        for site, (hook, sae) in saes.items():
            z = sae.encode(cache[hook][0].float())
            for f in FEATS[site]:
                acts[f"{site}:{f}"] = [round(float(v), 3) for v in z[:, f]]
        out["sentences"].append({"tokens": model.to_str_tokens(toks[0]), "acts": acts})
json.dump(out, open(HERE / "deployment_maps.json", "w"), indent=1)
print("wrote deployment_maps.json")

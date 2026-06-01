"""Produce the OV deployable-winner tuples_json for Fig 2 (Fig 3's exact rule).

Per SAE seed: attribution prefilter (dep-clean activation diff, top-20) -> rerank
by signed cos(W_dec_ln1[f], v_md) with v_md the attention-weighted dep-minus-clean
ln1 difference -> take the cosine top-3 -> keep the lowest-ASR feature (greedy OV
ablation screen at a high alpha). Writes a winner tuples_json for scripts/eval.py.
Out: results/v3_ov_tuples.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sleeper.eval import LN1_HOOK, PAT_HOOK
from sleeper.hooks import compute_sae_delta, ov_only_steer_hook, generate_with_hooks, make_greedy_sampler
from sleeper.metrics import rank_features_by_dep_clean, sleeper_fired_mask
from sleeper.model import (MODELS, load_sleeper_model, cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all

DEV = "cuda"; MODEL = "tinystories"; SAE_DIR = "weights/seeds_leftpad"
SAE_SEEDS = [0, 1, 2, 3, 4, 5]; N_SEL = 200; GEN = 16
TOPK = 20; TOP_COS = 3; SCREEN_A = 6.0; OUT = "results/v3_ov_tuples.json"


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    seq_len = MODELS[MODEL].seq_len
    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = sel.is_deployment.to(DEV)
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, LN1_HOOK])
    A = acts[PAT_HOOK].to(DEV).float(); X = acts[LN1_HOOK].to(DEV).float()
    pmf = sel.attention_mask.to(DEV).float()
    recv = A.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (X * recv.unsqueeze(-1)).sum(1)
    vmd = amean[isd].mean(0) - amean[~isd].mean(0); vn = vmd / vmd.norm().clamp_min(1e-9)
    W_V = model.W_V[0].detach()

    # deployment prompts for the greedy ASR screen (left-padded by the loader)
    deplp = sel.tokens[sel.is_deployment].to(DEV)
    depat = sel.attention_mask[sel.is_deployment].to(DEV); pm = depat.bool()
    sampler = make_greedy_sampler()

    per_seed = {}
    for s in SAE_SEEDS:
        sae, _ = sae_load(Path(f"{SAE_DIR}/sae_ln1_s{s}.pt"), device=DEV)
        z = encode_all(sae, acts[LN1_HOOK])
        cand = rank_features_by_dep_clean(z, sel.is_deployment, sel.attention_mask, top_k=TOPK)["top_indices"].cpu().tolist()
        W = sae.W_dec.to(DEV).float()
        scored = sorted(((float(W[f] @ vn / W[f].norm().clamp_min(1e-9)), int(f)) for f in cand), reverse=True)
        top3 = scored[:TOP_COS]
        best = (2.0, top3[0][1])
        for cos, f in top3:
            delta = compute_sae_delta(model, sae, LN1_HOOK, f, deplp, pm, attention_mask=depat)
            gtok = generate_with_hooks(model, deplp, ov_only_steer_hook(delta, SCREEN_A, W_V, 0),
                                       GEN, sampler, attention_mask=depat)
            asr = float(sleeper_fired_mask(gtok.cpu(), tok).float().mean())
            print(f"[fig2-ov] seed {s} feat {f} (cos={cos:.3f}): screen asr@a{SCREEN_A}={asr:.3f}", flush=True)
            if asr < best[0]:
                best = (asr, f)
        feat = best[1]
        per_seed[str(s)] = [[[feat, "V"]]]
        print(f"[fig2-ov] seed {s}: WINNER feat {feat} (min screen-asr={best[0]:.3f} of cos-top{TOP_COS})", flush=True)

    out = {"channel": "ov", "regime": "diff", "mode": "winner",
           "selection": "attr_top20_attncos_top3_minasr", "sae_dir": SAE_DIR, "per_seed": per_seed}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"[fig2-ov] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

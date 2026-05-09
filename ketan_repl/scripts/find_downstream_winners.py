"""Per-resid_mid-SAE downstream attribution: find the feature that best
suppresses sleeper at α∈{2,4} for each (training_duration, sae_seed).

Mirrors jamie's `feature_set_pipeline.py --eval_mode single` flow exactly,
substituting:
  • selection: dep-vs-clean activation difference on prompt positions
    (analog of `select_features(method="jamie")` for OV — there's no W_V
     to project through at resid_mid, so the per-feature mean activation
     diff is the natural attribution score)
  • intervention: `additive_steer_hook` at `blocks.0.hook_resid_mid`
    (analog of `channel_steer_hook` at `attn.hook_v` for OV-only)

Stage 0 / Stage 1 / winner-pick are byte-equivalent to jamie's pipeline.

Writes results/per_seed_downstream_winners.json:
  {<key>: {"path", "winner", "min_asr", "screen_alpha", "topK"}}
where <key> is e.g. "4k_s0", "50k_s2".
"""
from __future__ import annotations
import json
from pathlib import Path

import torch

from sleeper.hooks import (additive_steer_hook, compute_sae_delta,
                            generate_with_hooks, make_greedy_sampler)
from sleeper.metrics import (asr_16, rank_features_by_dep_clean,
                              teacher_forced_sleeper_logp)
from sleeper.model import (left_pad_prompts, load_dep_prompts,
                            load_paired_dataset, load_sleeper_model,
                            prompt_mask_from_markers)
from sleeper.sae import encode_all, load as sae_load


RESID_MID = "blocks.0.hook_resid_mid"
N_SEL = 100                  # 50 dep + 50 clean (jamie default)
SEQ_LEN = 128
SCREEN_ALPHAS = [2.0, 4.0]
TOP_K = 20
GEN_TOKENS = 16


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok   = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Selection split (jamie's setup): fixed-length from load_paired_dataset
    splits = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=0,
                                  seq_len=SEQ_LEN, seed=0)
    sel_split = splits["val"]
    sel_pmask = prompt_mask_from_markers(SEQ_LEN, sel_split.story_marker_pos)
    sel_dep         = sel_split.tokens[sel_split.is_deployment].to(device)
    sel_dep_pmask   = sel_pmask[sel_split.is_deployment].to(device)

    # Combined dep+clean for activation-difference ranking
    is_dep = sel_split.is_deployment.to(device)
    combined_lp = sel_split.tokens.to(device)
    combined_pmask = sel_pmask.to(device)

    # Variable-length dep prompts for ASR screen (jamie uses load_dep_prompts here)
    raw_dep = load_dep_prompts(tok, N_SEL // 2, split="test")
    sel_lp, sel_attn = left_pad_prompts(raw_dep, pad_id)
    sel_lp, sel_attn = sel_lp.to(device), sel_attn.to(device)

    sel_base_logp = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()
    print(f"[find] baseline Δdep-logp ref = {sel_base_logp:.4f}")

    cells = [
        ("4k_s0",  "weights/sae_resid_mid.pt"),
        ("4k_s1",  "weights/seeds/sae_resid_mid_s1.pt"),
        ("4k_s2",  "weights/seeds/sae_resid_mid_s2.pt"),
        ("50k_s0", "weights/sae_resid_mid_50k.pt"),
        ("50k_s1", "weights/seeds_50k/sae_resid_mid_s1.pt"),
        ("50k_s2", "weights/seeds_50k/sae_resid_mid_s2.pt"),
    ]
    out: dict = {}
    for label, path in cells:
        print(f"\n[find] === {label}  {path} ===")
        sae_mid, _ = sae_load(Path(path), device=device)

        # Encode SAE features at resid_mid for the combined dep+clean prompts
        _, cache = model.run_with_cache(
            combined_lp, return_type=None,
            names_filter=lambda n: n == RESID_MID,
        )
        acts = cache[RESID_MID]                          # (2*B, T, d)
        z = encode_all(sae_mid, acts).to(device)         # (2*B, T, d_sae)

        ranked = rank_features_by_dep_clean(z, is_dep, combined_pmask, top_k=TOP_K)
        topK = ranked["top_indices"].cpu().tolist()
        print(f"[find] {label} top-{TOP_K}: {topK[:10]}…")

        # Stage-0 dlogp screen — same shape as jamie's pipeline
        dlogp_per_feat: dict[int, float] = {}
        for f in topK:
            best = float("inf")
            for a in SCREEN_ALPHAS:
                delta = compute_sae_delta(model, sae_mid, RESID_MID, int(f),
                                           sel_dep, sel_dep_pmask)
                hooks = additive_steer_hook(delta, a, RESID_MID)
                lp = teacher_forced_sleeper_logp(model, tok, sel_dep,
                                                   fwd_hooks=hooks).mean().item()
                d = lp - sel_base_logp
                if d < best: best = d
            dlogp_per_feat[int(f)] = best
        keep_n = max(1, TOP_K // 2)
        survivors = sorted(dlogp_per_feat, key=dlogp_per_feat.get)[:keep_n]
        print(f"[find] {label} stage-0 keep {keep_n}: {survivors[:6]}…  "
              f"best Δdep-logp = {dlogp_per_feat[survivors[0]]:+.3f}")

        # Stage-1 ASR screen on survivors
        sampler = make_greedy_sampler()
        asr_table: dict[int, tuple[float, float]] = {}
        for f in survivors:
            best_asr = 1.0
            best_alpha = SCREEN_ALPHAS[0]
            for a in SCREEN_ALPHAS:
                delta = compute_sae_delta(model, sae_mid, RESID_MID, int(f),
                                           sel_lp, sel_attn.bool(),
                                           attention_mask=sel_attn)
                hooks = additive_steer_hook(delta, a, RESID_MID)
                gen = generate_with_hooks(model, sel_lp, hooks, GEN_TOKENS,
                                           sampler, attention_mask=sel_attn)
                asr = asr_16(gen, tok)
                if asr < best_asr:
                    best_asr = asr; best_alpha = a
            asr_table[int(f)] = (best_asr, best_alpha)
            print(f"[find] {label} f={f:>5}  min-asr={best_asr:.3f}  best_α={best_alpha}")

        winner = min(asr_table, key=lambda f: asr_table[f][0])
        out[label] = {
            "path": path,
            "winner": int(winner),
            "min_asr": float(asr_table[winner][0]),
            "screen_alpha": float(asr_table[winner][1]),
            "topK": [int(x) for x in topK],
        }
        print(f"[find] {label} ⇒ winner f={winner} (min-asr={asr_table[winner][0]:.3f})")

    Path("results/per_seed_downstream_winners.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/per_seed_downstream_winners.json")
    print(json.dumps({k: v["winner"] for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()

"""Per-checkpoint feature selection + individual-steer screen → winner.

Selection is the Wang **diff-of-means**, target-free, depending only on the SAE
under test (+ the model's frozen W_V/W_O):
  • OV cell        — `rank_ov_diff` (‖Σ_h diff_M[h,λ]·(W_dec[λ]@W_OV^h)‖₂)
  • resid-mid cell — `rank_features_by_dep_clean` (mean_dep − mean_clean activation)

The screen mirrors `find_downstream_winners.py` (jamie's `feature_set_pipeline
--eval_mode single`): stage-0 Δlogp cull to top-K/2, stage-1 greedy-ASR winner
by (min-ASR → min attr-rank → min α). No precomputed winners; every SAE is
re-screened (a feature index is a checkpoint coordinate, not a recipe).

The selection-split activations (attention pattern + ln1/resid_mid acts) are
checkpoint-invariant — build them ONCE with `build_sel_caches` and reuse across
all SAEs; only `z = sae.encode(acts)` is per-SAE.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from sleeper.attribution import rank_ov_diff
from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks, compute_sae_delta,
    generate_with_hooks, make_greedy_sampler, resolve_channel_deltas,
)
from sleeper.metrics import (
    asr_16, batched_asr_16, rank_features_by_dep_clean, teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts, load_paired_dataset,
    prompt_mask_from_markers,
)
from sleeper.sae import encode_all

LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID = "blocks.0.hook_resid_mid"
PAT_HOOK = "blocks.0.attn.hook_pattern"
N_SEL = 100              # 50 dep + 50 clean (jamie default)
SEQ_LEN = 128
SCREEN_ALPHAS = (2.0, 4.0)
TOP_K = 20
GEN_TOKENS = 16


@dataclass
class SelCaches:
    """Checkpoint-invariant selection-split tensors. Build once per process."""
    A: torch.Tensor                 # (B, n_heads, T, T) attention pattern
    ln1_acts: torch.Tensor          # (B, T, d) cpu — for OV z encoding
    resid_mid_acts: torch.Tensor    # (B, T, d) cpu — for resid-mid z encoding
    is_dep: torch.Tensor            # (B,) bool on device
    sel_pmask: torch.Tensor         # (B, T) bool on device (query/prompt mask)
    sel_dep: torch.Tensor           # (B_dep, T) dep-only fixed-length tokens
    sel_dep_pmask: torch.Tensor     # (B_dep, T) bool
    sel_lp: torch.Tensor            # (B_dep, P) left-padded variable-length dep prompts
    sel_attn: torch.Tensor          # (B_dep, P) bool
    sel_base_logp: float            # baseline teacher-forced sleeper logp
    W: dict                         # {Q,K,V: (n_heads, d_model, d_head)} at block 0
    W_O: torch.Tensor               # (n_heads, d_head, d_model) at block 0


@torch.no_grad()
def build_sel_caches(model, device) -> SelCaches:
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    splits = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=0,
                                 seq_len=SEQ_LEN, seed=0)
    sel = splits["val"]
    sel_pmask = prompt_mask_from_markers(SEQ_LEN, sel.story_marker_pos)

    acts = cache_activations(model, sel.tokens, [PAT_HOOK, LN1_HOOK, RESID_MID])

    is_dep = sel.is_deployment.to(device)
    sel_dep = sel.tokens[sel.is_deployment].to(device)
    sel_dep_pmask = sel_pmask[sel.is_deployment].to(device)

    raw_dep = load_dep_prompts(tok, N_SEL // 2, split="test")
    sel_lp, sel_attn = left_pad_prompts(raw_dep, pad_id)
    sel_lp, sel_attn = sel_lp.to(device), sel_attn.to(device)

    sel_base_logp = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()

    W = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O = model.W_O[0].detach().to(device)
    return SelCaches(
        A=acts[PAT_HOOK].to(device), ln1_acts=acts[LN1_HOOK],
        resid_mid_acts=acts[RESID_MID], is_dep=is_dep,
        sel_pmask=sel_pmask.to(device), sel_dep=sel_dep, sel_dep_pmask=sel_dep_pmask,
        sel_lp=sel_lp, sel_attn=sel_attn, sel_base_logp=sel_base_logp, W=W, W_O=W_O,
    )


def _winner_from_screen(asr_table: dict, attr_rank: dict) -> int:
    """min-ASR → min attr-rank → min α (jamie's `--final_selection rank` rule)."""
    return min(asr_table, key=lambda f: (asr_table[f][0], attr_rank[f], asr_table[f][1]))


@torch.no_grad()
def screen_winner_ov(model, sae_ln1, c: SelCaches, device, *,
                     top_k: int = TOP_K, screen_alphas=SCREEN_ALPHAS) -> dict:
    """OV cell: Wang OV-diff ranking → stage-0 Δlogp cull → stage-1 greedy-ASR winner."""
    tok = model.tokenizer
    z_ln1 = encode_all(sae_ln1, c.ln1_acts).to(device)
    ranked = rank_ov_diff(c.A, z_ln1, sae_ln1, c.W["V"], c.W_O, c.is_dep,
                          query_mask=c.sel_pmask)
    topK = ranked["top_indices"][:top_k].cpu().tolist()
    attr_rank = {int(f): i for i, f in enumerate(topK)}

    dlogp_per_feat: dict[int, float] = {}
    for f in topK:
        cd = resolve_channel_deltas([(int(f), "V")], ACTIVE_CHANNELS["ov"], model,
                                    sae_ln1, LN1_HOOK, c.sel_dep, c.sel_dep_pmask)
        best = float("inf")
        for a in screen_alphas:
            hooks = build_hooks(cd, a, ACTIVE_CHANNELS["ov"], c.W, LN1_HOOK, 0)
            lp = teacher_forced_sleeper_logp(model, tok, c.sel_dep,
                                             fwd_hooks=hooks).mean().item()
            best = min(best, lp - c.sel_base_logp)
        dlogp_per_feat[int(f)] = best
    keep_n = max(1, top_k // 2)
    survivors = sorted(dlogp_per_feat, key=dlogp_per_feat.get)[:keep_n]

    asr_table: dict[int, tuple[float, float]] = {}
    for f in survivors:
        best_asr, best_alpha = 1.0, float(screen_alphas[0])
        for a in screen_alphas:
            asr = batched_asr_16(model, sae_ln1, LN1_HOOK, [(int(f), "V")], a,
                                 ACTIVE_CHANNELS["ov"], c.W, 0, c.sel_lp, c.sel_attn,
                                 GEN_TOKENS, sampler=None)
            if asr < best_asr:
                best_asr, best_alpha = asr, float(a)
        asr_table[int(f)] = (best_asr, best_alpha)

    winner = _winner_from_screen(asr_table, attr_rank)
    return {
        "winner": int(winner),
        "attr_rank": attr_rank[winner],
        "min_asr": asr_table[winner][0],
        "screen_alpha": asr_table[winner][1],
        "topK": [int(x) for x in topK],
        "stage0_survivors": [int(x) for x in survivors],
    }


@torch.no_grad()
def screen_winner_resid_mid(model, sae_mid, c: SelCaches, device, *,
                            top_k: int = TOP_K, screen_alphas=SCREEN_ALPHAS) -> dict:
    """Resid-mid cell: activation-diff ranking → stage-0 Δlogp cull → stage-1 greedy-ASR winner."""
    tok = model.tokenizer
    z = encode_all(sae_mid, c.resid_mid_acts).to(device)
    ranked = rank_features_by_dep_clean(z, c.is_dep, c.sel_pmask, top_k=top_k)
    topK = ranked["top_indices"].cpu().tolist()
    attr_rank = {int(f): i for i, f in enumerate(topK)}

    dlogp_per_feat: dict[int, float] = {}
    for f in topK:
        delta = compute_sae_delta(model, sae_mid, RESID_MID, int(f),
                                  c.sel_dep, c.sel_dep_pmask)
        best = float("inf")
        for a in screen_alphas:
            hooks = additive_steer_hook(delta, a, RESID_MID)
            lp = teacher_forced_sleeper_logp(model, tok, c.sel_dep,
                                             fwd_hooks=hooks).mean().item()
            best = min(best, lp - c.sel_base_logp)
        dlogp_per_feat[int(f)] = best
    keep_n = max(1, top_k // 2)
    survivors = sorted(dlogp_per_feat, key=dlogp_per_feat.get)[:keep_n]

    sampler = make_greedy_sampler()
    asr_table: dict[int, tuple[float, float]] = {}
    for f in survivors:
        delta = compute_sae_delta(model, sae_mid, RESID_MID, int(f),
                                  c.sel_lp, c.sel_attn.bool(), attention_mask=c.sel_attn)
        best_asr, best_alpha = 1.0, float(screen_alphas[0])
        for a in screen_alphas:
            hooks = additive_steer_hook(delta, a, RESID_MID)
            gen = generate_with_hooks(model, c.sel_lp, hooks, GEN_TOKENS, sampler,
                                      attention_mask=c.sel_attn)
            asr = asr_16(gen, tok)
            if asr < best_asr:
                best_asr, best_alpha = asr, float(a)
        asr_table[int(f)] = (best_asr, best_alpha)

    winner = _winner_from_screen(asr_table, attr_rank)
    return {
        "winner": int(winner),
        "attr_rank": attr_rank[winner],
        "min_asr": asr_table[winner][0],
        "screen_alpha": asr_table[winner][1],
        "topK": [int(x) for x in topK],
        "stage0_survivors": [int(x) for x in survivors],
    }

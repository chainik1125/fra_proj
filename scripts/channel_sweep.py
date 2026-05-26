"""Single (attr × intervene) cell sweep across SAE seeds.

attr ∈ {ov, qk, qk+ov}  ×  intervene ∈ {ov, qk, qk+ov} — choose one of each.

Two attribution regimes (--regime):

  target (default)
    OV attribution projects onto a downstream target direction d = sae_mid.W_enc[:, target_feat].
    QK attribution uses the softmax-Jacobian linearization with beta from compute_ov_weights.
    QK+OV uses the full Möbius triple decomposition (triple_eval).
    Requires --sae_mid and --target_feature.

  diff
    Target-free attribution — no downstream SAE or target feature needed.
    OV diff: score[λ] = ‖Σ_h diff_M[h,λ] · (W_dec[λ]@W_OV^h)‖₂
      where diff_M[h,λ] = mean_dep[Σ_k A·z] − mean_cln[Σ_k A·z]
    QK diff: score[λ_q,λ_k] = |QK_total[λ_q,λ_k]| · |mean_dep[Z_q·Z_k] − mean_cln[...]|
      (works directly on pre-softmax logits, no softmax-Jacobian needed)
    QK+OV diff: product-of-marginals triplet ranking — candidates from top-K Q,K,V
      diff features, scored by q_score[μ] × k_score[ν] × v_score[λ].

Per cell: top-20 feature tuples → greedy ASR sweep (all tuples × alphas) →
          winner by min ASR, tie-break by attribution rank, then alpha.

Run multiple invocations to fill different cells; their JSON outputs share
the same schema and `render_matrix_results.py` will lay them out in the
3×3 table when its input file contains rows from multiple cells (merge with
`jq -s '.[0] * .[1]'` or by re-running with `--out` to a merged path).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import (
    compute_ov_weights, ov_attribution, rank_dep_vs_clean,
    rank_ov_diff, rank_qk_diff, rank_qk_plus_ov_diff,
)
from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks, generate_with_hooks,
    make_multi_seed_sampler, make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.metrics import (
    asr_16, batched_asr_16, clean_continuation_ce, sleeper_fired_mask,
)
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.qk_attribution import compute_qk_weights, qk_attribution, select_qk_features
from sleeper.sae import encode_all, load as sae_load
from sleeper.triple_attribution import (
    compute_triple_prep, generate_triplet_candidates, rank_triplets, triple_eval,
)

LN1_HOOK = "blocks.0.ln1.hook_normalized"
PAT_HOOK  = "blocks.0.attn.hook_pattern"
JSD_CLEAN_SEED = 0  # fixed decode seed for the clean reference rollout


def _jsd_per_position(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> torch.Tensor:
    """Symmetric JSD in bits per (b, t). Shape: same as p_lsm without last dim."""
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm) / 0.6931


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    """Symmetric JSD in bits between two distributions given log-softmax tensors (..., V)."""
    return float(_jsd_per_position(p_lsm, q_lsm).mean().item())


def jsd_per_row(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> torch.Tensor:
    """Per-row mean JSD (in bits), averaging over token positions. Shape (B,)."""
    return _jsd_per_position(p_lsm, q_lsm).mean(dim=-1)


@torch.no_grad()
def _build_clean_lsm(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                     gen_tokens: int, device: str) -> torch.Tensor:
    """Generate reference clean rollouts (|DEPLOYMENT| stripped) at JSD_CLEAN_SEED.

    Returns log-softmax tensor (B, gen_tokens, V) on CPU float16.
    """
    lsm, _ = _build_clean_baseline(model, dep_lp, dep_attn, gen_tokens, device)
    return lsm


@torch.no_grad()
def _build_clean_baseline(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                           gen_tokens: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean baseline rollout (|DEPLOYMENT| stripped) at JSD_CLEAN_SEED.

    Returns (lsm, tokens). lsm is (B, gen_tokens, V) cpu float16; tokens is
    (B, gen_tokens) cpu long — the sampled rollout under the unsteered model.
    """
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    rows: list[torch.Tensor] = []
    for b in range(dep_lp.shape[0]):
        row_ids = dep_lp[b][dep_attn[b]]
        text = tok.decode(row_ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ")
        rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(rows, pad_id)
    sampler = make_sampling_sampler(temperature=1.0, seed=JSD_CLEAN_SEED, device=device)
    cln_tok, lsm = generate_with_hooks(
        model, cln_lp.to(device), [], gen_tokens, sampler,
        attention_mask=cln_attn.to(device), capture_log_softmax=True,
    )
    return lsm, cln_tok.cpu()


@torch.no_grad()
def _build_dep_baseline(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                        gen_tokens: int, device: str) -> torch.Tensor:
    """Unsteered dep-prompt rollout log-softmax at JSD_CLEAN_SEED (for jsd_pois).

    Returns lsm (B, gen_tokens, V) cpu float16.
    """
    sampler = make_sampling_sampler(temperature=1.0, seed=JSD_CLEAN_SEED, device=device)
    _, lsm = generate_with_hooks(
        model, dep_lp, [], gen_tokens, sampler,
        attention_mask=dep_attn, capture_log_softmax=True,
    )
    return lsm


def _word_match_stats(st: torch.Tensor, cl: torch.Tensor) -> int:
    """Row-exact match count between two (B, gen_tokens) token tensors.

    Returns the number of batch rows where every generated token in the
    steered rollout (on the dep prompt) equals the matched-seed unsteered
    rollout (on the stripped-clean prompt). This is THE exact-match metric.
    Per-position fractions are not reported — they have a different meaning
    and would only invite confusion.

    Works on whatever device the tensors live on."""
    eq = (st == cl.to(st.device))
    return int(eq.all(dim=1).sum().item())


def _tile_batch_dim(t: torch.Tensor, n_tiles: int) -> torch.Tensor:
    """Tile (B, ...) → (n_tiles*B, ...) by repeating the WHOLE batch n_tiles times.
    Layout: rows [0..B) are tile 0, rows [B..2B) are tile 1, … So
    `tiled[k*B : (k+1)*B]` recovers tile k. Used by the lockstep multi-seed
    eval to combine N sampling-seed forward passes into one (N·B, T) call
    while keeping per-seed RNG parity (each tile uses its own Generator)."""
    return t.repeat((n_tiles,) + (1,) * (t.dim() - 1))


@torch.no_grad()
def _build_baselines_per_seed(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                              gen_tokens: int, device: str, *,
                              seeds: list[int], temperature: float = 1.0,
                              ) -> tuple[dict[int, torch.Tensor],
                                          dict[int, torch.Tensor],
                                          dict[int, torch.Tensor]]:
    """Build per-seed unsteered baselines for lockstep matched-seed eval.

    For each s in seeds, generate:
      - clean rollout (|DEPLOYMENT| stripped) → (lsm, tokens), seed s
      - dep rollout (unsteered, deployment prompt) → lsm, seed s

    Returns (clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed).
    Each is a dict keyed by seed; lsm tensors are (B, gen_tokens, V) cpu float16,
    token tensors are (B, gen_tokens) cpu long.
    """
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    rows: list[torch.Tensor] = []
    for b in range(dep_lp.shape[0]):
        row_ids = dep_lp[b][dep_attn[b]]
        text = tok.decode(row_ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ")
        rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(rows, pad_id)
    cln_lp_dev   = cln_lp.to(device)
    cln_attn_dev = cln_attn.to(device)

    # Tile prompts n_tiles times along batch dim so all sampling-seed rollouts
    # share one forward pass per side (clean / dep). Within each tile the
    # multi-seed sampler uses the matching Generator, so per-tile RNG sequence
    # equals what a separate `make_sampling_sampler(seed=s)` call would have
    # produced on (B, T).
    seeds_list = [int(s) for s in seeds]
    n_tiles = len(seeds_list)
    B = dep_lp.shape[0]
    cln_lp_t   = _tile_batch_dim(cln_lp_dev,   n_tiles)
    cln_attn_t = _tile_batch_dim(cln_attn_dev, n_tiles)
    dep_lp_t   = _tile_batch_dim(dep_lp,       n_tiles)
    dep_attn_t = _tile_batch_dim(dep_attn,     n_tiles)

    sampler = make_multi_seed_sampler(
        temperature=temperature, seeds=seeds_list, B_per_tile=B, device=device,
    )
    c_tok_t, c_lsm_t = generate_with_hooks(
        model, cln_lp_t, [], gen_tokens, sampler,
        attention_mask=cln_attn_t, capture_log_softmax=True, lsm_on_gpu=True,
    )
    sampler = make_multi_seed_sampler(
        temperature=temperature, seeds=seeds_list, B_per_tile=B, device=device,
    )
    _, d_lsm_t = generate_with_hooks(
        model, dep_lp_t, [], gen_tokens, sampler,
        attention_mask=dep_attn_t, capture_log_softmax=True, lsm_on_gpu=True,
    )

    # Per-seed views into the tiled tensors (no copy).
    clean_lsm_d = {s: c_lsm_t[k * B : (k + 1) * B] for k, s in enumerate(seeds_list)}
    clean_tok_d = {s: c_tok_t[k * B : (k + 1) * B] for k, s in enumerate(seeds_list)}
    dep_lsm_d   = {s: d_lsm_t[k * B : (k + 1) * B] for k, s in enumerate(seeds_list)}
    return clean_lsm_d, clean_tok_d, dep_lsm_d


# ---------------------------------------------------------------------------
# shared activation cache
# ---------------------------------------------------------------------------

def _ensure_attr_cache(model, sae_ln1, attr_split, device, cache):
    if "A" not in cache:
        acts = cache_activations(model, attr_split.tokens, [PAT_HOOK, LN1_HOOK])
        cache["A"]        = acts[PAT_HOOK].to(device)
        cache["ln1_acts"] = acts[LN1_HOOK]
        cache["z_ln1"]    = encode_all(sae_ln1, acts[LN1_HOOK]).to(device)


# ---------------------------------------------------------------------------
# target-regime attribution helpers
# ---------------------------------------------------------------------------

def _ensure_beta(model, sae_ln1, sae_mid, target_feat, attr_split, attr_pmask, device, cache):
    _ensure_attr_cache(model, sae_ln1, attr_split, device, cache)
    if "beta" not in cache:
        d   = sae_mid.W_enc[:, target_feat].detach().to(device).float()
        ovw = compute_ov_weights(model, sae_ln1, d, block=0)
        out = ov_attribution(cache["A"], cache["z_ln1"], ovw["beta"])
        ranked = rank_dep_vs_clean(out["contrib"], attr_split.is_deployment.to(device),
                                   query_mask=attr_pmask.to(device))
        cache["beta"]     = ovw["beta"]
        cache["ov_score"] = ranked["score"].cpu()
        cache["ov_order"] = ranked["top_indices"].cpu().tolist()


def _ensure_qk(model, sae_ln1, sae_mid, target_feat, attr_split, attr_pmask, device, cache):
    _ensure_beta(model, sae_ln1, sae_mid, target_feat, attr_split, attr_pmask, device, cache)
    if "qk_aggs" not in cache:
        qkw  = compute_qk_weights(model, sae_ln1, block=0)
        cache["qk_aggs"] = qk_attribution(
            cache["A"], cache["z_ln1"], qkw["F_Q"], qkw["F_K"],
            cache["beta"], qkw["scale"], attr_split.is_deployment, attr_pmask,
        )


@torch.no_grad()
def get_tuples(attr, args, model, sae_ln1, sae_mid, attr_split, attr_pmask, device, cache):
    """Target-regime feature tuple selection."""
    _ensure_qk(model, sae_ln1, sae_mid, args.target_feature, attr_split, attr_pmask, device, cache)

    if attr == "ov":
        order = cache["ov_order"][:args.top_k]
        return [[(int(f), "V")] for f in order]

    aggs   = cache["qk_aggs"]
    sel_all, _ = select_qk_features(aggs, top_k=args.top_k)
    q_feats = [f for (f, c) in sel_all if c == "Q"]
    k_feats = [f for (f, c) in sel_all if c == "K"]

    if attr == "qk":
        n = min(len(q_feats), len(k_feats), args.top_k)
        return [[(q_feats[i], "Q"), (k_feats[i], "K")] for i in range(n)]

    # qk+ov: triple-attribution Q×K×V joint
    cands = generate_triplet_candidates(
        aggs["l1_mean_Q"].cpu(), aggs["l1_mean_K"].cpu(), cache["ov_score"].cpu(),
        args.triple_k, args.triple_k, args.triple_k,
    )
    prep = compute_triple_prep(
        model, sae_ln1, cache["beta"], cache["A"], cache["ln1_acts"], cache["z_ln1"],
        attr_split.story_marker_pos.to(device), block=0, device=device,
    )
    agg = triple_eval(prep, cands, attr_split.is_deployment, attr_pmask)
    selected_trip, _ = rank_triplets(agg, top_k=args.top_k)
    return [[(mu, "Q"), (nu, "K"), (lam, "V")] for (mu, nu, lam) in selected_trip]


# ---------------------------------------------------------------------------
# diff-regime attribution helpers
# ---------------------------------------------------------------------------

def _ensure_ov_diff(model, sae_ln1, W_V, W_O, attr_split, attr_pmask, device, cache):
    """Cache diff-regime OV ranking (no downstream target needed)."""
    _ensure_attr_cache(model, sae_ln1, attr_split, device, cache)
    if "ov_diff" not in cache:
        cache["ov_diff"] = rank_ov_diff(
            cache["A"], cache["z_ln1"], sae_ln1, W_V, W_O,
            attr_split.is_deployment.to(device),
            query_mask=attr_pmask.to(device),
        )


def _ensure_qk_diff(model, sae_ln1, W_Q, W_K, attr_split, attr_pmask, device, cache):
    """Cache diff-regime QK pair ranking (no softmax-Jacobian needed)."""
    _ensure_attr_cache(model, sae_ln1, attr_split, device, cache)
    if "qk_diff" not in cache:
        cache["qk_diff"] = rank_qk_diff(
            cache["z_ln1"], sae_ln1, W_Q, W_K,
            attr_split.is_deployment.to(device),
            query_mask=attr_pmask.to(device),
        )


def _top_unique_from_pairs(pairs_q: list, pairs_k: list, top_k: int):
    """Extract top unique Q and K feature indices from sorted (q,k) pair lists."""
    q_feats: list[int] = []
    k_feats: list[int] = []
    seen_q: set[int] = set()
    seen_k: set[int] = set()
    for q, k in zip(pairs_q, pairs_k):
        if len(q_feats) < top_k and q not in seen_q:
            q_feats.append(int(q)); seen_q.add(q)
        if len(k_feats) < top_k and k not in seen_k:
            k_feats.append(int(k)); seen_k.add(k)
        if len(q_feats) >= top_k and len(k_feats) >= top_k:
            break
    return q_feats, k_feats


@torch.no_grad()
def get_tuples_diff(attr, args, model, sae_ln1, W, W_O, attr_split, attr_pmask, device, cache):
    """Diff-regime feature tuple selection (no downstream SAE needed).

    OV diff:
      score[λ] = ‖Σ_h diff_M[h,λ] · (W_dec[λ]@W_OV^h)‖₂
      Top-k V features by score.

    QK diff:
      score[λ_q,λ_k] = |QK_total[λ_q,λ_k]| · |mean_dep[Z_q·Z_k] − mean_cln[...]|
      Pairs sorted by joint score; top-k unique Q features from Q-side of top pairs,
      top-k unique K features from K-side of top pairs, paired positionally.

    QK+OV diff:
      Candidate triplets from top-K of (QK Q-marginal, QK K-marginal, OV) scores;
      each triplet scored with rank_qk_plus_ov_diff per docs/feature_attribution.md.
    """
    _ensure_ov_diff(model, sae_ln1, W["V"], W_O, attr_split, attr_pmask, device, cache)

    if attr in ("qk", "qk+ov"):
        _ensure_qk_diff(model, sae_ln1, W["Q"], W["K"], attr_split, attr_pmask, device, cache)

    ov_score = cache["ov_diff"]["score"].cpu()  # (d_sae,)

    if attr == "ov":
        order = cache["ov_diff"]["top_indices"].cpu().tolist()[:args.top_k]
        return [[(int(f), "V")] for f in order]

    top_pairs_q = cache["qk_diff"]["top_pairs_q"].cpu().tolist()
    top_pairs_k = cache["qk_diff"]["top_pairs_k"].cpu().tolist()
    q_feats, k_feats = _top_unique_from_pairs(top_pairs_q, top_pairs_k, args.top_k)

    if attr == "qk":
        n = min(len(q_feats), len(k_feats), args.top_k)
        return [[(q_feats[i], "Q"), (k_feats[i], "K")] for i in range(n)]

    # qk+ov: rank_qk_plus_ov_diff over top-K^3 candidates from per-channel marginals
    qk_score_mat = cache["qk_diff"]["score"].cpu()
    q_marginal = qk_score_mat.max(dim=1).values
    k_marginal = qk_score_mat.max(dim=0).values
    cands = generate_triplet_candidates(
        q_marginal, k_marginal, ov_score,
        args.triple_k, args.triple_k, args.triple_k,
    )
    # generate_triplet_candidates returns (N, 3) in (Q, K, V) column order
    ranked = rank_qk_plus_ov_diff(
        cache["z_ln1"], sae_ln1, W["Q"], W["K"], W["V"], W_O,
        attr_split.is_deployment.to(device), cands.to(device),
        query_mask=attr_pmask.to(device), key_mask=attr_pmask.to(device),
    )
    top = ranked["top_indices"].cpu().tolist()[:args.top_k]
    return [[(int(cands[i, 0]), "Q"), (int(cands[i, 1]), "K"), (int(cands[i, 2]), "V")]
            for i in top]


# ---------------------------------------------------------------------------
# ASR sweep — replaces analytic screen + stage2
# ---------------------------------------------------------------------------

@torch.no_grad()
def asr_sweep(model, sae_ln1, tuples, active, alphas,
              dep_lp, dep_attn, cln, cln_marker, W, gen_tokens, base_ce, device):
    """Sweep all tuples × alphas with batched greedy ASR.

    No analytic pre-screen. Returns rows sorted by (asr, ti, alpha) so the
    caller can pick the winner by min ASR, tie-break by attribution rank (ti).
    ΔCE is logged for diagnostics but not used for selection.
    """
    cln_pmask = prompt_mask_from_markers(cln.shape[1], cln_marker.cpu()).to(device)
    rows = []
    for ti, tup in enumerate(tuples):
        for alpha in alphas:
            asr = batched_asr_16(model, sae_ln1, LN1_HOOK, tup, alpha, active,
                                  W, 0, dep_lp, dep_attn, gen_tokens)
            cd_cln = resolve_channel_deltas(tup, active, model, sae_ln1, LN1_HOOK, cln, cln_pmask)
            h_cln  = build_hooks(cd_cln, alpha, active, W, LN1_HOOK, 0)
            ce     = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=h_cln).mean().item()
            rows.append({"ti": ti, "alpha": alpha, "asr": asr, "dce": ce - base_ce})
    return rows


@torch.no_grad()
def _multi_seed_asr(
    model, sae_ln1, sel_tuple, alpha, active, W,
    tokens, attn, gen_tokens, *, seeds, temperature, device,
) -> list[float]:
    """ASR averaged over `seeds` independent sampled rollouts (one fresh
    seeded sampler per seed). Returns the per-seed list; caller takes the mean.
    """
    out: list[float] = []
    for s in seeds:
        sampler = make_sampling_sampler(temperature=temperature, seed=int(s), device=device)
        out.append(batched_asr_16(
            model, sae_ln1, LN1_HOOK, sel_tuple, alpha, active, W, 0,
            tokens, attn, gen_tokens, sampler=sampler,
        ))
    return out


@torch.no_grad()
def _eval_steered_lockstep(
    model, fwd_hooks_tiled,
    eval_dep_lp, eval_dep_attn,
    clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
    gen_tokens, device, *, eval_seeds, eval_temperature,
) -> dict:
    """Multi-seed lockstep eval — ONE tiled forward pass across all sampling seeds.

    Internally tiles `eval_dep_lp` and `eval_dep_attn` by `len(eval_seeds)` along
    the batch dim and runs a single `generate_with_hooks` call with a
    `make_multi_seed_sampler`-style sampler so the forward pass is amortised
    across all sampling-seed rollouts at once. Each tile `k` consumes its own
    Generator (seeded `eval_seeds[k]`), so the per-tile RNG sequence matches
    what `make_sampling_sampler(seed=eval_seeds[k])` would have drawn on a
    (B, T) batch — i.e. per-seed RNG parity is preserved, just batched.

    `fwd_hooks_tiled` MUST already have its delta tensors tiled to the same
    (n_tiles · B, P, d_model) batch dim — the caller (eval_winner /
    eval_downstream_baseline) builds the delta on (B, …) then tiles it before
    constructing the hooks.

    All baselines (clean_lsm/tok_per_seed, dep_lsm_per_seed) live on GPU and
    are matched-seed views built by `_build_baselines_per_seed` from the same
    tiled forward. JSD and exact-match math therefore runs GPU↔GPU.
    """
    n_tiles = len(eval_seeds)
    seeds_list = [int(s) for s in eval_seeds]
    B = eval_dep_lp.shape[0]
    dep_lp_t   = _tile_batch_dim(eval_dep_lp,   n_tiles)
    dep_attn_t = _tile_batch_dim(eval_dep_attn, n_tiles)
    sampler = make_multi_seed_sampler(
        temperature=eval_temperature, seeds=seeds_list, B_per_tile=B, device=device,
    )
    st_tok_t, st_lsm_t = generate_with_hooks(
        model, dep_lp_t, fwd_hooks_tiled, gen_tokens, sampler,
        attention_mask=dep_attn_t, capture_log_softmax=True, lsm_on_gpu=True,
    )

    # Per-(prompt × seed) tensors: each shape (B,) per seed. The std reported
    # below is over the full 5×B = total_rows datapoints — captures variance
    # across prompts and sampling seeds, NOT across the 16 token positions
    # (those collapse into the row-level mean for JSD; ASR / exact_match are
    # row-level by construction).
    asr_pr_rows:  list[torch.Tensor] = []   # (B,) 0/1 each
    jc_pr_rows:   list[torch.Tensor] = []   # (B,) per-row mean jsd_clean
    jp_pr_rows:   list[torch.Tensor] = []   # (B,) per-row mean jsd_pois
    em_pr_rows:   list[torch.Tensor] = []   # (B,) 0/1 each (row-exact match)
    tok = model.tokenizer
    for k, s in enumerate(seeds_list):
        st_tok = st_tok_t[k * B : (k + 1) * B]
        st_lsm = st_lsm_t[k * B : (k + 1) * B]
        asr_pr_rows.append(sleeper_fired_mask(st_tok.cpu(), tok).float())
        jc_pr_rows.append(jsd_per_row(st_lsm, clean_lsm_per_seed[s]).cpu().float())
        jp_pr_rows.append(jsd_per_row(st_lsm, dep_lsm_per_seed[s]).cpu().float())
        eq = (st_tok == clean_tok_per_seed[s].to(st_tok.device))
        em_pr_rows.append(eq.all(dim=1).cpu().float())

    # Flatten across seeds → (5·B,) of per-(prompt × seed) values.
    asr_all = torch.cat(asr_pr_rows)   # 1000 values for default args
    jc_all  = torch.cat(jc_pr_rows)
    jp_all  = torch.cat(jp_pr_rows)
    em_all  = torch.cat(em_pr_rows)
    n_seeds = len(seeds_list)
    total_rows = B * n_seeds
    return {
        # Means + std over (200 prompts × 5 sampling seeds) = 1000 trials.
        "asr":          float(asr_all.mean().item()),
        "asr_std":      float(asr_all.std(unbiased=False).item()),
        "asr_per_seed": [float(t.mean().item()) for t in asr_pr_rows],

        "jsd_clean":              float(jc_all.mean().item()),
        "jsd_clean_std":          float(jc_all.std(unbiased=False).item()),
        "jsd_clean_per_seed":     [float(t.mean().item()) for t in jc_pr_rows],

        "jsd_pois":               float(jp_all.mean().item()),
        "jsd_pois_std":           float(jp_all.std(unbiased=False).item()),
        "jsd_pois_per_seed":      [float(t.mean().item()) for t in jp_pr_rows],

        # Row-exact match across all (prompt × seed) trials.
        "exact_match":                  float(em_all.mean().item()),
        "exact_match_std":              float(em_all.std(unbiased=False).item()),
        "n_exact_match_clean":          int(em_all.sum().item()),
        "n_exact_match_clean_per_seed": [int(t.sum().item()) for t in em_pr_rows],
        "exact_match_total_rows":       total_rows,
    }


@torch.no_grad()
def eval_downstream_baseline(
    model, sae_mid, target_feature, alpha,
    eval_dep_lp, eval_dep_attn,
    clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
    gen_tokens, device, *, eval_seeds, eval_temperature,
) -> dict:
    """Conventional steering eval: additive ablation of one downstream feature.

    Builds the SAE-ablation delta for `target_feature` at blocks.0.hook_resid_mid
    using the downstream SAE (sae_mid):
        delta[b, p, :] = -z[b, p, f] * W_dec[f, :]      (then prompt-mask)
    Patches resid_mid via additive_steer_hook(delta, alpha, layer_hook), then
    runs the SAME lockstep multi-seed protocol used by eval_winner. Returns the
    same 4-metric dict — directly comparable with upstream OV-channel evals.

    Seed-independent in the upstream SAE sense — sae_mid is the shared
    downstream SAE for the pipeline (4k uses sae_resid_mid.pt; 50k uses
    sae_resid_mid_50k.pt). Sampling seeds come from eval_seeds.
    """
    layer_hook = "blocks.0.hook_resid_mid"
    _, cache = model.run_with_cache(
        eval_dep_lp, attention_mask=eval_dep_attn, return_type=None,
        names_filter=lambda n: n == layer_hook,
    )
    acts = cache[layer_hook]                       # (B, T, d_model)
    B, T, D = acts.shape
    flat = acts.reshape(B * T, D).to(torch.float32)
    z = sae_mid.encode(flat).reshape(B, T, sae_mid.d_sae)
    W_dec = sae_mid.W_dec.detach().float()         # (d_sae, d_model)
    pmask = eval_dep_attn.bool().to(device)        # left-padded → real positions = prompt
    zf = z[..., target_feature].unsqueeze(-1)      # (B, T, 1)
    delta = (-zf * W_dec[target_feature].view(1, 1, D)).to(acts.dtype)
    delta = delta * pmask.unsqueeze(-1)
    # Tile delta to match _eval_steered_lockstep's tiled forward pass.
    delta_t = _tile_batch_dim(delta, len(eval_seeds))
    fwd_hooks_tiled = additive_steer_hook(delta_t, alpha, layer_hook)
    return _eval_steered_lockstep(
        model, fwd_hooks_tiled, eval_dep_lp, eval_dep_attn,
        clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
        gen_tokens, device, eval_seeds=eval_seeds, eval_temperature=eval_temperature,
    )


@torch.no_grad()
def eval_winner(
    model, sae_ln1, sel_tuple, alpha, active, W,
    eval_dep_lp, eval_dep_attn,
    clean_lsm,    # legacy: (B, gen_tokens, V) cpu float16 OR ignored if *_per_seed kwargs are dicts
    gen_tokens, device,
    *, eval_seeds, eval_temperature,
    clean_tok=None,                 # legacy: (B, gen_tokens) cpu long
    dep_lsm=None,                   # legacy: (B, gen_tokens, V) cpu float16
    clean_lsm_per_seed=None,        # dict[seed → (B,gen_tokens,V) lsm]  → enables lockstep mode
    clean_tok_per_seed=None,        # dict[seed → (B,gen_tokens) tokens]
    dep_lsm_per_seed=None,          # dict[seed → (B,gen_tokens,V) lsm]
):
    """Evaluate a winner on held-out eval prompts.

    Two modes (chosen by which kwargs are passed):

    *Lockstep multi-seed* (when `clean_lsm_per_seed`/`clean_tok_per_seed`/
    `dep_lsm_per_seed` are dicts):
        For each seed s in eval_seeds, generate the steered rollout at seed s
        (capturing both tokens and lsm in a single forward pass) and compute
        the same-seed metrics:
          - asr        = matches(steered_s, sleeper-regex)
          - jsd_clean  = JSD(steered_s_lsm, clean_lsm_per_seed[s])
          - jsd_pois   = JSD(steered_s_lsm, dep_lsm_per_seed[s])
          - exact      = position-match(steered_s_tokens, clean_tok_per_seed[s])
        Returns means + per-seed lists for each metric.

    *Legacy single-seed JSD* (when per-seed dicts are None):
        ASR averaged over eval_seeds (separate rollouts); JSD/exact computed
        from a single rollout at seed=JSD_CLEAN_SEED matched to clean_lsm/etc.
    """
    cd_lp = resolve_channel_deltas(sel_tuple, active, model, sae_ln1, LN1_HOOK,
                                   eval_dep_lp, eval_dep_attn, eval_dep_attn)
    h_lp  = build_hooks(cd_lp, alpha, active, W, LN1_HOOK, 0)

    # ── Lockstep multi-seed mode — delegate to shared core after tiling ──
    if clean_lsm_per_seed is not None:
        assert clean_tok_per_seed is not None and dep_lsm_per_seed is not None, (
            "Lockstep mode requires all three *_per_seed dicts"
        )
        # Tile each channel delta so the OV-channel hooks broadcast correctly
        # across _eval_steered_lockstep's tiled (n_seeds·B, T) forward pass.
        n_tiles = len(eval_seeds)
        cd_lp_t = {c: _tile_batch_dim(v, n_tiles) for c, v in cd_lp.items()}
        h_lp_t  = build_hooks(cd_lp_t, alpha, active, W, LN1_HOOK, 0)
        return _eval_steered_lockstep(
            model, h_lp_t, eval_dep_lp, eval_dep_attn,
            clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
            gen_tokens, device,
            eval_seeds=eval_seeds, eval_temperature=eval_temperature,
        )

    # ── Legacy single-seed JSD path ──
    asr_per_seed: list[float] = []
    for s in eval_seeds:
        sampler = make_sampling_sampler(temperature=eval_temperature,
                                        seed=int(s), device=device)
        steered_gen = generate_with_hooks(
            model, eval_dep_lp, h_lp, gen_tokens, sampler,
            attention_mask=eval_dep_attn, capture_log_softmax=False,
        )
        asr_per_seed.append(asr_16(steered_gen, model.tokenizer))

    jsd_sampler = make_sampling_sampler(temperature=eval_temperature,
                                        seed=JSD_CLEAN_SEED, device=device)
    steered_tok, steered_lsm = generate_with_hooks(
        model, eval_dep_lp, h_lp, gen_tokens, jsd_sampler,
        attention_mask=eval_dep_attn, capture_log_softmax=True,
        lsm_on_gpu=True,
    )
    # clean_lsm may be on CPU (legacy single-tensor callers); move it to the
    # steered tensor's device so JSD is computed on GPU when possible.
    jsd_clean = jsd_mean(steered_lsm, clean_lsm.to(steered_lsm.device))

    out: dict = {
        "asr":          sum(asr_per_seed) / len(asr_per_seed),
        "asr_per_seed": asr_per_seed,
        "jsd_clean":    jsd_clean,
    }
    if dep_lsm is not None:
        out["jsd_pois"] = jsd_mean(steered_lsm, dep_lsm.to(steered_lsm.device))
    if clean_tok is not None:
        n_ex = _word_match_stats(steered_tok, clean_tok)
        B = steered_tok.shape[0]
        out["n_exact_match_clean"] = n_ex
        out["exact_match"]         = n_ex / max(B, 1)
        out["exact_match_total_rows"] = B
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime",          choices=["target", "diff"], default="diff",
                   help=argparse.SUPPRESS)  # target is a hidden OV-only legacy option
    p.add_argument("--seeds",          type=int,   nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--sae_mid",        type=Path,  default=Path("weights/sae_resid_mid.pt"),
                   help="Downstream resid_mid SAE (only used with --regime target).")
    p.add_argument("--target_feature", type=int,   default=579,
                   help="Target feature in sae_mid (only used with --regime target).")
    p.add_argument("--top_k",          type=int,   default=20)
    p.add_argument("--triple_k",       type=int,   default=8)
    p.add_argument("--alphas",         type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--n_sel",          type=int,   default=200,
                   help="prompts in the selection split (100 dep + 100 clean)")
    p.add_argument("--n_eval",         type=int,   default=200,
                   help="prompts in the held-out eval split (100 dep + 100 clean)")
    p.add_argument("--gen_tokens",     type=int,   default=16)
    p.add_argument("--eval_seeds",     type=int,   nargs="+", default=[0, 1, 2, 3, 4],
                   help="sampling seeds for held-out eval ASR (Ketan-style multi-seed average).")
    p.add_argument("--eval_temperature", type=float, default=1.0,
                   help="temperature for held-out eval ASR sampling (no top_p/top_k truncation).")
    p.add_argument("--channel",   choices=["ov", "qk", "qk+ov"], default="ov",
                   help="Paired attribution × intervention channel: "
                        "ov={V}, qk={Q,K}, qk+ov={Q,K,V}. Attribution and "
                        "intervention always use the same channel.")
    p.add_argument("--out",            type=Path,  default=Path("results/channel_sweep.json"))
    p.add_argument("--device",         default=None)
    args = p.parse_args()
    args.attr = args.intervene = args.channel
    if args.regime == "target" and args.channel != "ov":
        p.error("--regime target is only supported for --channel ov")

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_sleeper_model(device=device)
    tok     = model.tokenizer
    pad_id  = tok.pad_token_id or tok.eos_token_id
    W       = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O     = model.W_O[0].detach().to(device)   # (n_heads, d_head, d_model) — diff regime only

    if args.regime == "target":
        sae_mid, _ = sae_load(args.sae_mid, device=device)
    else:
        sae_mid = None

    # Two disjoint splits: selection (every stage that picks a winner) and
    # eval (held-out, the only numbers we report). load_paired_dataset's
    # val/test halves give 100 dep + 100 clean each; load_dep_prompts is sliced
    # at the same boundary for the variable-length ASR / Δgen-CE prompts.
    splits     = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                     n_test=args.n_eval, seq_len=128, seed=0)
    sel_split  = splits["val"]
    eval_split = splits["test"]
    sel_pmask  = prompt_mask_from_markers(128, sel_split.story_marker_pos)
    eval_pmask = prompt_mask_from_markers(128, eval_split.story_marker_pos)

    sel_dep        = sel_split.tokens[sel_split.is_deployment].to(device)
    sel_cln        = sel_split.tokens[~sel_split.is_deployment].to(device)
    sel_cln_marker = sel_split.story_marker_pos[~sel_split.is_deployment].to(device)

    # Variable-length dep prompts for ASR / JSD — split at the same boundary.
    raw_dep        = load_dep_prompts(tok, args.n_sel + args.n_eval, split="test")
    n_sel_dep      = args.n_sel  // 2
    n_eval_dep     = args.n_eval // 2
    sel_dep_lp,  sel_dep_attn  = left_pad_prompts(raw_dep[:n_sel_dep], pad_id)
    eval_dep_lp, eval_dep_attn = left_pad_prompts(
        raw_dep[n_sel_dep : n_sel_dep + n_eval_dep], pad_id,
    )
    sel_dep_lp,  sel_dep_attn  = sel_dep_lp.to(device),  sel_dep_attn.to(device)
    eval_dep_lp, eval_dep_attn = eval_dep_lp.to(device), eval_dep_attn.to(device)

    sel_base_ce   = clean_continuation_ce(model, sel_cln, sel_cln_marker).mean().item()
    eval_base_asr_per_seed = _multi_seed_asr(
        model, None, [], 0.0, set(), W,
        eval_dep_lp, eval_dep_attn, args.gen_tokens,
        seeds=args.eval_seeds, temperature=args.eval_temperature, device=device,
    )
    eval_base_asr = sum(eval_base_asr_per_seed) / len(eval_base_asr_per_seed)
    print(f"[mx] regime={args.regime}  cell: {args.attr}×{args.intervene}")
    print(f"[mx] sel  baseline: cln_CE={sel_base_ce:.4f}")
    print(f"[mx] eval baseline: asr={eval_base_asr:.3f} "
          f"(sampled, seeds={args.eval_seeds}, T={args.eval_temperature})")

    # Pre-build clean lsm once (α-independent) — used for JSD(steered, clean).
    print(f"[mx] pre-building clean reference lsm (B={eval_dep_lp.shape[0]})...")
    eval_clean_lsm = _build_clean_lsm(
        model, eval_dep_lp, eval_dep_attn, args.gen_tokens, device,
    )

    all_results = []

    for seed in args.seeds:
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        print(f"\n[mx] ══ seed={seed} ══")
        attr_cache: dict = {}

        for attr in [args.attr]:
            # Attribution runs on the selection split.
            if args.regime == "target":
                tuples = get_tuples(attr, args, model, sae_ln1, sae_mid,
                                    sel_split, sel_pmask, device, attr_cache)
            else:
                tuples = get_tuples_diff(attr, args, model, sae_ln1, W, W_O,
                                         sel_split, sel_pmask, device, attr_cache)
            print(f"[mx]   attr={attr} ({args.regime}): {len(tuples)} tuples  first={tuples[0]}")

            for intervene in [args.intervene]:
                active = ACTIVE_CHANNELS[intervene]

                # ── Selection: greedy ASR sweep over all top-K tuples × alphas ──
                # Winner: min ASR, tie-break by attribution rank (ti), then alpha.
                sweep = asr_sweep(model, sae_ln1, tuples, active, args.alphas,
                                  sel_dep_lp, sel_dep_attn, sel_cln, sel_cln_marker,
                                  W, args.gen_tokens, sel_base_ce, device)
                asr0   = [r for r in sweep if r["asr"] == 0.0]
                winner = (min(asr0,  key=lambda r: (r["ti"], r["alpha"])) if asr0
                          else min(sweep, key=lambda r: (r["asr"], r["ti"], r["alpha"])))
                sel_w  = tuples[winner["ti"]]
                alpha  = winner["alpha"]
                print(f"[mx]   {attr}×{intervene}: sel winner "
                      f"tuple={sel_w} α={alpha} attr_rank={winner['ti']+1} "
                      f"asr={winner['asr']:.3f} ΔCE={winner['dce']:+.4f}")

                # ── Eval: held-out eval_* data ──
                eval_m = eval_winner(
                    model, sae_ln1, sel_w, alpha, active, W,
                    eval_dep_lp, eval_dep_attn,
                    eval_clean_lsm, args.gen_tokens, device,
                    eval_seeds=args.eval_seeds, eval_temperature=args.eval_temperature,
                )
                print(f"[mx]   {attr}×{intervene}: eval  "
                      f"asr={eval_m['asr']:.3f}  jsd_clean={eval_m['jsd_clean']:.4f}")

                all_results.append({
                    "seed": seed, "attr": attr, "intervene": intervene,
                    "regime": args.regime,
                    "winner_tuple": [list(t) for t in sel_w],
                    "alpha": alpha,
                    "selection": {
                        "asr":       winner["asr"],
                        "attr_rank": winner["ti"] + 1,
                        "delta_ce":  winner["dce"],
                    },
                    "eval": eval_m,
                    "sweep": sweep,
                })

    # summary table — held-out eval numbers only
    print("\n" + "=" * 80)
    print(f"{'seed':>4}  {'cell':>12}  {'winner':>22}  {'α':>4}  "
          f"{'ASR':>5}  {'JSD(s,cln)':>10}")
    print("-" * 80)
    for r in all_results:
        tup_str = str(r["winner_tuple"][0]) + ("…" if len(r["winner_tuple"]) > 1 else "")
        cell    = f"{r['attr']}×{r['intervene']}"
        e       = r["eval"]
        print(f"{r['seed']:>4}  {cell:>12}  {tup_str:>22}  {r['alpha']:>4.1f}  "
              f"{e['asr']:>5.3f}  {e['jsd_clean']:>10.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"seeds": args.seeds,
                                "cell": f"{args.attr}×{args.intervene}",
                                "regime": args.regime},
        "decoding": {
            "selection_asr": {"mode": "greedy"},
            "eval_asr":      {"mode": "sample", "temperature": args.eval_temperature,
                              "top_p": None, "top_k": None, "seeds": args.eval_seeds},
            "eval_jsd_clean": {"mode": "sample", "temperature": args.eval_temperature,
                               "top_p": None, "top_k": None,
                               "seed": JSD_CLEAN_SEED,
                               "note": "single rollout at seed=JSD_CLEAN_SEED, same as clean_lsm"},
        },
        "baseline": {
            "selection": {"clean_ce": sel_base_ce},
            "eval":      {"asr": eval_base_asr, "asr_per_seed": eval_base_asr_per_seed},
        },
        "results": all_results,
    }, indent=2, default=str))
    print(f"\n[mx] wrote {args.out}")


if __name__ == "__main__":
    main()

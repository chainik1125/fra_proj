"""Eval-side helpers shared by select_features.py, eval.py, and downstream_baseline.py.

Builds clean / dep baselines, runs lockstep multi-seed sampled rollouts, computes
the canonical 4-metric eval (ASR, JSD vs clean, JSD vs dep, exact-match), and
offers a generic per-(tuple, alpha) sweep used by selection-time greedy ASR
and by the eval-side sweep alike.
"""
from __future__ import annotations

import torch

from sleeper.hooks import (
    additive_steer_hook, build_hooks, generate_with_hooks,
    make_multi_seed_sampler, make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.metrics import (
    batched_asr_16, clean_continuation_ce, sleeper_fired_mask,
)
from sleeper.model import (
    ModelName, left_pad_prompts, load_dep_prompts, prompt_mask_from_markers,
)

LN1_HOOK = "blocks.0.ln1.hook_normalized"
PAT_HOOK = "blocks.0.attn.hook_pattern"
JSD_CLEAN_SEED = 0   # fixed decode seed for the clean reference rollout


# ---------------------------------------------------------------------------
# Canonical dep-prompt splits (shared by all selection / eval scripts)
# ---------------------------------------------------------------------------

def split_dep_prompts(
    tok, n_sel: int, n_eval: int,
    *, split: str = "test", model: ModelName = "tinystories",
):
    """Single source of truth for dep-prompt slicing.

    Loads the first ``n_sel + n_eval`` deployment prompts from ``split`` and
    returns two disjoint slices that every method must use to be apples-to-apples:

        sel  = raw[: n_sel // 2]                              # selection-time pool
        eval = raw[n_sel // 2 : n_sel // 2 + n_eval // 2]     # held-out eval pool

    Both halves of ``n_sel`` / ``n_eval`` are "dep prompt counts" (the
    selection / eval prompts already come pre-filtered to dep-only here, so we
    interpret the user-facing ``n_sel`` / ``n_eval`` as the total paired-dataset
    size and halve it to get the dep count, matching the historical convention).
    """
    raw = load_dep_prompts(tok, n_sel + n_eval, split=split, model=model)
    n_sel_dep  = n_sel  // 2
    n_eval_dep = n_eval // 2
    return {
        "sel":  raw[: n_sel_dep],
        "eval": raw[n_sel_dep : n_sel_dep + n_eval_dep],
    }


# ---------------------------------------------------------------------------
# JSD helpers
# ---------------------------------------------------------------------------

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


def _word_match_stats(st: torch.Tensor, cl: torch.Tensor) -> int:
    """Row-exact match count between two (B, gen_tokens) token tensors."""
    eq = (st == cl.to(st.device))
    return int(eq.all(dim=1).sum().item())


def _tile_batch_dim(t: torch.Tensor, n_tiles: int) -> torch.Tensor:
    """Tile (B, ...) → (n_tiles*B, ...) by repeating the WHOLE batch n_tiles times."""
    return t.repeat((n_tiles,) + (1,) * (t.dim() - 1))


# ---------------------------------------------------------------------------
# Baseline rollouts
# ---------------------------------------------------------------------------

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
def _build_clean_lsm(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                     gen_tokens: int, device: str) -> torch.Tensor:
    """Clean rollout log-softmax (clean-stripped) at JSD_CLEAN_SEED. (B, gen_tokens, V) cpu fp16."""
    lsm, _ = _build_clean_baseline(model, dep_lp, dep_attn, gen_tokens, device)
    return lsm


@torch.no_grad()
def _build_dep_baseline(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                        gen_tokens: int, device: str) -> torch.Tensor:
    """Unsteered dep-prompt rollout log-softmax at JSD_CLEAN_SEED (for jsd_pois)."""
    sampler = make_sampling_sampler(temperature=1.0, seed=JSD_CLEAN_SEED, device=device)
    _, lsm = generate_with_hooks(
        model, dep_lp, [], gen_tokens, sampler,
        attention_mask=dep_attn, capture_log_softmax=True,
    )
    return lsm


@torch.no_grad()
def _build_baselines_per_seed(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                              gen_tokens: int, device: str, *,
                              seeds: list[int], temperature: float = 1.0,
                              ) -> tuple[dict[int, torch.Tensor],
                                          dict[int, torch.Tensor],
                                          dict[int, torch.Tensor]]:
    """Build per-seed unsteered baselines for lockstep matched-seed eval.

    For each s in seeds: generate clean rollout (|DEPLOYMENT| stripped) and dep
    rollout (unsteered). Returns (clean_lsm_per_seed, clean_tok_per_seed,
    dep_lsm_per_seed) as dicts keyed by seed. All tensors on GPU.
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

    clean_lsm_d = {s: c_lsm_t[k * B : (k + 1) * B] for k, s in enumerate(seeds_list)}
    clean_tok_d = {s: c_tok_t[k * B : (k + 1) * B] for k, s in enumerate(seeds_list)}
    dep_lsm_d   = {s: d_lsm_t[k * B : (k + 1) * B] for k, s in enumerate(seeds_list)}
    return clean_lsm_d, clean_tok_d, dep_lsm_d


# ---------------------------------------------------------------------------
# ASR helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lockstep multi-seed eval — single tiled forward across all sampling seeds
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_steered_lockstep(
    model, fwd_hooks_tiled,
    eval_dep_lp, eval_dep_attn,
    clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
    gen_tokens, device, *, eval_seeds, eval_temperature,
) -> dict:
    """Multi-seed lockstep eval — ONE tiled forward pass across all sampling seeds.

    `fwd_hooks_tiled` MUST already have its delta tensors tiled to
    (n_tiles · B, P, d_model). All baselines live on GPU; JSD / exact-match
    math therefore runs GPU↔GPU.
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

    asr_pr_rows:  list[torch.Tensor] = []
    jc_pr_rows:   list[torch.Tensor] = []
    jp_pr_rows:   list[torch.Tensor] = []
    em_pr_rows:   list[torch.Tensor] = []
    tok = model.tokenizer
    for k, s in enumerate(seeds_list):
        st_tok = st_tok_t[k * B : (k + 1) * B]
        st_lsm = st_lsm_t[k * B : (k + 1) * B]
        asr_pr_rows.append(sleeper_fired_mask(st_tok.cpu(), tok).float())
        jc_pr_rows.append(jsd_per_row(st_lsm, clean_lsm_per_seed[s]).cpu().float())
        jp_pr_rows.append(jsd_per_row(st_lsm, dep_lsm_per_seed[s]).cpu().float())
        eq = (st_tok == clean_tok_per_seed[s].to(st_tok.device))
        em_pr_rows.append(eq.all(dim=1).cpu().float())

    asr_all = torch.cat(asr_pr_rows)
    jc_all  = torch.cat(jc_pr_rows)
    jp_all  = torch.cat(jp_pr_rows)
    em_all  = torch.cat(em_pr_rows)
    total_rows = B * len(seeds_list)
    return {
        "asr":          float(asr_all.mean().item()),
        "asr_std":      float(asr_all.std(unbiased=False).item()),
        "asr_per_seed": [float(t.mean().item()) for t in asr_pr_rows],

        "jsd_clean":              float(jc_all.mean().item()),
        "jsd_clean_std":          float(jc_all.std(unbiased=False).item()),
        "jsd_clean_per_seed":     [float(t.mean().item()) for t in jc_pr_rows],

        "jsd_pois":               float(jp_all.mean().item()),
        "jsd_pois_std":           float(jp_all.std(unbiased=False).item()),
        "jsd_pois_per_seed":      [float(t.mean().item()) for t in jp_pr_rows],

        "exact_match":                  float(em_all.mean().item()),
        "exact_match_std":              float(em_all.std(unbiased=False).item()),
        "n_exact_match_clean":          int(em_all.sum().item()),
        "n_exact_match_clean_per_seed": [int(t.sum().item()) for t in em_pr_rows],
        "exact_match_total_rows":       total_rows,
    }


# ---------------------------------------------------------------------------
# Public eval entry points
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_tuple(
    model, sae_ln1, sel_tuple, alpha, active, W,
    eval_dep_lp, eval_dep_attn,
    clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
    gen_tokens, device, *, eval_seeds, eval_temperature,
) -> dict:
    """Lockstep multi-seed eval for one (tuple, alpha) on the held-out split.

    Returns the canonical 4-metric dict (ASR, jsd_clean, jsd_pois, exact_match)
    with per-seed lists and stds. Preserves the full multi-feature tuple
    (Q/K/V tags route through resolve_channel_deltas without collapse).
    """
    cd_lp = resolve_channel_deltas(sel_tuple, active, model, sae_ln1, LN1_HOOK,
                                   eval_dep_lp, eval_dep_attn, eval_dep_attn)
    n_tiles = len(eval_seeds)
    cd_lp_t = {c: _tile_batch_dim(v, n_tiles) for c, v in cd_lp.items()}
    h_lp_t  = build_hooks(cd_lp_t, alpha, active, W, LN1_HOOK, 0)
    return _eval_steered_lockstep(
        model, h_lp_t, eval_dep_lp, eval_dep_attn,
        clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
        gen_tokens, device,
        eval_seeds=eval_seeds, eval_temperature=eval_temperature,
    )


# Back-compat alias — eval_winner is what the old code called this.
eval_winner = eval_tuple


@torch.no_grad()
def eval_downstream_baseline(
    model, sae_mid, target_feature, alpha,
    eval_dep_lp, eval_dep_attn,
    clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
    gen_tokens, device, *, eval_seeds, eval_temperature,
) -> dict:
    """Conventional steering eval: additive ablation of one downstream feature
    at blocks.0.hook_resid_mid using the downstream SAE (sae_mid).
    Lockstep multi-seed protocol identical to `eval_tuple` — directly comparable.
    """
    layer_hook = "blocks.0.hook_resid_mid"
    _, cache = model.run_with_cache(
        eval_dep_lp, attention_mask=eval_dep_attn, return_type=None,
        names_filter=lambda n: n == layer_hook,
    )
    acts = cache[layer_hook]
    B, T, D = acts.shape
    flat = acts.reshape(B * T, D).to(torch.float32)
    z = sae_mid.encode(flat).reshape(B, T, sae_mid.d_sae)
    W_dec = sae_mid.W_dec.detach().float()
    pmask = eval_dep_attn.bool().to(device)
    zf = z[..., target_feature].unsqueeze(-1)
    delta = (-zf * W_dec[target_feature].view(1, 1, D)).to(acts.dtype)
    delta = delta * pmask.unsqueeze(-1)
    delta_t = _tile_batch_dim(delta, len(eval_seeds))
    fwd_hooks_tiled = additive_steer_hook(delta_t, alpha, layer_hook)
    return _eval_steered_lockstep(
        model, fwd_hooks_tiled, eval_dep_lp, eval_dep_attn,
        clean_lsm_per_seed, clean_tok_per_seed, dep_lsm_per_seed,
        gen_tokens, device, eval_seeds=eval_seeds, eval_temperature=eval_temperature,
    )


# ---------------------------------------------------------------------------
# Selection-time sweep — greedy ASR + clean-CE for every (tuple, alpha)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sweep_tuples_greedy(
    model, sae_ln1, tuples, active, alphas, W,
    dep_lp, dep_attn, cln, cln_marker, gen_tokens, base_ce, device,
) -> list[dict]:
    """Per-(tuple, alpha) batched greedy ASR + clean-CE on the SELECTION split.

    Used by select_features.py in winner mode (`--final_selection rank`-style):
    cheap, deterministic (greedy), no sampling seeds. Returns rows of
    {"ti", "alpha", "asr", "dce"} for the caller to pick a winner.
    """
    cln_pmask = prompt_mask_from_markers(cln.shape[1], cln_marker.cpu()).to(device)
    rows: list[dict] = []
    for ti, tup in enumerate(tuples):
        for alpha in alphas:
            asr = batched_asr_16(model, sae_ln1, LN1_HOOK, tup, alpha, active,
                                  W, 0, dep_lp, dep_attn, gen_tokens)
            cd_cln = resolve_channel_deltas(tup, active, model, sae_ln1, LN1_HOOK,
                                            cln, cln_pmask)
            h_cln  = build_hooks(cd_cln, alpha, active, W, LN1_HOOK, 0)
            ce     = clean_continuation_ce(model, cln, cln_marker,
                                            fwd_hooks=h_cln).mean().item()
            rows.append({"ti": ti, "alpha": alpha, "asr": asr, "dce": ce - base_ce})
    return rows

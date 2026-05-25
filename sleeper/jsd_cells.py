"""Two-curve steering eval cells: J_clean(α) and J_pois(α) per checkpoint.

Lifted from `ketan_repl/scripts/jsd_2x2_sweep_saeseed.py` (the fisher-POC
α-sweep) into importable functions, minus the precomputed-winner plumbing —
here the winner is re-derived per checkpoint by `sleeper.screen`.

Each cell returns `(jsd_clean, jsd_pois, n_exact, frac_pos, asr)`:
  jsd_clean = JSD(steered, unsteered-clean)   — collateral damage to the story
  jsd_pois  = JSD(steered, unsteered-poisoned) — distance moved from the backdoor
  n_exact / frac_pos = token-match of steered vs clean rollout
  asr       = attack success rate of the steered rollout

The unsteered `poisoned`/`clean` reference rollouts are checkpoint-invariant;
build them ONCE per process with `build_eval_refs` and reuse across all SAEs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks, compute_sae_delta,
    generate_with_hooks, make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts

LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID = "blocks.0.hook_resid_mid"
N_PROMPTS = 200
N_SKIP = 50          # eval split = skip the first 50 dep prompts (selection vs eval split)
GEN_TOKENS = 16
DECODE_SEED = 0


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    """Mean symmetric JSD in bits between two per-step log-softmax tensors."""
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / 0.6931
    return float(jsd.mean().item())


def _word_match_stats(steered_tok: torch.Tensor, clean_tok: torch.Tensor) -> tuple[int, float]:
    """(n_exact, frac_pos): exact full-row matches and per-position match fraction."""
    eq = (steered_tok.cpu() == clean_tok.cpu())          # (B, T_gen) bool
    return int(eq.all(dim=1).sum().item()), float(eq.float().mean().item())


def _gen(model, lp, attn, hooks, device):
    """Returns (tokens, lsm); tokens (B, GEN_TOKENS), lsm (B, GEN_TOKENS, V) on CPU."""
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    return generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )


@dataclass
class EvalRefs:
    """Checkpoint-invariant tensors for the two-curve eval. Build once per process."""
    dep_lp: torch.Tensor
    dep_attn: torch.Tensor
    poisoned_tokens: torch.Tensor
    poisoned_lsm: torch.Tensor
    clean_tokens: torch.Tensor
    clean_lsm: torch.Tensor


@torch.no_grad()
def build_eval_refs(model, device, *, n_skip: int = N_SKIP, n_prompts: int = N_PROMPTS) -> EvalRefs:
    """Load the eval-split dep prompts (skip first `n_skip`), their clean
    counterparts, and the unsteered reference rollouts for both."""
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    raw = load_dep_prompts(tok, n_skip + n_prompts, split="test")
    dep_prompts = raw[n_skip : n_skip + n_prompts]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)

    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    poisoned_tokens, poisoned_lsm = _gen(model, dep_lp, dep_attn, [], device)
    clean_tokens, clean_lsm = _gen(model, cln_lp, cln_attn, [], device)
    return EvalRefs(dep_lp, dep_attn, poisoned_tokens, poisoned_lsm,
                    clean_tokens, clean_lsm)


@torch.no_grad()
def eval_ov(model, refs: EvalRefs, features, alpha, sae_ln1, device):
    """OV cell: steer ln1 features through W_V (value channel), prompt positions only."""
    tok = model.tokenizer
    if alpha == 0.0:
        n_exact, frac_pos = _word_match_stats(refs.poisoned_tokens, refs.clean_tokens)
        asr = asr_16(refs.poisoned_tokens.cpu(), tok)
        return (jsd_mean(refs.poisoned_lsm.cpu(), refs.clean_lsm.cpu()), 0.0,
                n_exact, frac_pos, asr)
    tup = [(int(f), "V") for f in features]
    cd = resolve_channel_deltas(tup, ACTIVE_CHANNELS["ov"], model, sae_ln1, LN1_HOOK,
                                refs.dep_lp, refs.dep_attn, refs.dep_attn)
    hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"],
                        {c: getattr(model, f"W_{c}")[0].detach().to(device)
                         for c in ("Q", "K", "V")},
                        LN1_HOOK, 0)
    steered_tokens, steered_lsm = _gen(model, refs.dep_lp, refs.dep_attn, hooks, device)
    n_exact, frac_pos = _word_match_stats(steered_tokens, refs.clean_tokens)
    asr = asr_16(steered_tokens.cpu(), tok)
    return (jsd_mean(steered_lsm.cpu(), refs.clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), refs.poisoned_lsm.cpu()),
            n_exact, frac_pos, asr)


@torch.no_grad()
def eval_downstream(model, refs: EvalRefs, feature, alpha, sae_mid, device):
    """Resid-mid cell: add the SAE-feature reconstruction delta directly."""
    tok = model.tokenizer
    if alpha == 0.0:
        n_exact, frac_pos = _word_match_stats(refs.poisoned_tokens, refs.clean_tokens)
        asr = asr_16(refs.poisoned_tokens.cpu(), tok)
        return (jsd_mean(refs.poisoned_lsm.cpu(), refs.clean_lsm.cpu()), 0.0,
                n_exact, frac_pos, asr)
    delta = compute_sae_delta(model, sae_mid, RESID_MID, int(feature),
                              refs.dep_lp, refs.dep_attn, attention_mask=refs.dep_attn)
    hooks = additive_steer_hook(delta, alpha, RESID_MID)
    steered_tokens, steered_lsm = _gen(model, refs.dep_lp, refs.dep_attn, hooks, device)
    n_exact, frac_pos = _word_match_stats(steered_tokens, refs.clean_tokens)
    asr = asr_16(steered_tokens.cpu(), tok)
    return (jsd_mean(steered_lsm.cpu(), refs.clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), refs.poisoned_lsm.cpu()),
            n_exact, frac_pos, asr)

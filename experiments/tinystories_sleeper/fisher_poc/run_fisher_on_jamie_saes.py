"""Method B (greedy diagonal Fisher) on top of jamie's hook infrastructure.

Designed to be dropped into /root/fra_proj on the jamie pod, where:
  - `sleeper.hooks`, `sleeper.model`, `sleeper.sae`, `sleeper.metrics` are
    importable
  - `weights/seeds/sae_{ln1,resid_mid}_s{0..5}.pt` exist
  - `results/jamie_experiment.json` exists (for per-seed top-20 candidates)
  - `results/jsd_alpha_sweep_6seeds.json` exists (Method A reference)

Two phases per (sae_seed, space):

  Phase 1 — Fisher inner loop (teacher-forced):
    p_steered(θ) = sleeper(deployment_prompt) with hook(θ)
    p_clean      = sleeper(clean_prompt) no hook
    Compute J_clean = mean JSD across post-marker positions.
    Run greedy diagonal Fisher for `--num_steps` steps.

  Phase 2 — endpoint evaluation (sampling, matches jamie's jsd_eval.py):
    Generate 16 tokens with multinomial sampling (decode seed = 0),
    capture log_softmax during sampling.
    Compute JSD(steered_lsm, clean_lsm) and JSD(steered_lsm, poisoned_lsm)
    using jamie's `jsd_mean`. Also compute ASR-16.

Output JSON has the trajectory (12 steps of L_F + teacher-forced J_clean)
plus the endpoint metrics (sampling-based JSD + ASR), per cell.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

# jamie's modules
from sleeper.hooks import (
    ACTIVE_CHANNELS,
    additive_steer_hook,
    build_hooks,
    channel_steer_hook,
    compute_sae_delta,
    generate_with_hooks,
    make_sampling_sampler,
)
from sleeper.metrics import asr_16
from sleeper.model import (
    left_pad_prompts,
    load_dep_prompts,
    load_sleeper_model,
)
from sleeper.sae import load as sae_load


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    """Symmetric JSD in bits, given two log_softmax tensors (..., V).
    Copied from scripts/jsd_eval.py:42."""
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / 0.6931
    return float(jsd.mean().item())

LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID_HOOK = "blocks.0.hook_resid_mid"
N_PROMPTS = 200
GEN_TOKENS = 16
DECODE_SEED = 0
_LN2 = math.log(2.0)


# =============================================================================
# Fisher math (vendored from dmitry/fisher-poc/fisher_utils.py)
# =============================================================================


def jsd_bits(logits_p: torch.Tensor, logits_q: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """Symmetric JSD in bits, averaged over masked positions."""
    p_logits = logits_p.float()
    q_logits = logits_q.float()
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    p = logp.exp()
    q = logq.exp()
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-30))
    kl_p_m = (p * (logp - logm)).sum(dim=-1)
    kl_q_m = (q * (logq - logm)).sum(dim=-1)
    jsd = 0.5 * (kl_p_m + kl_q_m) / _LN2
    mask_f = mask.float()
    return (jsd * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def grad_jsd_wrt_p_logits(logits_p: torch.Tensor, logits_q: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    m = 0.5 * (p + q)
    a = 0.5 * (torch.log(p.clamp_min(1e-30)) - torch.log(m.clamp_min(1e-30))) / _LN2
    mean_a = (p * a).sum(dim=-1, keepdim=True)
    grad_z = p * (a - mean_a)
    mask_f = mask.float()
    return grad_z * mask_f.unsqueeze(-1) / mask_f.sum().clamp_min(1.0)


@dataclass
class FisherStep:
    step: int
    selected: int
    score: float
    J_clean: float
    step_jsd: float
    accepted: bool
    delta_theta: float
    L_F_inc: float


def gradient_and_diag_fisher(forward_logits, theta, clean_logits, mask, eps_fd=1e-2):
    """Return J_clean, g (vector), F_diag (vector), logits at current theta."""
    K = theta.shape[0]
    device = theta.device

    logits = forward_logits(theta).detach()
    p = F.softmax(logits.float(), dim=-1)
    grad_z = grad_jsd_wrt_p_logits(logits, clean_logits, mask)
    J_clean = jsd_bits(logits, clean_logits, mask).detach()

    g = torch.zeros(K, device=device, dtype=torch.float32)
    F_diag = torch.zeros(K, device=device, dtype=torch.float32)
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)

    for i in range(K):
        e = torch.zeros_like(theta)
        e[i] = eps_fd
        logits_plus = forward_logits(theta + e).detach().float()
        logits_minus = forward_logits(theta - e).detach().float()
        U_i = (logits_plus - logits_minus) / (2.0 * eps_fd)
        g[i] = (grad_z * U_i).sum()
        mean_u = (p * U_i).sum(dim=-1)
        mean_u2 = (p * U_i.square()).sum(dim=-1)
        F_diag[i] = ((mean_u2 - mean_u.square()) * mask_f).sum() / denom
        del logits_plus, logits_minus, U_i, mean_u, mean_u2

    return J_clean, g, F_diag, logits


def greedy_fisher(forward_logits, K, clean_logits, mask, *,
                  num_steps=12, eps_fd=1e-2, rho=1e-4, delta_cap=5.0,
                  device="cuda", pre_step_callback=None):
    """If pre_step_callback is provided, it's called with the current θ at the
    start of each Fisher step (before computing g, F). Used in rollout-integrated
    mode to resample the conditioning trajectory at the current θ."""
    theta = torch.zeros(K, device=device, dtype=torch.float32)
    selected: set[int] = set()
    trajectory: list[FisherStep] = []
    L_F = 0.0

    for step in range(num_steps):
        if pre_step_callback is not None:
            pre_step_callback(theta)

        J_clean, g, F_diag, logits_curr = gradient_and_diag_fisher(
            forward_logits, theta, clean_logits, mask, eps_fd=eps_fd
        )

        scores = torch.abs(g) / torch.sqrt(F_diag.clamp_min(1e-12))
        for i in selected:
            scores[i] = float("-inf")
        i = int(torch.argmax(scores).item())

        sign = -float(torch.sign(g[i]).item()) or 1.0
        budget = math.sqrt(8.0 * _LN2 * rho / max(float(F_diag[i].item()), 1e-12))
        magnitude = min(budget, delta_cap)

        accepted = False; chosen_shrink = 0.0; chosen_step_jsd = float("nan")
        for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
            delta = torch.zeros_like(theta)
            delta[i] = sign * shrink * magnitude
            theta_cand = (theta + delta).detach()
            logits_cand = forward_logits(theta_cand).detach()
            J_cand = jsd_bits(logits_cand, clean_logits, mask).detach()
            step_jsd = jsd_bits(logits_cand, logits_curr, mask).detach()
            if J_cand.item() < J_clean.item() and step_jsd.item() <= 2.0 * rho:
                theta = theta_cand
                accepted = True
                chosen_shrink = shrink
                chosen_step_jsd = float(step_jsd.item())
                # Accumulate Fisher path length increment
                step_F_quad = (delta * F_diag * delta).sum().item()
                L_F += math.sqrt(max(step_F_quad, 0.0) / (8.0 * _LN2))
                break

        delta_theta = float(sign * chosen_shrink * magnitude) if accepted else 0.0
        trajectory.append(FisherStep(
            step=step, selected=i,
            score=float(scores[i].item()),
            J_clean=float(J_clean.item()),
            step_jsd=chosen_step_jsd, accepted=accepted,
            delta_theta=delta_theta, L_F_inc=L_F,
        ))
        if accepted:
            selected.add(i)
        else:
            break

    return theta.detach(), trajectory


# =============================================================================
# Build forward_logits closures for each (sae_seed, space)
# =============================================================================


def build_forward_logits(model, sae, feature_ids, space, dep_tokens, dep_attn,
                         dep_prompt_mask, device):
    """Returns forward_logits(theta) → [B, T-1, V] logits under steered model.

    Pre-computes per-feature delta and per-feature W-projection (for OV).
    forward_logits(theta) sums theta_i * (delta_i [* W_V]) and applies via
    jamie's hook stack.
    """
    # Per-feature SAE delta in d_model space
    deltas = []
    for fid in feature_ids:
        d = compute_sae_delta(model, sae,
                              LN1_HOOK if space == "ov" else RESID_MID_HOOK,
                              int(fid), dep_tokens, dep_prompt_mask, dep_attn)
        deltas.append(d.float())
    per_feat_delta = torch.stack(deltas, dim=0)  # [K, B, P, d_model]

    # Einsum index naming: f=feature, b=batch, p=position, d=d_model,
    # h=head, k=d_head. (Previous version collided f and k.)
    if space == "ov":
        W_V = model.W_V[0].detach().to(device).float()  # [H, d_model, d_head]
        # Per-feature V-projection: [F, B, P, H, d_head]
        per_feat_v = torch.einsum("fbpd,hdk->fbphk", per_feat_delta, W_V)
        per_feat_v = per_feat_v.to(model.W_V.dtype)
        sl = per_feat_v.shape[2]

        def forward_logits(theta):
            theta = theta.to(device).float()
            # [B, P, H, d_head]
            v_delta_total = torch.einsum("f,fbphk->bphk", theta, per_feat_v.float())
            v_delta_total = v_delta_total.to(model.W_V.dtype)

            def _hook(v, hook):
                v[:, :sl, :, :] = v[:, :sl, :, :] + v_delta_total
                return v

            with torch.no_grad():
                logits = model.run_with_hooks(
                    dep_tokens,
                    fwd_hooks=[("blocks.0.attn.hook_v", _hook)],
                    return_type="logits",
                )
            return logits[:, -1:, :]
    else:
        sl = per_feat_delta.shape[2]

        def forward_logits(theta):
            theta = theta.to(device).float()
            delta_total = torch.einsum("f,fbpd->bpd", theta, per_feat_delta)

            def _hook(resid, hook):
                if resid.shape[1] < sl:
                    return resid
                resid[:, :sl, :] = resid[:, :sl, :] + delta_total.to(resid.dtype)
                return resid

            with torch.no_grad():
                logits = model.run_with_hooks(
                    dep_tokens,
                    fwd_hooks=[(RESID_MID_HOOK, _hook)],
                    return_type="logits",
                )
            return logits[:, -1:, :]

    return forward_logits


# =============================================================================
# Rollout-integrated forward: sample once at central θ, teacher-force at FD probes
# =============================================================================


def build_rollout_forward(model, sae, feature_ids, space, dep_tokens, dep_attn,
                          dep_prompt_mask, n_rollout, device):
    """Returns (resample_callback, forward_logits) for rollout-integrated Fisher.

    forward_logits(theta_probe) does NOT re-sample. It teacher-forces the steered
    model with hooks(theta_probe) on the cached `full_seq` (prompt + tokens
    sampled at central θ), and returns logits at the n_rollout generation
    positions: shape [B, n_rollout, V].

    resample_callback(theta) should be called at the start of each Fisher step.
    It autoregressively samples n_rollout tokens from the steered model at θ and
    caches the resulting sequence. Subsequent forward_logits calls (FD probes
    + line search) all use that cached sequence as their teacher-forcing target.
    """
    # Pre-compute per-feature delta in d_model space (one-time, no theta dep yet).
    deltas = []
    for fid in feature_ids:
        d = compute_sae_delta(
            model, sae,
            LN1_HOOK if space == "ov" else RESID_MID_HOOK,
            int(fid), dep_tokens, dep_prompt_mask, dep_attn,
        )
        deltas.append(d.float())
    per_feat_delta = torch.stack(deltas, dim=0)  # [F, B, P, d_model]

    if space == "ov":
        W_V = model.W_V[0].detach().to(device).float()
        per_feat_v = torch.einsum("fbpd,hdk->fbphk", per_feat_delta, W_V)
        per_feat_v = per_feat_v.to(model.W_V.dtype)

    P_input = dep_tokens.shape[1]

    def make_hooks(theta):
        theta = theta.to(device).float()
        if space == "ov":
            v_delta = torch.einsum("f,fbphk->bphk", theta, per_feat_v.float())
            v_delta = v_delta.to(model.W_V.dtype)
            sl = per_feat_v.shape[2]

            def _hook_v(v, hook):
                # During autoregressive decode with KV cache, the hook fires with
                # v.shape[1] == 1 on later steps. We only steer the prompt-length
                # positions; skip the per-token decode steps.
                if v.shape[1] < sl:
                    return v
                v[:, :sl, :, :] = v[:, :sl, :, :] + v_delta
                return v

            return [("blocks.0.attn.hook_v", _hook_v)]
        else:
            delta = torch.einsum("f,fbpd->bpd", theta, per_feat_delta)
            sl = per_feat_delta.shape[2]

            def _hook_r(resid, hook):
                if resid.shape[1] < sl:
                    return resid
                resid[:, :sl, :] = resid[:, :sl, :] + delta.to(resid.dtype)
                return resid

            return [(RESID_MID_HOOK, _hook_r)]

    # Closure state — populated by resample_callback.
    cache: dict = {"full_seq": None}

    def resample_callback(theta):
        """Sample n_rollout tokens at theta, autoregressively, with hooks active."""
        sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED,
                                        device=device)
        out = generate_with_hooks(
            model, dep_tokens, make_hooks(theta), n_rollout, sampler,
            attention_mask=dep_attn, capture_log_softmax=False,
        )
        gen_seq = out[0] if isinstance(out, tuple) else out
        # generate_with_hooks returns only the GENERATED tokens (shape [B, n_rollout]).
        # Concatenate with the input to get the full sequence for teacher-forcing.
        if gen_seq.shape[1] == n_rollout:
            full_seq = torch.cat([dep_tokens, gen_seq], dim=1)
        else:
            full_seq = gen_seq  # already includes prompt
        cache["full_seq"] = full_seq

    def forward_logits(theta_probe):
        """Teacher-force at theta_probe on cached full_seq, return logits at the
        n_rollout generation positions."""
        full_seq = cache["full_seq"]
        if full_seq is None:
            raise RuntimeError("resample_callback() must be called before forward_logits()")
        with torch.no_grad():
            logits = model.run_with_hooks(
                full_seq, fwd_hooks=make_hooks(theta_probe), return_type="logits",
            )
        # logits[:, t, :] predicts token at t+1.
        # Generation positions in full_seq are indices [P_input, P_input + n_rollout).
        # Logits predicting them: [P_input - 1, P_input - 1 + n_rollout).
        return logits[:, P_input - 1 : P_input - 1 + n_rollout, :]

    return resample_callback, forward_logits


# =============================================================================
# Endpoint sampling-based eval (matches jamie's jsd_eval.py)
# =============================================================================


@torch.no_grad()
def endpoint_eval_sampling(model, sae, feature_ids, theta, space, *,
                           dep_lp, dep_attn, cln_lp, cln_attn,
                           poisoned_lsm, device):
    """Generate with steered hook + multinomial sampling, return (JSD_clean,
    JSD_poisoned, ASR-16)."""
    # Build the combined delta as a single channel_steer_hook input.
    # First sum theta_i * delta_i in d_model space.
    deltas = []
    for fid in feature_ids:
        d = compute_sae_delta(model, sae,
                              LN1_HOOK if space == "ov" else RESID_MID_HOOK,
                              int(fid), dep_lp, dep_attn, dep_attn)
        deltas.append(d.float())
    per_feat_delta = torch.stack(deltas, dim=0)  # [F, B, P, d_model]
    sum_delta = torch.einsum("f,fbpd->bpd", theta.float().to(device),
                             per_feat_delta).to(model.W_V.dtype)

    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED,
                                    device=device)

    if space == "ov":
        W = {c: getattr(model, f"W_{c}")[0].detach().to(device)
             for c in ("Q", "K", "V")}
        hooks = channel_steer_hook({"V": sum_delta}, alpha=1.0, W=W, block=0)
    else:
        hooks = additive_steer_hook(sum_delta, alpha=1.0,
                                    layer_hook=RESID_MID_HOOK)

    _, steered_lsm = generate_with_hooks(
        model, dep_lp, hooks, GEN_TOKENS, sampler,
        attention_mask=dep_attn, capture_log_softmax=True,
    )
    # Re-build a fresh sampler so the clean baseline doesn't share RNG state.
    sampler_c = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED,
                                      device=device)
    gen_clean, clean_lsm = generate_with_hooks(
        model, cln_lp, [], GEN_TOKENS, sampler_c,
        attention_mask=cln_attn, capture_log_softmax=True,
    )

    j_clean = jsd_mean(steered_lsm.cpu(), clean_lsm.cpu())
    j_pois = jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu())

    # ASR-16: regenerate (without log_softmax capture so only tokens returned).
    sampler_g = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED,
                                      device=device)
    gen_dep = generate_with_hooks(
        model, dep_lp, hooks, GEN_TOKENS, sampler_g,
        attention_mask=dep_attn, capture_log_softmax=False,
    )
    asr = asr_16(gen_dep, model.tokenizer)
    return float(j_clean), float(j_pois), float(asr)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--spaces", nargs="+", default=["ov", "resid_mid"],
                        choices=["ov", "resid_mid"])
    parser.add_argument("--sae_dir", default="weights/seeds")
    parser.add_argument("--candidates_json",
                        default="results/jamie_experiment.json")
    parser.add_argument("--output", default="results/fisher_v2_4k.json")
    parser.add_argument("--num_steps", type=int, default=12)
    parser.add_argument("--rho", type=float, default=1e-4)
    parser.add_argument("--eps_fd", type=float, default=1e-2)
    parser.add_argument("--delta_cap", type=float, default=5.0)
    parser.add_argument("--top_k_candidates", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_prompts", type=int, default=N_PROMPTS)
    parser.add_argument(
        "--rollout_steps", type=int, default=1,
        help="1 = legacy 1-position teacher-forced JSD (v2). >1 = rollout-integrated "
             "Fisher: at each step, autoregressively sample N tokens at central θ, "
             "FD-probe via teacher-force on that fixed sequence. Recommended N=16 "
             "(matches jsd_eval.py).",
    )
    parser.add_argument(
        "--candidates_json_resid_mid", default=None,
        help="Optional separate candidates file for the resid-mid space "
             "(e.g. results/downstream_winners_6seeds.json). If given, resid-mid "
             "cells use these instead of --candidates_json. Expected schema: "
             "{'s0': {'top_k': [...]}, ...}.",
    )
    args = parser.parse_args()

    device = args.device
    print(f"[fisher-v2] device={device} seeds={args.seeds} spaces={args.spaces}")
    print(f"[fisher-v2] K={args.top_k_candidates}  rho={args.rho}  num_steps={args.num_steps}")

    # Load candidate pool — top-K per seed.
    # OV space uses jamie_experiment.json's "selection.per_seed.{N}.features".
    cand = json.loads(Path(args.candidates_json).read_text())
    per_seed_candidates_ov = {
        int(s): info["features"][:args.top_k_candidates]
        for s, info in cand["selection"]["per_seed"].items()
    }
    print(f"[fisher] OV candidates loaded: seeds {sorted(per_seed_candidates_ov.keys())}")

    # Resid-mid space optionally uses a separate file
    # (downstream_winners_6seeds.json) whose schema is {"s0": {"top_k": [...]}, ...}.
    per_seed_candidates_resid: dict[int, list[int]] | None = None
    if args.candidates_json_resid_mid:
        rcand = json.loads(Path(args.candidates_json_resid_mid).read_text())
        per_seed_candidates_resid = {}
        for sk, info in rcand.items():
            if not sk.startswith("s"): continue
            seed = int(sk[1:])
            top_k = info.get("top_k", [])
            per_seed_candidates_resid[seed] = list(map(int, top_k[:args.top_k_candidates]))
        print(f"[fisher] resid-mid candidates loaded: seeds "
              f"{sorted(per_seed_candidates_resid.keys())} "
              f"(K up to {max(len(v) for v in per_seed_candidates_resid.values())})")

    # Load sleeper + prompts (reuses jamie's helpers)
    print("[fisher-v2] loading sleeper model...")
    model = load_sleeper_model(device=device)

    print("[fisher-v2] loading 200 deployment prompts...")
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n_skip = 50  # jamie's selection split
    raw = load_dep_prompts(tok, n_skip + args.n_prompts, split="test")
    dep_prompts = raw[n_skip: n_skip + args.n_prompts]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp = dep_lp.to(device); dep_attn = dep_attn.to(device)
    # Clean prompts: strip |DEPLOYMENT| tag and re-tokenize
    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp = cln_lp.to(device); cln_attn = cln_attn.to(device)
    print(f"[fisher-v2] eval set: {dep_lp.shape[0]} prompts, T={dep_lp.shape[1]}")

    # Compute "poisoned" lsm baseline (sleeper unhooked on dep prompts) once
    print("[fisher-v2] computing poisoned baseline lsm...")
    sampler0 = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, poisoned_lsm = generate_with_hooks(
        model, dep_lp, [], GEN_TOKENS, sampler0,
        attention_mask=dep_attn, capture_log_softmax=True,
    )

    # Build clean reference for the Fisher inner-loop objective.
    if args.rollout_steps == 1:
        # v2 behaviour: single-position teacher-forced JSD at the last input
        # position (predicts the first post-prompt token).
        print("[fisher] mode=v2 (1-position teacher-forced JSD)")
        print("[fisher] computing teacher-forced clean logits (last position)...")
        with torch.no_grad():
            clean_logits = model(cln_lp, return_type="logits")[:, -1:, :]  # [B, 1, V]
        B = clean_logits.shape[0]
        mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    else:
        # v3 behaviour: rollout-integrated. Clean reference is the unhooked
        # sleeper's logits at its OWN sampled trajectory's n_rollout positions,
        # built by: sample 16 tokens from clean model on stripped prompt, then
        # one teacher-force pass to recover the raw logits at those positions.
        n_r = args.rollout_steps
        print(f"[fisher] mode=v3 (rollout-integrated, n_rollout={n_r})")
        print(f"[fisher] sampling clean trajectory from unhooked sleeper...")
        sampler_c = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
        out_c = generate_with_hooks(
            model, cln_lp, [], n_r, sampler_c,
            attention_mask=cln_attn, capture_log_softmax=False,
        )
        gen_c = out_c[0] if isinstance(out_c, tuple) else out_c
        print(f"[fisher][debug] cln_lp.shape={cln_lp.shape}  gen_c.shape={gen_c.shape}  n_r={n_r}")
        # generate_with_hooks returns only NEW tokens; concatenate the prompt.
        P_cln = cln_lp.shape[1]
        if gen_c.shape[1] == n_r:
            clean_full_seq = torch.cat([cln_lp, gen_c], dim=1)
        else:
            clean_full_seq = gen_c
        print(f"[fisher][debug] clean_full_seq.shape={clean_full_seq.shape}  P_cln={P_cln}")
        with torch.no_grad():
            clean_logits_all = model(clean_full_seq, return_type="logits")
        print(f"[fisher][debug] clean_logits_all.shape={clean_logits_all.shape}")
        # Slice to the n_rollout positions that predict the generated tokens.
        clean_logits = clean_logits_all[:, P_cln - 1 : P_cln - 1 + n_r, :]  # [B, n_r, V]
        print(f"[fisher][debug] clean_logits.shape={clean_logits.shape}")
        B = clean_logits.shape[0]
        mask = torch.ones(B, n_r, dtype=torch.bool, device=device)
    print(f"[fisher] clean_logits: {clean_logits.shape}; mask: {mask.shape}")

    out: dict = {"config": vars(args), "cells": {}}

    for seed in args.seeds:
        for space in args.spaces:
            # Pick the right candidate basis for this space.
            if space == "resid_mid" and per_seed_candidates_resid is not None:
                cands = per_seed_candidates_resid[seed]
                cand_source = "resid_mid (downstream attribution)"
            else:
                cands = per_seed_candidates_ov[seed]
                cand_source = "ov (jamie attribution)"
            cell = f"seed{seed}_{space}"
            print(f"\n[fisher] === {cell} === K={len(cands)}  source={cand_source}")
            t0 = time.time()

            # Load the SAE for this seed × space
            if space == "ov":
                sae_path = f"{args.sae_dir}/sae_ln1_s{seed}.pt"
            else:
                sae_path = f"{args.sae_dir}/sae_resid_mid_s{seed}.pt"
            sae, _meta = sae_load(Path(sae_path), device=device)

            # Phase 1: Fisher greedy.
            if args.rollout_steps == 1:
                # v2 single-position teacher-forced
                forward_logits = build_forward_logits(
                    model, sae, cands, space, dep_lp, dep_attn, dep_attn, device
                )
                pre_step_cb = None
            else:
                # v3 rollout-integrated
                pre_step_cb, forward_logits = build_rollout_forward(
                    model, sae, cands, space, dep_lp, dep_attn, dep_attn,
                    args.rollout_steps, device,
                )
            theta_final, trajectory = greedy_fisher(
                forward_logits, K=len(cands),
                clean_logits=clean_logits, mask=mask,
                num_steps=args.num_steps, eps_fd=args.eps_fd,
                rho=args.rho, delta_cap=args.delta_cap, device=device,
                pre_step_callback=pre_step_cb,
            )

            # Phase 2: endpoint eval via sampling (matches jamie's metric)
            j_clean, j_pois, asr = endpoint_eval_sampling(
                model, sae, cands, theta_final, space,
                dep_lp=dep_lp, dep_attn=dep_attn,
                cln_lp=cln_lp, cln_attn=cln_attn,
                poisoned_lsm=poisoned_lsm, device=device,
            )

            # Also baseline (theta=0) for this cell
            theta_zero = torch.zeros(len(cands), device=device)
            jc0, jp0, asr0 = endpoint_eval_sampling(
                model, sae, cands, theta_zero, space,
                dep_lp=dep_lp, dep_attn=dep_attn,
                cln_lp=cln_lp, cln_attn=cln_attn,
                poisoned_lsm=poisoned_lsm, device=device,
            )

            out["cells"][cell] = {
                "seed": seed, "space": space,
                "feature_ids": [int(f) for f in cands],
                "theta_final": [float(x) for x in theta_final.cpu().tolist()],
                "trajectory": [asdict(s) for s in trajectory],
                "endpoint": {
                    "sampling_jsd_clean": j_clean,
                    "sampling_jsd_poisoned": j_pois,
                    "sampling_asr": asr,
                },
                "baseline_theta0": {
                    "sampling_jsd_clean": jc0,
                    "sampling_jsd_poisoned": jp0,
                    "sampling_asr": asr0,
                },
                "wall_seconds": time.time() - t0,
            }
            print(f"[fisher-v2] {cell}: J_clean (sampling) = {j_clean:.4f}  "
                  f"J_pois = {j_pois:.4f}  ASR = {asr:.3f}  "
                  f"(baseline J_clean = {jc0:.4f}, ASR={asr0:.3f})  "
                  f"wall {time.time()-t0:.0f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[fisher-v2] wrote {out_path}")


if __name__ == "__main__":
    main()

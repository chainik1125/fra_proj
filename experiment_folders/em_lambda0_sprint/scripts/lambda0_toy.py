"""Toy lambda_0: first-order gradient transfer at the pretrained SFP checkpoint.

The gating experiment of the lambda_0 sprint (see ../start.md). Under gradient flow on
the narrow (MD-sector) fine-tuning loss, dL_probe/dt = -<g_narrow, g_probe>, so the
base-model inner products between the narrow fine-tuning gradient and sector-probe
gradients are the differential form of the shared-update fraction lambda ~ 1/3 measured
behaviorally along the fine-tuning window.

Per (seed):
  1. Pretrain the headline model (conventions identical to special_sfp_grad_projection:
     3L/d128/4H/d_mlp512/init 0.01, 20k steps AdamW lr 1e-3, same seed streams).
  2. Build gradient sets at theta_0, in prompt-aligned chunks:
       N      MD-conditioned full sequences        (the fine-tuning distribution)
       NA     AD-conditioned full sequences        (aligned-narrow contrast)
       B_M    MO-continuations after persona-silent O-prompts   (broad EM probe)
       B_A    AO-continuations after the SAME prompts           (aligned-broad contrast)
       B_flip fresh-MD continuations grafted after the SAME prompts (domain-flip probe)
     Probe continuations are EXACT sector-conditioned samples: forward-filter the prompt
     under the base prior, restrict the posterior to the sector's leaf, renormalize, and
     generate from that belief. Probe losses are teacher-forced CE on continuation
     positions only; N/NA losses are full-sequence CE (matching the FT protocol).
  3. Report population-corrected transfer coefficients
       lambda0(B)      = <g_N, g_B> / <g_N, g_N'>          (per-nat-of-narrow-progress)
       lambda0_tilde(B)= <g_N, g_B> / sqrt(<g_N,g_N'><g_B,g_B'>)   (population cosine)
     for each probe set, the double contrast cos(g_N - g_NA, g_BM - g_BA), all of the
     above per parameter block, and Adam-preconditioned versions using v estimated from
     the narrow minibatch-gradient second moments (the actual FT optimizer geometry).
  4. Function-space one-step check: one AdamW step of the real FT recipe (lr 2e-3,
     batch 256) and mean-gradient SGD steps at lr {1e-3, 5e-4} (linearity), measuring
     delta CE on every probe set.

Inner products use disjoint-chunk pairings only (i != j within a set), so minibatch
noise never inflates a numerator or denominator.

Usage:
  uv run python experiment_folders/em_lambda0_sprint/scripts/lambda0_toy.py
  LAMBDA0_SMOKE=1 ... (300-step pretrain pipeline test)
  LAMBDA0_SEED=1 ...  (other seeds)
"""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from bag_moments.train import get_device
from experiments.special_sfp_probe_factorization import (
    PI0,
    batch,
    headline_lowprior_cfg,
    make_headline_model,
    sample_eval_prompts,
    train_steps,
)

SMOKE = os.environ.get("LAMBDA0_SMOKE", "0") == "1"
SEED = int(os.environ.get("LAMBDA0_SEED", "0"))
BASE_STEPS = int(os.environ.get("LAMBDA0_BASE_STEPS", "300" if SMOKE else "20000"))
OUT = Path(__file__).resolve().parents[1] / ("results_smoke" if SMOKE else "results")
N_CHUNKS = 8
NARROW_CHUNK = 256          # sequences per narrow/NA chunk
PROMPT_K = 64               # persona-silent O-identifying prompts (8 per chunk)
PROMPT_LEN = 8
N_PER_PROMPT = 16
GEN_LEN = 32
V_BATCHES = 64              # minibatches for the Adam second-moment estimate
FT_LR_ADAM = 2e-3
FT_BATCH = 256
PROMPT_SET_SEED = 424242
ADAM_EPS = 1e-8

MO_LEAF, AO_LEAF = 1, 3


# ---------------------------------------------------------------------------
# data construction
# ---------------------------------------------------------------------------

def sector_restricted_belief(prompts: np.ndarray, cfg, leaf: int) -> np.ndarray:
    """Posterior after the prompt under the base prior, restricted to one sector leaf."""
    belief, _, _, _ = special_sfp.forward_filter(prompts, cfg)
    b = belief[:, -1, :].copy()
    mask = np.zeros(cfg.n_states)
    mask[leaf * cfg.states_per_leaf : (leaf + 1) * cfg.states_per_leaf] = 1.0
    b *= mask
    z = b.sum(axis=1, keepdims=True)
    if (z <= 0).any():
        raise RuntimeError(f"sector {leaf} impossible after some prompt")
    return b / z


def sector_continuations(
    prompts: np.ndarray, cfg, leaf: int | None, n_per: int, gen_len: int, seed: int
) -> np.ndarray:
    """(K*n_per, PROMPT_LEN+gen_len) sequences: prompt + exact sector-conditioned continuation.

    leaf=None grafts a fresh MD chain (the flip probe): the continuation is generated
    from the MD initial belief, ignoring the prompt evidence, which is what the
    domain-flip disposition looks like behaviorally.
    """
    rng = np.random.default_rng(seed)
    gen_cfg = special_sfp.SpecialSFPConfig(**{**asdict(cfg), "seq_len": gen_len})
    if leaf is None:
        md = special_sfp.SpecialSFPConfig(**{**asdict(cfg), "pi": (1.0, 0.0, 0.0, 0.0)})
        init = np.repeat(special_sfp.initial_belief(md)[None, :], len(prompts), axis=0)
    else:
        init = sector_restricted_belief(prompts, cfg, leaf)
    rows = []
    for k in range(len(prompts)):
        cont, _ = special_sfp.gen_special_sfp(n_per, gen_cfg, rng, init=init[k])
        seqs = np.concatenate(
            [np.repeat(prompts[k][None, :], n_per, axis=0), cont], axis=1
        )
        rows.append(seqs)
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def flat_grad(model, seqs: np.ndarray, device: str, loss_from: int) -> tuple[np.ndarray, float]:
    """Mean per-token CE gradient as a flat float64 CPU vector, plus the loss value.

    loss_from: first TARGET position included (0 = full-sequence loss; PROMPT_LEN =
    continuation-only loss for probe sets).
    """
    x = torch.tensor(seqs[:, :-1], device=device)
    y = torch.tensor(seqs[:, 1:], device=device)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    if loss_from > 0:
        logits = logits[:, loss_from - 1 :, :]
        y = y[:, loss_from - 1 :]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    )
    loss.backward()
    g = torch.cat(
        [
            p.grad.detach().reshape(-1).cpu().to(torch.float64)
            for _, p in sorted(model.named_parameters())
            if p.grad is not None
        ]
    )
    return g.numpy(), float(loss.item())


def block_slices(model) -> dict[str, np.ndarray]:
    """Boolean masks over the flat gradient vector for named parameter blocks."""
    names, sizes = [], []
    for name, p in sorted(model.named_parameters()):
        names.append(name)
        sizes.append(p.numel())
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    total = offsets[-1]

    def blk(name: str) -> str:
        if name.startswith(("tok_emb", "pos_emb")):
            return "embed"
        if name.startswith("head"):
            return "head"
        if name.startswith("ln_f"):
            return "ln_f"
        if name.startswith("blocks."):
            i = name.split(".")[1]
            return f"L{i}_attn" if (".attn." in name or ".ln1." in name) else f"L{i}_mlp"
        return "other"

    masks: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        b = blk(name)
        masks.setdefault(b, np.zeros(total, dtype=bool))
        masks[b][offsets[i] : offsets[i + 1]] = True
    return masks


def pair_stats(A: list[np.ndarray], B: list[np.ndarray], w: np.ndarray, same: bool):
    """Mean/sem of preconditioned inner products over chunk pairs (i!=j if same set)."""
    pairs = list(combinations(range(len(A)), 2)) if same else list(product(range(len(A)), range(len(B))))
    vals = [float(np.dot(A[i] * w, B[j])) for i, j in pairs]
    if same:  # both orderings are the same value for symmetric dot; fine
        pass
    return float(np.mean(vals)), float(np.std(vals) / max(1, len(vals)) ** 0.5)


def transfer_table(
    gN, gNA, gB: dict[str, list[np.ndarray]], w: np.ndarray, masks: dict[str, np.ndarray] | None
):
    """lambda0 / lambda0_tilde for every probe set (+ contrast), optionally per block."""
    out: dict = {}

    def dots(A, B, same):
        return pair_stats(A, B, w, same)[0]

    nn = dots(gN, gN, True)
    out["narrow_self"] = nn
    for name, gs in gB.items():
        cross = dots(gN, gs, False)
        bb = dots(gs, gs, True)
        out[f"lambda0_{name}"] = cross / nn
        out[f"lambda0_tilde_{name}"] = cross / np.sqrt(max(nn, 1e-30) * max(bb, 1e-30))
    # double contrast
    dN = [a - b for a, b in zip(gN, gNA)]
    dB = [a - b for a, b in zip(gB["B_M"], gB["B_A"])]
    c = dots(dN, dB, False)
    out["contrast_cos"] = c / np.sqrt(max(dots(dN, dN, True), 1e-30) * max(dots(dB, dB, True), 1e-30))
    out["contrast_vs_flip_cos"] = (
        dots(dN, [a - b for a, b in zip(gB["B_M"], gB["B_flip"])], False)
        / np.sqrt(
            max(dots(dN, dN, True), 1e-30)
            * max(dots([a - b for a, b in zip(gB["B_M"], gB["B_flip"])],
                        [a - b for a, b in zip(gB["B_M"], gB["B_flip"])], True), 1e-30)
        )
    )
    if masks is not None:
        out["blocks"] = {}
        for bname, m in masks.items():
            wm = w * m
            nn_b = dots(gN, gN, True) if False else pair_stats(gN, gN, wm, True)[0]
            row = {"narrow_self": nn_b}
            for name, gs in gB.items():
                cross = pair_stats(gN, gs, wm, False)[0]
                bb = pair_stats(gs, gs, wm, True)[0]
                row[f"lambda0_{name}"] = cross / nn_b if abs(nn_b) > 1e-30 else float("nan")
                row[f"lambda0_tilde_{name}"] = cross / np.sqrt(max(nn_b, 1e-30) * max(bb, 1e-30))
            out["blocks"][bname] = row
    return out


# ---------------------------------------------------------------------------
# one-step function-space check
# ---------------------------------------------------------------------------

def eval_losses(model, sets: dict[str, tuple[np.ndarray, int]], device: str) -> dict[str, float]:
    out = {}
    with torch.no_grad():
        for name, (seqs, loss_from) in sets.items():
            x = torch.tensor(seqs[:, :-1], device=device)
            y = torch.tensor(seqs[:, 1:], device=device)
            logits = model(x)
            if loss_from > 0:
                logits = logits[:, loss_from - 1 :, :]
                y = y[:, loss_from - 1 :]
            out[name] = float(
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                ).item()
            )
    return out


def one_step_deltas(model, ft_cfg, eval_sets, device: str) -> dict:
    base = eval_losses(model, eval_sets, device)
    res: dict = {"base_losses": base}

    # actual first AdamW FT step
    m = deepcopy(model)
    opt = torch.optim.AdamW(m.parameters(), lr=FT_LR_ADAM)
    _, x, y = batch(ft_cfg, FT_BATCH, 500_000 + SEED * 10_000 + 1, device)
    logits = m(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    res["adam_first"] = {k: v - base[k] for k, v in eval_losses(m, eval_sets, device).items()}

    # mean-gradient SGD steps at two lrs (linearity)
    gm = deepcopy(model)
    gm.zero_grad()
    for s in range(1, 9):
        _, x, y = batch(ft_cfg, FT_BATCH, 500_000 + SEED * 10_000 + s, device)
        logits = gm(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        ) / 8.0
        loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in gm.named_parameters() if p.grad is not None}
    for tag, lr in (("sgd_1e3", 1e-3), ("sgd_5e4", 5e-4)):
        m2 = deepcopy(model)
        with torch.no_grad():
            for n, p in m2.named_parameters():
                if n in grads:
                    p.add_(grads[n], alpha=-lr)
        res[tag] = {k: v - base[k] for k, v in eval_losses(m2, eval_sets, device).items()}
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    cfg = headline_lowprior_cfg(PI0)
    ft_cfg = special_sfp.SpecialSFPConfig(**{**asdict(cfg), "pi": (1.0, 0.0, 0.0, 0.0)})
    ad_cfg = special_sfp.SpecialSFPConfig(**{**asdict(cfg), "pi": (0.0, 0.0, 1.0, 0.0)})

    print(f"[lambda0] pretraining {BASE_STEPS} steps on {device} (pi0={PI0})", flush=True)
    model = make_headline_model(cfg, SEED, device)
    train_steps(model, cfg, 1e-3, BASE_STEPS, 100_000 + SEED * 10_000, device)
    t_pre = time.time() - t0
    print(f"[lambda0] pretrain done in {t_pre:.0f}s", flush=True)

    # --- gradient sets, prompt-aligned chunks
    prompts = sample_eval_prompts(cfg, "O", PROMPT_K, PROMPT_LEN, PROMPT_SET_SEED)
    per_chunk = PROMPT_K // N_CHUNKS
    masks = block_slices(model)

    gN, gNA = [], []
    gB: dict[str, list[np.ndarray]] = {"B_M": [], "B_A": [], "B_flip": []}
    losses0: dict[str, list[float]] = {}
    probe_cache: dict[str, list[np.ndarray]] = {"B_M": [], "B_A": [], "B_flip": []}
    for c in range(N_CHUNKS):
        obs, _, _ = batch(ft_cfg, NARROW_CHUNK, 810_000 + SEED * 10_000 + c, device)
        g, l = flat_grad(model, obs, device, 0)
        gN.append(g)
        losses0.setdefault("N", []).append(l)
        obs, _, _ = batch(ad_cfg, NARROW_CHUNK, 820_000 + SEED * 10_000 + c, device)
        g, l = flat_grad(model, obs, device, 0)
        gNA.append(g)
        losses0.setdefault("NA", []).append(l)
        pch = prompts[c * per_chunk : (c + 1) * per_chunk]
        for name, leaf in (("B_M", MO_LEAF), ("B_A", AO_LEAF), ("B_flip", None)):
            seqs = sector_continuations(
                pch, cfg, leaf, N_PER_PROMPT, GEN_LEN, 830_000 + SEED * 10_000 + 97 * c
            )
            probe_cache[name].append(seqs)
            g, l = flat_grad(model, seqs, device, PROMPT_LEN)
            gB[name].append(g)
            losses0.setdefault(name, []).append(l)
        print(f"[lambda0] chunk {c + 1}/{N_CHUNKS} grads done ({time.time() - t0:.0f}s)", flush=True)

    # --- Adam second-moment estimate from the narrow FT stream
    v = np.zeros_like(gN[0])
    for s in range(V_BATCHES):
        obs, _, _ = batch(ft_cfg, FT_BATCH, 840_000 + SEED * 10_000 + s, device)
        g, _ = flat_grad(model, obs, device, 0)
        v += g * g
    v /= V_BATCHES
    w_plain = np.ones_like(v)
    w_adam = 1.0 / (np.sqrt(v) + ADAM_EPS)
    # Capped variant: coordinates the narrow stream never moves have v ~ 0 and weight
    # ~ 1/eps = 1e8, which corrupts the SYMMETRIC readouts (probe-side P-norms) with
    # noise from parameters Adam-on-narrow-data would barely touch. The asymmetric
    # lambda0_* entries (<g_b, P^-1 g_n>/<g_n', P^-1 g_n>) are the true Adam transfer
    # objects and are reported from BOTH tables; if they disagree, trust neither.
    w_adam_capped = np.minimum(w_adam, np.percentile(w_adam, 99.5))
    print(f"[lambda0] v estimated over {V_BATCHES} batches ({time.time() - t0:.0f}s)", flush=True)

    results = {
        "plain": transfer_table(gN, gNA, gB, w_plain, masks),
        "adam_preconditioned": transfer_table(gN, gNA, gB, w_adam, masks),
        "adam_preconditioned_capped": transfer_table(gN, gNA, gB, w_adam_capped, masks),
        "mean_losses_theta0": {k: float(np.mean(vs)) for k, vs in losses0.items()},
        "naive_cosine_BM": float(
            np.dot(np.mean(gN, axis=0), np.mean(gB["B_M"], axis=0))
            / (np.linalg.norm(np.mean(gN, axis=0)) * np.linalg.norm(np.mean(gB["B_M"], axis=0)))
        ),
    }

    # --- one-step function-space check on held-out data
    obs_hold, _, _ = batch(ft_cfg, NARROW_CHUNK, 850_000 + SEED, device)
    eval_sets = {
        "N_holdout": (obs_hold, 0),
        "B_M": (np.concatenate(probe_cache["B_M"][:4], axis=0), PROMPT_LEN),
        "B_A": (np.concatenate(probe_cache["B_A"][:4], axis=0), PROMPT_LEN),
        "B_flip": (np.concatenate(probe_cache["B_flip"][:4], axis=0), PROMPT_LEN),
    }
    results["one_step"] = one_step_deltas(model, ft_cfg, eval_sets, device)
    for tag in ("adam_first", "sgd_1e3", "sgd_5e4"):
        d = results["one_step"][tag]
        if abs(d["N_holdout"]) > 1e-12:
            results["one_step"][f"{tag}_lambda_fn"] = {
                k: d[k] / d["N_holdout"] for k in ("B_M", "B_A", "B_flip")
            }
    results["one_step"]["linearity_ratio_sgd"] = {
        k: (results["one_step"]["sgd_1e3"][k] / results["one_step"]["sgd_5e4"][k])
        if abs(results["one_step"]["sgd_5e4"][k]) > 1e-12
        else float("nan")
        for k in ("N_holdout", "B_M", "B_A", "B_flip")
    }

    meta = {
        "seed": SEED,
        "smoke": SMOKE,
        "base_steps": BASE_STEPS,
        "pi0": PI0,
        "n_chunks": N_CHUNKS,
        "narrow_chunk": NARROW_CHUNK,
        "prompt_k": PROMPT_K,
        "n_per_prompt": N_PER_PROMPT,
        "gen_len": GEN_LEN,
        "v_batches": V_BATCHES,
        "device": str(device),
        "pretrain_s": t_pre,
        "elapsed_s": time.time() - t0,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    out_path = OUT / f"lambda0_toy_seed{SEED}.json"
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n")
    print(f"[lambda0] wrote {out_path}")

    # --- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for i, (tag, tab) in enumerate((("plain", results["plain"]), ("Adam-precond. (capped)", results["adam_preconditioned_capped"]))):
        vals = [tab["lambda0_tilde_B_M"], tab["lambda0_tilde_B_A"], tab["lambda0_tilde_B_flip"], tab["contrast_cos"]]
        ax.bar(np.arange(4) + 0.38 * i, vals, width=0.36, label=tag)
    ax.axhline(1 / 3, color="k", ls="--", lw=1, label="integrated λ ≈ 1/3")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(np.arange(4) + 0.19)
    ax.set_xticklabels(
        ["broad misaligned\n(MO probe)", "broad aligned\n(AO probe)", "domain flip\n(MD-after-O)", "double contrast\ncos"]
    )
    ax.set_ylabel("first-order transfer per unit narrow step")
    ax.set_title(f"Base-model gradient transfer, toy SFP (seed {SEED})")
    ax.legend()

    ax = axes[1]
    blocks = list(results["adam_preconditioned_capped"]["blocks"].keys())
    vals = [results["adam_preconditioned_capped"]["blocks"][b]["lambda0_tilde_B_M"] for b in blocks]
    ax.bar(blocks, vals, color="tab:red")
    ax.axhline(1 / 3, color="k", ls="--", lw=1)
    ax.set_ylabel("λ̃₀ (Adam-precond.) toward broad-misaligned probe")
    ax.set_title("Per-block transfer to broad misalignment")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / f"lambda0_toy_seed{SEED}.png", dpi=150)
    print(f"[lambda0] wrote figure; total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

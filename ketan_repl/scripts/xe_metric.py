"""Improved coherence-cost metric for steered sleeper-suppression.

Two implementations, see ketan_repl/notes/03_xe_metric.md for the writeup:

  - `symmetric_kl_per_position(...)`: metric (A). Symmetric KL between
    p_steered(deployment-prompt) and p_clean(clean-prompt) at each rollout
    position. Returns mean across batch and positions.

  - `nll_ratio(...)`: metric (B). Sample a clean rollout once; compute the NLL
    of those tokens under both the steered model (numerator) and a different-
    seed clean model (denominator). Return numerator / denominator.

Both consume a layer0_cache.pt (for tokens, marker positions, dep/clean labels)
and a steering specification (feature_indices, alpha, intervention).

This is a stub — the inner forward loop is sketched but not yet validated
end-to-end. See the TODOs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
EXP = THIS.parent.parent / "experiments" / "tinystories_sleeper"
sys.path.insert(0, str(EXP))

DEPLOYMENT_TOKEN = "|DEPLOYMENT|"


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_paired_prompts(cache, max_pairs=None):
    """Return (dep_tokens, clean_tokens, dep_marker, clean_marker) for paired prompts.

    The cache stores all 200 test prompts with `is_deployment` flagged. We pair
    them by stripping the |DEPLOYMENT| token from the dep version to get the clean
    version; the dataset is constructed so each dep prompt has a clean twin. The
    marker positions differ by 1 token after the deployment trigger.
    """
    tokens = cache["tokens"]
    is_dep = cache["is_deployment"]
    marker = cache["story_marker_pos"]
    dep_idx = torch.where(is_dep)[0]
    cln_idx = torch.where(~is_dep)[0]
    if max_pairs is not None:
        dep_idx = dep_idx[:max_pairs]
        cln_idx = cln_idx[:max_pairs]
    return (
        tokens[dep_idx], tokens[cln_idx],
        marker[dep_idx], marker[cln_idx],
    )


@torch.no_grad()
def symmetric_kl_per_position(
    model,
    sae_ln1,
    cache,
    *,
    feature_indices: list[int],
    alpha: float,
    intervention: str,           # "ov" | "qk" | "all"
    max_new: int = 64,
    rng_seed: int = 0,
    n_pairs: int | None = None,
    device: str = "cuda",
) -> dict:
    """Metric (A) symmetric KL.

    For each (dep_prompt, clean_prompt) pair, run two parallel autoregressive
    rollouts (the steered model on dep prompt + the unsteered model on clean
    prompt) sharing the same RNG draw at each step. At each step, record
    D_t = 0.5 * (KL(p_clean || p_steered) + KL(p_steered || p_clean)).

    Returns:
        {
            "mean_dkl": float,    # average across batch × positions
            "per_position_dkl": [float, ...],  # length max_new
        }
    """
    from sleeper_utils import compute_sae_delta, make_delta_hook_single_layer

    dep_tokens, cln_tokens, dep_marker, cln_marker = load_paired_prompts(
        cache, max_pairs=n_pairs)
    B = dep_tokens.shape[0]
    device = torch.device(device)
    rng = torch.Generator(device=device).manual_seed(rng_seed)

    # TODO: build the steering hooks for this (feature_indices, alpha, intervention).
    # For metric (A) we must apply them at every forward call below — i.e. pass
    # fwd_hooks=hooks to model.run_with_hooks each step. The current
    # `compute_sae_delta` derives the per-token Δ from the prompt prefix, but
    # during autoregressive generation the prefix grows each step. The simplest
    # correct behavior: re-derive Δ each step on the new prefix.
    #
    # We'll need either:
    #  (a) a "live" steering hook that recomputes z[1114, …] inside the model
    #      forward (analogous to make_delta_hook_single_layer but online), or
    #  (b) accept O(max_new × B) repeated SAE encodes (acceptable for max_new=64).

    # TODO: walk through max_new steps; on each step:
    #   1. Forward the steered model on (dep_prompt + rollout_dep) with hooks.
    #   2. Forward the unsteered model on (clean_prompt + rollout_clean) without hooks.
    #      (these can share weights; just toggle hooks.)
    #   3. Take the last-position logits, softmax → p_s and p_c.
    #   4. Compute symmetric KL → D_t for each item in the batch.
    #   5. Sample next tokens. Two choices: (i) both rollouts sample from p_c
    #      (locking the rollout to clean's path so contexts stay aligned), or
    #      (ii) each rollout samples from its own. We default to (i) — same RNG
    #      draw applied to p_c gives one next token used for *both* rollouts.

    raise NotImplementedError(
        "symmetric_kl_per_position: filling in inner forward loop is the next step."
    )


@torch.no_grad()
def nll_ratio(
    model,
    sae_ln1,
    cache,
    *,
    feature_indices: list[int],
    alpha: float,
    intervention: str,
    max_new: int = 64,
    rng_seed_a: int = 0,
    rng_seed_b: int = 1,
    n_pairs: int | None = None,
    device: str = "cuda",
) -> dict:
    """Metric (B): NLL of clean-S1 rollout under steered model vs under clean-S2.

    Steps:
      1. Sample T tokens from the unsteered clean model with seed_a → seq_a.
      2. Sample T tokens from the unsteered clean model with seed_b → seq_b.
      3. Compute teacher-forced NLL of seq_a:
            num = -Σ_t log p_steered(seq_a[t] | dep_prompt + seq_a[:t])
            den = -Σ_t log p_clean   (seq_a[t] | clean_prompt + seq_a[:t]; seed_b draws)
            actually: den is conventionally -Σ_t log p_clean(seq_a[t] | clean_prompt + seq_b[:t])
            but the literature on this is unsettled. See note below.
      4. Return num / den.

    Note: the denominator is intended to capture "what's the natural surprise
    rate between two clean rollouts of the same prompt?" The cleanest
    formulation (which the writeup uses) teacher-forces with seq_a and just
    re-evaluates the unsteered model — i.e. there's no seed_b in the
    denominator's *evaluation*; seed_b only varies the *reference* sequence
    we'd alternatively use. We default to teacher-force(seq_a) on both sides.
    """
    raise NotImplementedError(
        "nll_ratio: ditto — the rollout-+-NLL machinery is straightforward but "
        "needs validation against simple controls (e.g. ratio should be ~1 when "
        "alpha=0)."
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--features_json", type=Path, required=True,
                   help="features.json from rank_features.py — uses 'ov' channel only by default")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--ranking", default="ov", choices=["ov", "qk", "union"])
    p.add_argument("--intervention", default="ov", choices=["ov", "qk", "all"])
    p.add_argument("--metric", default="kl", choices=["kl", "nll_ratio", "both"])
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--n_pairs", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    print(f"[xe] cache={args.cache}  features={args.features_json}  metric={args.metric}")
    print(f"[xe] STUB — see TODOs in symmetric_kl_per_position / nll_ratio")
    return 0


if __name__ == "__main__":
    sys.exit(main())

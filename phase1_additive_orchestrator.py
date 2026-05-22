"""
Phase 1 additive-recipe orchestrator: 1 (em_model × eval_seed) stream over
all 5 pre-trained SAEs (Nura's L24 ln1 + our 4 surrounding from medical session).

Loads Qwen + EM LoRA once, iterates 5 SAEs:

  - L24_ln1_nura     Nura-J/Qwen2.5-14B_SAE_ln1.normalised    hook=blocks.24.ln1.hook_normalized
  - L24_resid_pre    dmanningcoe/em-repl-2026-05-07/phase3_benchmark/sae/resid_pre_L24/final
  - L24_resid_mid    dmanningcoe/em-repl-2026-05-07/phase3_benchmark/sae/resid_mid_L24/final
  - L24_resid_post   dmanningcoe/em-repl-2026-05-07/phase3_benchmark/sae/resid_post_L24/final
  - L25_ln1          dmanningcoe/em-repl-2026-05-07/phase3_benchmark/sae/ln1_normalised_L25/final

Per SAE: top-50 features by accumulated |f| across the 8 EM prompts, then
α-sweep ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0} via `generate_with_hooks_batch` with
per-prompt seeds [eval_seed, eval_seed+1, …, eval_seed+7] (the per-prompt seed
convention is what made Phase 0 pass — see
`docs/dmitry/c6_em/2026-05-08_em_repl_finance_sports/phase0_fastpath_validation.md`).

Output schema is Nura-compatible so `judge_multiseed.py` (parallel-judge variant)
can score everything.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch

# Quieter logs: silence cosmetic warnings, every print auto-flushes.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print = functools.partial(print, flush=True)  # type: ignore[assignment]
_T0 = time.time()


def _step(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}")


# (id, hook_name, sae_source_kind, sae_path, layer)
SAE_SPECS = [
    ("L24_ln1_nura",  "blocks.24.ln1.hook_normalized",  "nura",
     "Nura-J/Qwen2.5-14B_SAE_ln1.normalised",         24),
    ("L24_resid_pre", "blocks.24.hook_resid_pre",       "sae_lens",
     "phase3_benchmark/sae/resid_pre_L24/final",      24),
    ("L24_resid_mid", "blocks.24.hook_resid_mid",       "sae_lens",
     "phase3_benchmark/sae/resid_mid_L24/final",      24),
    ("L24_resid_post","blocks.24.hook_resid_post",      "sae_lens",
     "phase3_benchmark/sae/resid_post_L24/final",     24),
    ("L25_ln1",       "blocks.25.ln1.hook_normalized",  "sae_lens",
     "phase3_benchmark/sae/ln1_normalised_L25/final", 25),
]
SURROUNDING_REPO = "dmanningcoe/em-repl-2026-05-07"


def load_em_model(em_model: str, device: str = "cuda"):
    """Load Qwen-14B (base or merged with an EM LoRA)."""
    EM_MODELS = {
        "base":    "Qwen/Qwen2.5-14B-Instruct",
        "finance": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
        "medical": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        "sports":  "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
    }
    name = EM_MODELS[em_model]
    _step(f"[load] {em_model} → {name}")
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    if em_model == "base":
        hf = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        model = HookedTransformer.from_pretrained_no_processing(
            "Qwen/Qwen2.5-14B-Instruct", hf_model=hf, device=device, dtype=torch.bfloat16,
        )
        del hf
        torch.cuda.empty_cache()
        return model

    from peft import PeftModel
    base_hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu",
    )
    lora_hf = PeftModel.from_pretrained(base_hf, name)
    merged_hf = lora_hf.merge_and_unload()
    del base_hf, lora_hf
    model = HookedTransformer.from_pretrained_no_processing(
        "Qwen/Qwen2.5-14B-Instruct", hf_model=merged_hf, device=device, dtype=torch.bfloat16,
    )
    del merged_hf
    torch.cuda.empty_cache()
    return model


def load_sae_local(sae_id: str, source_kind: str, sae_path: str, layer: int, device: str):
    """Download (if needed) + load an SAE. Returns object with `.encode`, `.W_dec`."""
    if source_kind == "nura":
        from fra.sae_lens_wrapper import QwenLn1SAE
        sae = QwenLn1SAE(sae_path, layer=layer, device=device)
        sae.eval = lambda: sae
        return sae
    # sae_lens checkpoint stored on HF under SURROUNDING_REPO
    from huggingface_hub import snapshot_download
    _step(f"  [{sae_id}] snapshot_download patterns={sae_path}/*")
    local = snapshot_download(
        repo_id=SURROUNDING_REPO, repo_type="model",
        allow_patterns=[f"{sae_path}/*"],
    )
    full_path = Path(local) / sae_path
    from sae_lens import SAE
    sae = SAE.load_from_pretrained(str(full_path), device=device)
    sae.eval()
    return sae


@torch.no_grad()
def rank_features(model, sae, hook_name: str, prompts, top_k: int = 50,
                  max_length: int = 128) -> list[int]:
    """Top-k features by accumulated |f| across the prompt set."""
    d_sae = sae.cfg.d_sae if hasattr(sae, "cfg") else sae.W_enc.shape[1]
    device = sae.W_enc.device if hasattr(sae, "W_enc") else next(sae.parameters()).device
    scores = torch.zeros(d_sae, device=device, dtype=torch.float32)
    for prompt in prompts:
        ids = torch.tensor(model.tokenizer.encode(prompt)[:max_length], device=device).unsqueeze(0)
        _, cache = model.run_with_cache(ids, names_filter=[hook_name])
        acts = cache[hook_name][0].float()
        feats = sae.encode(acts)
        scores += feats.abs().sum(dim=0)
        del cache, acts, feats
        torch.cuda.empty_cache()
    return torch.topk(scores, top_k).indices.cpu().tolist()


@torch.no_grad()
def rank_features_by_diff(sae, diff_vector: torch.Tensor, top_k: int = 50) -> list[int]:
    """Top-k SAE features by cos-sim(W_dec[f], Δa).  Arditi-style ranking.

    Δa is the precomputed (EM_mean − base_mean) activation diff at the same
    hookpoint as the SAE was trained on.  Features whose decoder direction
    most aligns with Δa are most "EM-specific" in that subspace.
    """
    W_dec = sae.W_dec  # [d_sae, d_model] — keep native dtype to avoid 2GB intermediate on bf16 SAEs
    delta = diff_vector.to(W_dec.device).to(W_dec.dtype)
    dots = W_dec @ delta  # [d_sae]
    norms = W_dec.norm(dim=-1) + 1e-12  # [d_sae]
    delta_norm = delta.norm() + 1e-12
    cos_sim = (dots / (norms * delta_norm)).float()
    return torch.topk(cos_sim, top_k).indices.cpu().tolist()


def load_diff_vector(domain: str, hookpoint_id: str, layer: int,
                     hf_repo: str = "dmanningcoe/fra-phase1-steering-data") -> torch.Tensor:
    """Download Qwen14B_diff_vectors/<domain>/<hookpoint_id>.pt (shape [n_layers, d_model]),
    return the [d_model] slice for the requested layer.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(hf_repo, f"Qwen14B_diff_vectors/{domain}/{hookpoint_id}.pt",
                           repo_type="dataset")
    stack = torch.load(path, map_location="cpu")
    _step(f"  [diff] loaded {domain}/{hookpoint_id}.pt "
          f"shape={tuple(stack.shape)} L{layer}_norm={stack[layer].norm().item():.3f}")
    return stack[layer]


def make_additive_hook_batched(sae, feature_indices, alpha: float):
    """ACTIVATION-WEIGHTED additive steering at any residual hookpoint, batch-aware.

    act += (alpha-1) * sum_lambda f_lambda * W_dec_lambda
    where f_lambda = sae.encode(act)[lambda] — the feature's current activation.

    Use this when feature_indices were chosen because they FIRE on the loaded
    model (e.g. top-k by |activation|). For features ranked by a method that
    doesn't guarantee runtime activation (e.g. cos-sim with Δa), prefer
    `make_constant_additive_hook_batched` instead — otherwise inactive
    features contribute zero regardless of alpha.
    """
    device = sae.W_dec.device
    feat_idx = torch.tensor(feature_indices, device=device, dtype=torch.long)
    W_dec_F = sae.W_dec[feat_idx].detach()
    factor = float(alpha - 1.0)

    def _steer(act, hook):
        with torch.no_grad():
            feats = sae.encode(act.float())
            feats_F = feats.index_select(-1, feat_idx)
            delta = torch.einsum("bsf,fd->bsd", feats_F * factor, W_dec_F).to(act.dtype)
            return act + delta
    return _steer


def make_constant_additive_hook_batched(sae, feature_indices, alpha: float):
    """CONSTANT-MAGNITUDE additive steering: delta = alpha * sum(W_dec[feature_indices]).

    The delta is the same at every (batch, seq) position, independent of the
    current residual. This is the Arditi-style protocol, extended to bundle-N
    features: add the sum of decoder directions, scaled by alpha.

    Use this when feature_indices were selected by a method that doesn't
    guarantee runtime activation (e.g. cos-sim ranking against Δa). The
    activation-weighted hook produces ~0 delta for inactive features even
    at large alpha; the constant-magnitude hook forces the steering direction
    regardless.
    """
    device = sae.W_dec.device
    feat_idx = torch.tensor(feature_indices, device=device, dtype=torch.long)
    W_dec_sum = sae.W_dec[feat_idx].sum(dim=0).detach()  # [d_model]
    delta = (float(alpha) * W_dec_sum).detach()

    def _steer(act, hook):
        return act + delta.to(act.dtype)
    return _steer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-model", required=True, choices=["base", "finance", "medical", "sports"])
    p.add_argument("--eval-seed", type=int, required=True,
                   help="Base seed; per-prompt seeds are [seed, seed+1, ..., seed+7]")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--top-k-features", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True,
                   help="Per-stream output directory")
    p.add_argument("--saes", nargs="+", default=None,
                   help="If set, restrict to these SAE ids (else all 5).")
    p.add_argument("--diff-source", default=None,
                   choices=["medical", "finance", "sports"],
                   help="Rank features by cos-sim to precomputed Δa (EM-vs-base diff) "
                        "for this domain. Requires --em-model=base; downloads the "
                        "diff vector from HF dataset Qwen14B_diff_vectors/<domain>/<hookpoint>.pt. "
                        "If unset, falls back to |activation|-based ranking on the loaded model.")
    p.add_argument("--additive-mode", default="activation_weighted",
                   choices=["activation_weighted", "constant"],
                   help="Hook math: 'activation_weighted' (default) = (α-1) × Σ f_λ × W_dec[λ], "
                        "scales each feature's contribution by its runtime activation. "
                        "'constant' = α × Σ W_dec[λ], constant delta independent of activations. "
                        "Use 'constant' when features are ranked by diff/cos-sim (Arditi-style).")
    args = p.parse_args()
    if args.diff_source and args.em_model != "base":
        raise SystemExit(
            f"--diff-source={args.diff_source} only makes sense with --em-model=base "
            f"(was: {args.em_model}). The diff-based ranking steers the BASE model "
            "with EM-derived feature directions."
        )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    per_prompt_seeds = [args.eval_seed + i for i in range(args.n_prompts)]

    _step("=== Phase 1 additive orchestrator ===")
    _step(f"  em_model={args.em_model} eval_seed={args.eval_seed} "
          f"alphas={len(args.alphas)} prompts={len(prompts)} top-k={args.top_k_features}")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    _step(f"[load] model ready in {time.time() - t_start:.1f}s")

    specs = SAE_SPECS if args.saes is None else [s for s in SAE_SPECS if s[0] in args.saes]

    for sae_id, hook_name, source_kind, sae_path, layer in specs:
        _step(f"=== SAE: {sae_id} hook={hook_name} ===")
        t_sae = time.time()
        sae = load_sae_local(sae_id, source_kind, sae_path, layer, args.device)
        _step(f"  SAE loaded in {time.time() - t_sae:.1f}s "
              f"(d_in={getattr(sae, 'd_in', None) or sae.cfg.d_in}, "
              f"d_sae={getattr(sae, 'd_sae', None) or sae.cfg.d_sae})")

        t_rank = time.time()
        if args.diff_source:
            # Map hook_name → hookpoint_id used in the diff archive
            if "ln1" in hook_name: hp_id = "ln1"
            elif "resid_mid" in hook_name: hp_id = "resid_mid"
            elif "resid_post" in hook_name: hp_id = "resid_post"
            else: raise SystemExit(f"can't map hook_name={hook_name} to diff archive hp_id")
            delta = load_diff_vector(args.diff_source, hp_id, layer)
            feature_ids = rank_features_by_diff(sae, delta, top_k=args.top_k_features)
            rank_method = f"diff-cossim({args.diff_source})"
        else:
            feature_ids = rank_features(model, sae, hook_name, prompts,
                                        top_k=args.top_k_features)
            rank_method = "|activation|"
        _step(f"  ranked top-{args.top_k_features} by {rank_method} "
              f"in {time.time() - t_rank:.1f}s (head5={feature_ids[:5]})")

        qualitative = []
        t_gen = time.time()
        hook_factory = (
            make_constant_additive_hook_batched
            if args.additive_mode == "constant"
            else make_additive_hook_batched
        )
        _step(f"  sweep start: mode={args.additive_mode} hook={hook_factory.__name__}")
        n_alpha = len(args.alphas)
        for idx, alpha in enumerate(args.alphas, start=1):
            t_cell = time.time()
            hooks = [(hook_name, hook_factory(sae, feature_ids, alpha))]
            responses = generate_with_hooks_batch(
                model, tokenizer, prompts,
                fwd_hooks=hooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=per_prompt_seeds,
            )
            _step(f"  α[{idx:2d}/{n_alpha}] = {alpha:+5.1f}  ({time.time() - t_cell:.1f}s)")
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                qualitative.append({
                    "seed": per_prompt_seeds[i],
                    "scale": float(alpha),
                    "prompt_idx": i,
                    "prompt": prompt,
                    "condition": f"sae_resid_a{alpha}",
                    "response": response,
                    "alignment": 0,
                    "coherence": 0,
                    "sae_id": sae_id,
                    "hook_name": hook_name,
                    "em_model": args.em_model,
                    "eval_seed_base": args.eval_seed,
                })
            torch.cuda.empty_cache()
        _step(f"  sweep done in {time.time() - t_gen:.1f}s")

        out_path = out_root / f"qualitative_{sae_id}_{args.em_model}_evalseed{args.eval_seed}_top{args.top_k_features}.json"
        out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
        _step(f"  wrote {out_path} ({len(qualitative)} entries)")

        del sae
        torch.cuda.empty_cache()

    _step(f"=== TOTAL stream time: {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())

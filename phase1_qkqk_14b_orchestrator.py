"""Phase 1 QK→QK/OV→OV/QK→OV steering on Qwen-2.5-7B (+ EM LoRA), using a
locally-trained Arditi-style SAE at ln1.hook_normalized.

Mirrors `phase1_fra_orchestrator.py` shape for 14B exactly, with three
adaptations:

  1. Model: Qwen-2.5-7B-Instruct base or + bad-medical LoRA (merged).
  2. SAE: Arditi-trained BatchTopKSAE at ln1 (loaded from a local dir
     containing `ae.pt` + `config.json` produced by their training loop).
     A thin adapter (`_ArditiSAEAdapter`) exposes `encode/decode/W_dec` in
     the same shape Nura's wrapper does, so the FRA code requires no
     changes.
  3. Layer/head: defaults to L15 (head selected from a head-ablation run
     on a fresh A40 — see `run_experiments.py --task head_ablation`).

Output schema is identical to the 14B orchestrator, drops straight into
phase1_judge_and_combine.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch


EM_MODELS_7B = {
    "finance": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    "base":    "Qwen/Qwen2.5-14B-Instruct",
}


def load_em_model(em_model: str, device: str = "cuda",
                  base_model_id: str | None = None,
                  em_model_id: str | None = None):
    """Load base + (optionally) an EM LoRA, merge, hand to TL. Generalized to
    accept base_model_id/em_model_id (so new EM models — e.g. medical — work via
    --em-model-id without editing the dict). LoRA-merge tried first, full-model
    fallback. Mirrors phase1_grid_14b_orchestrator.load_em_model — the routing
    orchestrator imports THIS one, so it must accept the same kwargs."""
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    base_id = base_model_id or EM_MODELS_7B["base"]
    name = em_model_id or EM_MODELS_7B.get(em_model, em_model)
    print(f"[load] {em_model} → {name}  (base={base_id})")
    if em_model == "base":
        hf = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        model = HookedTransformer.from_pretrained_no_processing(
            base_id, hf_model=hf, device=device, dtype=torch.bfloat16,
        )
        del hf
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        try:
            lora = PeftModel.from_pretrained(base, name)
            merged = lora.merge_and_unload()
            del base, lora
        except (ValueError, OSError) as e:
            print(f"  PeftModel load failed ({e!r}); loading {name} as a full model", flush=True)
            del base; torch.cuda.empty_cache()
            merged = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.bfloat16, device_map="cpu",
            )
        model = HookedTransformer.from_pretrained_no_processing(
            base_id, hf_model=merged, device=device, dtype=torch.bfloat16,
        )
        del merged
    torch.cuda.empty_cache()
    return model


class _ArditiSAEAdapter:
    """Thin wrapper around a `dictionary_learning.BatchTopKSAE` instance
    that exposes the same shape interface as our `QwenLn1SAE` wrapper:

      .d_in, .d_sae
      .W_dec  — shape (d_sae, d_in) so W_dec[fid] gives a d_in-shaped vector
      .encode(x), .decode(features)

    Arditi's native decoder weight is (d_in, d_sae); we transpose at load
    so downstream code is unchanged.
    """
    def __init__(self, sae, force_topk=None):
        self._sae = sae
        w = sae.decoder.weight.detach()  # (d_in, d_sae)
        self.W_dec = w.T.contiguous()    # (d_sae, d_in)
        self.d_in, self.d_sae = w.shape
        # k for exact top-k gating. BatchTopK's learned eval-threshold can be
        # miscalibrated (we observed L0≈19765 ≫ k=64, l2_ratio≈20×, FVU<0 in
        # eval_results) — forcing exact top-k recovers the intended behaviour
        # (top-k of the threshold's superset == the true top-k).
        self._k = force_topk or getattr(sae, "k", None)
        if hasattr(self._k, "item"):
            self._k = int(self._k.item())
        # FRA pipeline reads .b_dec when hook_point contains "resid"; expose
        # Arditi's pre-encoder bias as a zero vector if absent.
        b = getattr(sae, "b_dec", None)
        if b is None:
            b = torch.zeros(self.d_in, device=w.device, dtype=w.dtype)
        self.b_dec = b.detach()
        # RMSNorm gain γ (= blocks[L].ln1.w). The SAE was trained on the HF
        # input_layernorm OUTPUT = (x/rms)·γ (post-gain), but TL's
        # ln1.hook_normalized is x/rms (PRE-gain). Callers pass pre-gain acts,
        # so encode() multiplies by γ to match the training distribution.
        # (Diagnosed 2026-05-26: var-expl +0.63 post-gain vs −2.8 pre-gain.)
        self._gamma = None  # set by caller after the model is loaded

    def encode(self, x):
        if self._gamma is not None:
            x = x * self._gamma
        f = self._sae.encode(x)
        # Bypass the (miscalibrated) learned threshold with exact top-k.
        if self._k and f.shape[-1] > self._k:
            topv, topi = f.topk(self._k, dim=-1)
            f = torch.zeros_like(f).scatter_(-1, topi, topv)
        return f

    def decode(self, features):
        return self._sae.decode(features)


def load_arditi_sae_from_dir(sae_dir: Path, device: str = "cuda") -> _ArditiSAEAdapter:
    """Load an Arditi-trained SAE from a local directory (their
    `run_from_config.py` save_dir layout: contains `ae.pt` + `config.json`).
    Returns an adapter compatible with our 14B FRA code."""
    from dictionary_learning.utils import load_dictionary
    print(f"[load] Arditi SAE from {sae_dir}")
    sae, cfg = load_dictionary(str(sae_dir), device=device)
    sae.eval()
    ad = _ArditiSAEAdapter(sae)
    print(f"[load] SAE d_in={ad.d_in}, d_sae={ad.d_sae}  W_dec shape={tuple(ad.W_dec.shape)}")
    return ad


@torch.no_grad()
def mc_forced_choice_under_hooks(model, mc_data, hooks):
    """Arditi's single-token forced-choice MC eval, run UNDER the given FRA
    fwd_hooks. For each MC item: one forward (batch=1, no padding → TL-safe),
    last-token log-softmax, then sum P over the misaligned / aligned / A / B
    label-token sets. Mirrors phase1_arditi_mc_peritem._process_batch_peritem
    but on the TL model so the qk→qk / ov hooks apply during the forward."""
    mc_prompts, mis_toks, ali_toks, a_set, b_set = mc_data
    dev = model.cfg.device
    rows = []
    for prompt, mis, ali in zip(mc_prompts, mis_toks, ali_toks):
        toks = prompt if hasattr(prompt, "dim") else torch.tensor(prompt)
        toks = toks.to(dev)
        if toks.dim() == 1:
            toks = toks.unsqueeze(0)
        logits = model.run_with_hooks(toks, fwd_hooks=hooks, return_type="logits")
        lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
        ex = lambda S: float(sum(torch.exp(lp[t]).item() for t in S))
        rows.append({"p_a": ex(a_set), "p_b": ex(b_set),
                     "p_mis": ex(mis), "p_ali": ex(ali)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-model", default="finance", choices=list(EM_MODELS_7B))
    p.add_argument("--eval-seed", type=int, required=True,
                   help="Base seed; per-prompt seeds are [seed, seed+1, …, seed+7]")
    p.add_argument("--sae-dir", required=True,
                   help="Directory containing the Arditi-trained SAE "
                        "(ae.pt + config.json from their run_from_config.py)")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--head", type=int, required=True,
                   help="Attention head to decompose at this layer. "
                        "Run head_ablation first to pick.")
    p.add_argument("--hook-point", default="ln1.hook_normalized")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True)
    p.add_argument("--mc-eval", action="store_true",
                   help="Also run Arditi's single-token forced-choice MC eval UNDER each "
                        "FRA hook condition (saves mc_FRA_*.json with per-item P(mis)/P(ali)/P(A)/P(B)).")
    p.add_argument("--osemf-root", default="/workspace/osemf",
                   help="safety-research/open-source-em-features checkout (for MC questions + label tokens).")
    args = p.parse_args()

    out_root = Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    per_prompt_seeds = [args.eval_seed + i for i in range(args.n_prompts)]

    print("=== Phase 1 QK→QK 7B orchestrator ===")
    print(f"  em_model         : {args.em_model}")
    print(f"  layer × head     : L{args.layer} H{args.head}  hook={args.hook_point}")
    print(f"  sae_dir          : {args.sae_dir}")
    print(f"  eval_seed (base) : {args.eval_seed}")
    print(f"  alphas           : {args.alphas}")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    print(f"[load] model loaded in {time.time()-t_start:.1f}s")

    sae = load_arditi_sae_from_dir(Path(args.sae_dir), device=args.device)
    # Feed the SAE post-gain (HF input_layernorm output) activations: multiply
    # TL's pre-gain ln1.hook_normalized by the RMSNorm gain γ = blocks[L].ln1.w.
    gamma = model.blocks[args.layer].ln1.w.detach().float()
    sae._gamma = gamma
    print(f"[gamma] ln1 gain set: ||γ||={gamma.norm().item():.2f}, mean|γ|={gamma.abs().mean().item():.3f}")

    # ── Optional: load Arditi MC forced-choice data (eval under the FRA hooks) ──
    mc_data = None
    mc_results = []
    if args.mc_eval:
        try:
            sys.path.insert(0, args.osemf_root)
            from open_source_em_features.data.mc_questions import (
                create_mc_prompts, _get_letter_token_set,
            )
            mcp, mis_toks, ali_toks = create_mc_prompts(tokenizer)
            a_set = _get_letter_token_set(tokenizer, "A")
            b_set = _get_letter_token_set(tokenizer, "B")
            mc_data = (mcp, mis_toks, ali_toks, a_set, b_set)
            print(f"[mc-eval] loaded {len(mcp)} MC items  (A toks {a_set}, B toks {b_set})")
        except Exception as e:
            print(f"[mc-eval WARN] could not load MC data ({type(e).__name__}: {e}); "
                  f"skipping MC eval", flush=True)
            mc_data = None

    # ── SAE reconstruction sanity check (did exact-top-k salvage the SAE?) ──
    # BatchTopK eval-threshold was miscalibrated (FVU<0 in eval_results); the
    # adapter now forces exact top-k. Verify on real ln1 activations.
    try:
        hn_chk = f"blocks.{args.layer}.{args.hook_point}"
        toks = model.to_tokens(prompts[:2])
        _, cache = model.run_with_cache(toks, names_filter=hn_chk)
        acts = cache[hn_chk].float()           # pre-gain (TL ln1.hook_normalized)
        acts_pg = acts * gamma                 # post-gain (what the SAE expects)
        feats = sae.encode(acts)               # adapter applies γ internally
        recon = sae.decode(feats).float()      # reconstruction is in post-gain space
        denom = (acts_pg - acts_pg.mean(dim=(0, 1), keepdim=True)).pow(2).mean() + 1e-8
        fvu = (acts_pg - recon).pow(2).mean() / denom
        l0 = (feats != 0).float().sum(-1).mean()
        print(f"[sae-check] post-gain exact-top-k: var-explained={1 - fvu.item():.3f}, "
              f"L0={l0.item():.1f}  (want var-expl>~0.5, L0≈64)", flush=True)
        del cache
    except Exception as e:
        print(f"[sae-check WARN] {type(e).__name__}: {e}", flush=True)

    # ── Rank features (FRA, multi-prompt) ─────────────────────────────
    from fra.em_evaluation import rank_features_multi_prompt
    print(f"[rank] ranking features across {len(prompts)} prompts …")
    t_rank = time.time()
    ranked = rank_features_multi_prompt(
        model, sae, args.layer, args.head, args.hook_point,
        prompts=prompts,
        max_length=args.max_length, top_k=args.top_k, k_pairs=args.k_pairs,
        verbose=True,
    )
    qk_features = ranked["qk"]
    ov_features = ranked["ov"]
    print(f"[rank] qk={len(qk_features)}, ov={len(ov_features)} in {time.time()-t_rank:.1f}s")

    # ── Build batched hooks ───────────────────────────────────────────
    from fra.core.helpers import get_W_V

    device = next(model.parameters()).device
    W_dec = sae.W_dec.float()
    W_V_h = get_W_V(model, args.layer, args.head).float()
    n_q_heads = model.cfg.n_heads
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_q_heads
    kv_head_idx = args.head * n_kv_heads // n_q_heads
    hook_name = f"blocks.{args.layer}.{args.hook_point}"
    v_hook_name = f"blocks.{args.layer}.attn.hook_v"
    sae_id = f"L{args.layer}_ln1_arditi_qwen7b_FRA"

    def make_ov_hooks_batched(feature_list: Sequence[int], scale: float):
        """qk_to_ov / ov_to_ov: capture features at ln1, write through W_V at hook_v."""
        feat_indices = list(feature_list)
        feat_indices_t = torch.tensor(feat_indices, device=device, dtype=torch.long)
        feat_v_proj = W_dec[feat_indices] @ W_V_h
        cached = {}

        def capture(activation, hook):
            features = sae.encode(activation)
            cached["feats"] = features.float()
            return activation

        def steer(v, hook):
            features = cached.get("feats")
            if features is None:
                return v
            seq_len = min(features.shape[1], v.shape[1])
            feat_acts = features[:, :seq_len, :].index_select(-1, feat_indices_t).float()
            delta = (scale - 1.0) * feat_acts @ feat_v_proj
            v[:, :seq_len, kv_head_idx, :] += delta.to(v.dtype)
            return v

        return [(hook_name, capture), (v_hook_name, steer)]

    def make_activation_hooks_batched(feature_list: Sequence[int], scale: float):
        """qk_to_qk: rescale features in place at the SAE hookpoint.

        Computed as a *delta* — x ← x + ((scale-1) · Σ_top-K W_dec[f] · features[f]),
        i.e., only the change from scaling the top-K features goes back into
        the residual. The rest of the activation (and the SAE reconstruction
        error of all OTHER features) is left untouched. This is robust to a
        lossy / undertrained SAE: at scale=1 the hook is exactly identity,
        and at scale=0 we subtract exactly the contribution of the top-K
        features. (The earlier `decode(rescale(encode(x)))` form was correct
        only when the SAE's encode→decode round-trip is near-identity; with
        our 100M-token Arditi SAE that round-trip corrupts the activation
        even at scale=1.)
        """
        feat_indices = list(feature_list)
        feat_indices_t = torch.tensor(feat_indices, device=device, dtype=torch.long)
        W_dec_local = sae.W_dec.float()  # (d_sae, d_in)
        W_dec_topk = W_dec_local[feat_indices].contiguous()  # (K, d_in)

        def ablate(activation, hook):
            features = sae.encode(activation).float()            # γ applied inside
            f_topk = features.index_select(-1, feat_indices_t)   # (B, T, K)
            delta = (scale - 1.0) * (f_topk @ W_dec_topk)        # post-gain space
            # activation is pre-gain (ln1.hook_normalized); map the post-gain
            # delta back by dividing by γ so attention sees γ⊙act + scale·W_dec.
            return (activation + (delta / gamma).to(activation.dtype))

        return [(hook_name, ablate)]

    # ── Sweep ──────────────────────────────────────────────────────────
    qualitative = []
    n_total = len(args.alphas) * 3 + 1
    n = 0
    t_gen = time.time()
    for scale in args.alphas:
        cond_specs = [
            (f"qk_to_ov_a{scale}", make_ov_hooks_batched(qk_features, scale)),
            (f"ov_to_ov_a{scale}", make_ov_hooks_batched(ov_features, scale)),
            (f"qk_to_qk_a{scale}", make_activation_hooks_batched(qk_features, scale)),
        ]
        if scale == 1.0:
            cond_specs.append(("baseline", []))
        for cond_name, hooks in cond_specs:
            n += 1
            t_cell = time.time()
            responses = generate_with_hooks_batch(
                model, tokenizer, prompts,
                fwd_hooks=hooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=per_prompt_seeds,
            )
            print(f"  [{n}/{n_total}] {cond_name:<22s} {time.time()-t_cell:.1f}s", flush=True)
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                qualitative.append({
                    "seed": per_prompt_seeds[i],
                    "scale": float(scale),
                    "prompt_idx": i,
                    "prompt": prompt,
                    "condition": cond_name,
                    "response": response,
                    "alignment": 0, "coherence": 0,
                    "sae_id": sae_id,
                    "hook_name": hook_name,
                    "em_model": args.em_model,
                    "eval_seed_base": args.eval_seed,
                })
            # MC forced-choice under the SAME FRA hooks (non-fatal: a bug here
            # must never lose the proven free-form generations above).
            if mc_data is not None:
                try:
                    t_mc = time.time()
                    rows = mc_forced_choice_under_hooks(model, mc_data, hooks)
                    mc_results.append({
                        "scale": float(scale), "condition": cond_name,
                        "em_model": args.em_model, "eval_seed_base": args.eval_seed,
                        "per_item": rows,
                    })
                    print(f"      mc-eval {cond_name}: {len(rows)} items {time.time()-t_mc:.1f}s", flush=True)
                except Exception as e:
                    print(f"      [mc-eval WARN] {cond_name}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
    print(f"[gen] total {time.time()-t_gen:.1f}s")

    out_path = out_root / f"qualitative_FRA_{args.em_model}_evalseed{args.eval_seed}.json"
    out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}  ({len(qualitative)} entries)")

    if mc_data is not None:
        mc_path = out_root / f"mc_FRA_{args.em_model}_evalseed{args.eval_seed}.json"
        mc_path.write_text(json.dumps({
            "em_model": args.em_model, "eval_seed_base": args.eval_seed,
            "layer": args.layer, "head": args.head, "sae_id": sae_id,
            "results": mc_results,
        }, indent=2))
        print(f"[save] {mc_path}  ({len(mc_results)} conditions)")
    print(f"=== TOTAL stream time: {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())

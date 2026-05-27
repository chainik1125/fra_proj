"""Phase 1 conventional-steering GRID orchestrator on Qwen-2.5-7B (+ EM LoRA).

One parametrized orchestrator covering the full FRA-vs-conventional grid:

    ranking      ∈ {wang, fra-qk, fra-ov}
    sae          ∈ {ln1 (ours, BatchTopK @ blocks.L.ln1.hook_normalized),
                    resid_post (Arditi published trainer_1 @ blocks.L.hook_resid_post)}
    granularity  ∈ {1 (each of top-50 individually), 2, 10, 50 (grouped)}

Steering is ALWAYS conventional ADDITIVE on the decoder direction. The only
methodological knobs are (a) which features are picked (ranking), (b) where the
direction is injected (SAE hookpoint), and (c) how many are added at once
(granularity). This keeps the whole grid in one comparable universe.

γ handling (the hard-won bit — mirrors phase1_qkqk_7b_orchestrator):
  - ln1 SAE was trained on the HF input_layernorm OUTPUT = (x/rms)·γ (post-gain),
    but TL's blocks.L.ln1.hook_normalized is x/rms (PRE-gain). To inject
    α·W_dec[f] in *post-gain* space at the pre-gain hook, we add α·W_dec[f]/γ
    (so attention sees α·W_dec[f]). Encoding for ranking feeds act·γ.
  - resid_post SAE lives in the raw residual stream; add α·W_dec[f] at
    blocks.L.hook_resid_post directly, no γ.

For both SAEs we force exact top-k=64 in the encoder (the BatchTopK eval
threshold is miscalibrated). Forcing top-k only matters for the *ranking*
(FRA / Wang read encoder features); the additive steering reads W_dec columns
directly and is independent of the threshold.

Rankings:
  - wang    : Δf_i = mean(f_i | medical answers) − mean(f_i | base answers),
              encoder-side, per-SAE. Computed inline here (so the ln1 γ is
              applied) when --ranking-json is not supplied. For resid_post the
              precomputed wang_ranker_L15_top50.json may be passed via
              --ranking-json to skip recomputation.
  - fra-qk  : rank_features_multi_prompt(...)["qk"]  (ln1 only).
  - fra-ov  : ln1        → rank_features_multi_prompt(...)["ov"];
              resid_post → rank by OV write-contribution ‖W_dec[f] · W_V_h‖
                            (how much the L15 head's OV circuit writes into the
                            feature direction). Computed inline.

Output schema feeds phase1_judge_and_combine.py. Per cell we emit ONE
qualitative_grid_<em>_evalseed<N>.json holding every (feature/group, α) entry;
condition strings are:
    gran=1 (single)  : feat_F{fid}_a{α}
    gran>=2 (grouped): grp{N}_a{α}
The granularity-tag lives in the HF path prefix
(qwen7b/grid/<ranking>_<sae>_<gran>/<model>_seed<seed>/), so a per-granularity
combine downstream is unambiguous.
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
    "medical": "andyrdt/Qwen2.5-7B-Instruct_bad-medical",
    "base":    "Qwen/Qwen2.5-7B-Instruct",
}
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


# ─────────────────────────── model load ──────────────────────────────────
def load_em_model(em_model: str, device: str = "cuda"):
    """Load Qwen-7B + (optionally) bad-medical LoRA, merge, hand to TL.
    Identical to phase1_qkqk_7b_orchestrator.load_em_model."""
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    name = EM_MODELS_7B[em_model]
    print(f"[load] {em_model} → {name}")
    if em_model == "base":
        hf = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        model = HookedTransformer.from_pretrained_no_processing(
            name, hf_model=hf, device=device, dtype=torch.bfloat16,
        )
        del hf
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        lora = PeftModel.from_pretrained(base, name)
        merged = lora.merge_and_unload()
        del base, lora
        model = HookedTransformer.from_pretrained_no_processing(
            BASE_MODEL_ID, hf_model=merged, device=device, dtype=torch.bfloat16,
        )
        del merged
    torch.cuda.empty_cache()
    return model


# ─────────────────────────── SAE adapter ─────────────────────────────────
class _SAEAdapter:
    """Wraps a dictionary_learning BatchTopKSAE for both SAE families.

    Exposes .W_dec (d_sae, d_in), .d_in, .d_sae, .encode, .decode. For the ln1
    SAE set ._gamma after the model loads: encode() then multiplies x by γ so
    it sees post-gain activations (the training distribution). The resid_post
    SAE leaves ._gamma=None.
    """
    def __init__(self, sae, force_topk=64):
        self._sae = sae
        w = sae.decoder.weight.detach()  # (d_in, d_sae)
        self.W_dec = w.T.contiguous()    # (d_sae, d_in)
        self.d_in, self.d_sae = w.shape
        k = force_topk or getattr(sae, "k", None)
        if hasattr(k, "item"):
            k = int(k.item())
        self._k = k
        self._gamma = None

    def encode(self, x):
        if self._gamma is not None:
            x = x * self._gamma
        f = self._sae.encode(x)
        if self._k and f.shape[-1] > self._k:
            topv, topi = f.topk(self._k, dim=-1)
            f = torch.zeros_like(f).scatter_(-1, topi, topv)
        return f

    def decode(self, features):
        return self._sae.decode(features)


def load_sae_from_dir(sae_dir: Path, device: str = "cuda") -> _SAEAdapter:
    from dictionary_learning.utils import load_dictionary
    print(f"[load] SAE from {sae_dir}")
    sae, _ = load_dictionary(str(sae_dir), device=device)
    sae.eval()
    ad = _SAEAdapter(sae)
    print(f"[load] SAE d_in={ad.d_in}, d_sae={ad.d_sae}  W_dec={tuple(ad.W_dec.shape)}  k={ad._k}")
    return ad


# ───────────────────────────── rankings ──────────────────────────────────
@torch.no_grad()
def rank_wang(model, sae, layer, hook_point, prompts, top_n,
              max_new_tokens=200, seed=42, device="cuda"):
    """Encoder-side Wang Δf ranking on the loaded (medical OR base) model.

    NOTE: a true Wang Δf needs BOTH medical and base answer-feature means. Here
    we only have one EM model loaded, so this in-process ranker is used only
    when --ranking-json is NOT supplied. For the campaign we precompute the
    full medical-vs-base Δf separately (scripts/compute_wang_feature_ranking.py,
    extended for ln1) and pass --ranking-json, so both means come from the same
    pod's medical+base passes. This fallback ranks by mean(f | this model's
    answers) which is only a sanity path.
    """
    hook = f"blocks.{layer}.{hook_point}"
    sum_f = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    n_tok = 0
    tok = model.tokenizer
    for i, p in enumerate(prompts):
        toks = model.to_tokens(p)
        torch.manual_seed(seed + i)
        # generate the model's own answer, then forward full seq with cache
        gen = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=1.0, verbose=False)
        _, cache = model.run_with_cache(gen, names_filter=hook)
        acts = cache[hook][0, toks.shape[1]:, :].float()
        if acts.shape[0] == 0:
            continue
        f = sae.encode(acts)
        sum_f += f.sum(0).float()
        n_tok += acts.shape[0]
        del cache
    mean_f = (sum_f / max(n_tok, 1))
    return torch.topk(mean_f, top_n).indices.tolist()


@torch.no_grad()
def rank_fra(model, sae, layer, head, hook_point, prompts, which,
             max_length=128, top_k=20, k_pairs=50):
    """FRA QK/OV ranking via rank_features_multi_prompt (ln1 SAE)."""
    from fra.em_evaluation import rank_features_multi_prompt
    ranked = rank_features_multi_prompt(
        model, sae, layer, head, hook_point, prompts=prompts,
        max_length=max_length, top_k=top_k, k_pairs=k_pairs, verbose=True,
    )
    return ranked[which]


@torch.no_grad()
def rank_ov_writecontrib(model, sae, layer, head, top_n):
    """resid_post FRA-OV proxy (EXPLORATORY): rank resid_post features by how
    much the L15 head's FULL OV output map writes into the feature direction.

    resid_post features live in the post-attention residual OUTPUT space, so the
    relevant operator is the full OV circuit W_OV = W_V_h · W_O_h (reads d_in
    from the residual, writes d_model back to the residual), NOT W_V_h alone
    (which only maps into the head's value subspace — the wrong side for a
    post-W_O feature). We score each feature by how strongly W_OV can write in
    its decoder direction: ‖W_dec[f] · W_OV‖ over the d_model output axis.

    (Flagged exploratory in the results: resid_post features are post-attention,
    so the proper FRA QK decomposition is ill-defined; this OV-write-contribution
    is the analogue CAMPAIGN.md asks for, with the team-lead's W_V·W_O fix.)"""
    from fra.core.helpers import get_W_V, get_W_O
    W_V_h = get_W_V(model, layer, head).float()       # (d_in, d_head)
    W_O_h = get_W_O(model, layer, head).float()       # (d_head, d_model)
    W_OV = W_V_h @ W_O_h                               # (d_in, d_model) full OV map
    # W_dec[f] is a d_model=d_in residual direction; project through W_OV and
    # measure the written magnitude in the d_model output axis.
    score = (sae.W_dec.float() @ W_OV).norm(dim=-1)    # (d_sae,)
    return torch.topk(score, top_n).indices.tolist()


# ───────────────────────── additive steering hooks ───────────────────────
def make_additive_hook(direction: torch.Tensor, alpha: float, gamma=None,
                       delta_a_norm: float | None = None):
    """h ← h + (steering vector) at every position of the hooked layer.

    Two magnitude conventions:
      - delta_a_norm is None (legacy weak):  steer = α · direction  (direction =
        W_dec[f] or Σ_topN W_dec, raw decoder magnitude).
      - delta_a_norm set (MAGNITUDE-MATCHED, primary): steer =
        α_nom · ‖Δa‖ · unit(direction). The direction is L2-normalized to a unit
        vector then scaled by the EM-vs-base mean-activation-difference norm ‖Δa‖
        (the F53258 / diff-cossim convention), so a nominal α reaches the same
        perturbation scale the δ-runs used (±2·‖Δa‖ ≈ the δ≈30-90 regime). Group
        magnitude is therefore constant across granularities (all unit·‖Δa‖).

    For the ln1 SAE pass gamma=γ → we inject the post-gain steer ÷γ at the
    pre-gain ln1.hook_normalized so attention sees the post-gain steer. For
    resid_post pass gamma=None (add directly)."""
    if delta_a_norm is not None:
        unit = direction / (direction.norm() + 1e-8)
        steer = delta_a_norm * unit
    else:
        steer = direction
    inj = steer / gamma if gamma is not None else steer

    def add(activation, hook):
        return activation + alpha * inj.to(device=activation.device, dtype=activation.dtype)

    return add


# ───────────────────── SAE reconstruction sanity check ───────────────────
@torch.no_grad()
def sae_varexpl(model, sae, layer, hook_point, prompts, gamma):
    hn = f"blocks.{layer}.{hook_point}"
    toks = model.to_tokens(prompts[:2])
    _, cache = model.run_with_cache(toks, names_filter=hn)
    acts = cache[hn].float()
    tgt = acts * gamma if gamma is not None else acts
    feats = sae.encode(acts)
    recon = sae.decode(feats).float()
    denom = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).pow(2).mean() + 1e-8
    fvu = (tgt - recon).pow(2).mean() / denom
    l0 = (feats != 0).float().sum(-1).mean()
    del cache
    return float(1 - fvu.item()), float(l0.item())


# ───────────────────────────────── main ──────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking", required=True, choices=["wang", "fra-qk", "fra-ov"])
    p.add_argument("--sae", required=True, choices=["ln1", "resid_post"])
    p.add_argument("--sae-dir", required=True,
                   help="Directory containing ae.pt + config.json for the chosen SAE.")
    p.add_argument("--em-model", required=True, choices=list(EM_MODELS_7B))
    p.add_argument("--eval-seed", type=int, required=True,
                   help="Base seed; per-(prompt,sample) seeds are [seed, seed+1, …].")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--head", type=int, default=0,
                   help="L15 head for FRA ranking (head-ablation argmax on base).")
    p.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 10, 50])
    p.add_argument("--ranking-json", default=None,
                   help="Precomputed ranking JSON with a 'feature_ids' list "
                        "(skips in-process ranking — REQUIRED for wang to get a "
                        "proper medical-vs-base Δf).")
    p.add_argument("--top-n", type=int, default=50,
                   help="Number of ranked features to take (top of the ranking).")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[round(-2 + 0.25 * i, 2) for i in range(17)])  # [-2..2] step .25
    p.add_argument("--delta-a-norm", type=float, default=None,
                   help="MAGNITUDE-MATCHED steering: scale unit(direction) by this "
                        "‖Δa‖ (EM-vs-base mean-act-diff norm at the hookpoint) so "
                        "steer = α_nom·‖Δa‖·unit(dir). Omit for legacy weak α·W_dec.")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--samples-per-prompt", type=int, default=4)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--rank-top-k", type=int, default=20)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True,
                   help="Per-granularity subdirs are created under here.")
    args = p.parse_args()

    # ── validate combination ──
    if args.ranking == "fra-qk" and args.sae == "resid_post":
        raise SystemExit("fra-qk × resid_post is ill-defined (SKIP per CAMPAIGN.md)")

    out_root = Path(args.output_root)
    base_prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    # K independent stochastic draws per prompt; distinct per-(prompt,sample) seed.
    prompts = base_prompts * args.samples_per_prompt
    per_prompt_seeds = [args.eval_seed + i for i in range(len(prompts))]

    hook_point = "ln1.hook_normalized" if args.sae == "ln1" else "hook_resid_post"
    hook_name = f"blocks.{args.layer}.{hook_point}"
    sae_tag = "ln1_arditi_qwen7b" if args.sae == "ln1" else "resid_post_andyrdt_qwen7b_trainer1"
    sae_id = f"L{args.layer}_{sae_tag}_grid_{args.ranking}"

    print("=== Phase 1 GRID orchestrator ===")
    print(f"  ranking × sae   : {args.ranking} × {args.sae}")
    print(f"  em_model        : {args.em_model}   eval_seed={args.eval_seed}")
    print(f"  layer × head    : L{args.layer} H{args.head}  hook={hook_name}")
    print(f"  granularities   : {args.granularities}")
    print(f"  n = {len(base_prompts)} prompts × {args.samples_per_prompt} samples = {len(prompts)}")
    print(f"  alphas ({len(args.alphas)}) : {args.alphas}")
    if args.delta_a_norm is not None:
        print(f"  steer mode      : MAGNITUDE-MATCHED  steer = α_nom·‖Δa‖·unit(dir), "
              f"‖Δa‖={args.delta_a_norm:.4f}  (effective ±{2*args.delta_a_norm:.1f})")
    else:
        print(f"  steer mode      : weak/raw  steer = α·W_dec (no ‖Δa‖ scaling)")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    print(f"[load] model in {time.time()-t_start:.1f}s")

    sae = load_sae_from_dir(Path(args.sae_dir), device=args.device)
    gamma = None
    if args.sae == "ln1":
        gamma = model.blocks[args.layer].ln1.w.detach().float()
        sae._gamma = gamma
        print(f"[gamma] ln1 gain: ||γ||={gamma.norm().item():.2f} mean|γ|={gamma.abs().mean().item():.3f}")

    # SAE reconstruction sanity check (abort-log if clearly broken).
    try:
        ve, l0 = sae_varexpl(model, sae, args.layer, hook_point, base_prompts, gamma)
        print(f"[sae-check] var-explained={ve:.3f}  L0={l0:.1f}  (want ve>~0.4, L0≈64)", flush=True)
        if ve < 0.30:
            print(f"[sae-check WARN] var-expl {ve:.3f} < 0.30 — SAE reconstruction looks broken; "
                  f"steering numbers suspect. Continuing (logged for analyst).", flush=True)
    except Exception as e:
        print(f"[sae-check WARN] {type(e).__name__}: {e}", flush=True)

    # ── rank features ──
    t_rank = time.time()
    if args.ranking_json:
        rj = json.loads(Path(args.ranking_json).read_text())
        feature_ids = list(rj["feature_ids"])[: args.top_n]
        print(f"[rank] loaded {len(feature_ids)} ids from {args.ranking_json} "
              f"(head 5: {feature_ids[:5]})")
    elif args.ranking == "wang":
        print("[rank] WARN: in-process wang ranks this model's answers only "
              "(no medical-vs-base Δf). Prefer --ranking-json.", flush=True)
        feature_ids = rank_wang(model, sae, args.layer, hook_point, base_prompts,
                                args.top_n, max_new_tokens=args.max_new_tokens,
                                seed=args.eval_seed, device=args.device)
    elif args.ranking == "fra-qk":
        feature_ids = rank_fra(model, sae, args.layer, args.head, hook_point,
                               base_prompts, "qk", max_length=args.max_length,
                               top_k=args.rank_top_k, k_pairs=args.k_pairs)[: args.top_n]
    else:  # fra-ov
        if args.sae == "ln1":
            feature_ids = rank_fra(model, sae, args.layer, args.head, hook_point,
                                   base_prompts, "ov", max_length=args.max_length,
                                   top_k=args.rank_top_k, k_pairs=args.k_pairs)[: args.top_n]
        else:
            feature_ids = rank_ov_writecontrib(model, sae, args.layer, args.head, args.top_n)
    print(f"[rank] {len(feature_ids)} features in {time.time()-t_rank:.1f}s "
          f"(head 5: {feature_ids[:5]})")

    W_dec = sae.W_dec.float()  # (d_sae, d_in)

    # ── per-granularity sweeps (one model load amortized over all) ──
    for gran in args.granularities:
        g_out = out_root / f"gran{gran}"
        g_out.mkdir(parents=True, exist_ok=True)
        qualitative = []

        if gran == 1:
            # steer each of the top-N features INDIVIDUALLY
            steer_units = [("feat_F%d" % fid, W_dec[fid].clone()) for fid in feature_ids]
        else:
            # grouped: constant-magnitude α·Σ_topN W_dec[f]
            grp_dir = W_dec[feature_ids[:gran]].sum(dim=0)
            steer_units = [("grp%d" % gran, grp_dir)]

        n_total = len(steer_units) * len(args.alphas)
        n = 0
        t_gen = time.time()
        print(f"\n[gran={gran}] {len(steer_units)} steering unit(s) × {len(args.alphas)} α "
              f"= {n_total} conditions", flush=True)
        for unit_tag, direction in steer_units:
            for alpha in args.alphas:
                n += 1
                t_cell = time.time()
                cond_name = f"{unit_tag}_a{alpha}"
                hooks = [(hook_name, make_additive_hook(direction, alpha, gamma,
                                                        delta_a_norm=args.delta_a_norm))]
                responses = generate_with_hooks_batch(
                    model, tokenizer, prompts, fwd_hooks=hooks,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    seed=per_prompt_seeds,
                )
                if n % 25 == 0 or n == n_total:
                    print(f"  [{n}/{n_total}] {cond_name:<24s} {time.time()-t_cell:.1f}s", flush=True)
                fid = int(unit_tag[6:]) if unit_tag.startswith("feat_F") else None
                for i, (prompt, response) in enumerate(zip(prompts, responses)):
                    qualitative.append({
                        "seed": per_prompt_seeds[i],
                        "scale": float(alpha),
                        "feature_id": fid,
                        "group_n": gran if gran > 1 else None,
                        "prompt_idx": i % args.n_prompts,
                        "sample_idx": i // args.n_prompts,
                        "prompt": prompt,
                        "condition": cond_name,
                        "response": response,
                        "alignment": 0, "coherence": 0,
                        "sae_id": sae_id,
                        "hook_name": hook_name,
                        "ranking": args.ranking,
                        "sae_family": args.sae,
                        "granularity": gran,
                        "delta_a_norm": args.delta_a_norm,
                        "steer_mode": "magmatched" if args.delta_a_norm is not None else "weak_raw",
                        "em_model": args.em_model,
                        "eval_seed_base": args.eval_seed,
                    })
                torch.cuda.empty_cache()

        out_path = g_out / f"qualitative_grid_{args.em_model}_evalseed{args.eval_seed}.json"
        out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
        print(f"[gran={gran}] [save] {out_path}  ({len(qualitative)} entries, "
              f"{time.time()-t_gen:.1f}s)", flush=True)

    # save the ranking used (so the analyst can map feature ids → scores)
    (out_root / f"ranking_{args.ranking}_{args.sae}.json").write_text(
        json.dumps({"ranking": args.ranking, "sae": args.sae, "layer": args.layer,
                    "head": args.head, "feature_ids": feature_ids,
                    "delta_a_norm": args.delta_a_norm,
                    "steer_mode": "magmatched" if args.delta_a_norm is not None else "weak_raw"},
                   indent=2))
    print(f"=== TOTAL grid time: {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())

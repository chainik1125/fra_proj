"""Conv and DoM channel handlers for the run_experiment pipeline.

These two baselines don't fit the ln1→W_{Q,K,V} FRA channel abstraction, so they
are evaluated here instead of in select_features/eval, but they emit the *same*
uniform results schema as scripts.eval.eval_tuples_json:

  {channel, regime, mode, config, baseline:{asr_per_seed, asr},
   results: [{seed, tuple, alpha_sweep:{"<α>": ev, ...}}]}

so every plot script reads one schema regardless of channel.

  conv — a conventional residual-stream SAE feature (sae_resid_mid), selected by
         dep−clean activation diff → cosine-to-v_md re-rank → greedy ASR screen,
         steered additively at blocks.0.hook_resid_mid (eval_downstream_baseline).
         One winning feature per SAE seed; `tuple` is [[f, "M"]].
  dom  — SAE-free attention-weighted difference-of-means direction, geometric
         ablation at blocks.0.hook_resid_mid (eval_dom). One direction, no SAE
         seed population; emitted as a single result row (seed 0, empty tuple).
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from sleeper.eval import (
    _build_baselines_per_seed, attn_weighted_vmd, cosine_rerank_top, eval_dom,
    eval_downstream_baseline, split_dep_prompts,
)
from sleeper.hooks import (
    additive_steer_hook, compute_sae_delta, generate_with_hooks, make_greedy_sampler,
    make_sampling_sampler,
)
from sleeper.metrics import asr_16, rank_features_by_dep_clean
from sleeper.model import (
    MODELS, ModelName, left_pad_prompts, load_paired_dataset, load_sleeper_model,
)
from sleeper.sae import encode_all, load as sae_load

RESID_HOOK = "blocks.0.hook_resid_mid"


# ---------------------------------------------------------------------------
# Shared eval setup (held-out dep prompts + per-seed unsteered baselines)
# ---------------------------------------------------------------------------

def _eval_setup(model, tok, *, n_sel, n_eval, gen_tokens, eval_seeds,
                eval_temperature, device, model_name, eval_set="disjoint"):
    """Build the held-out eval split + per-decode-seed unsteered baselines and
    the unsteered baseline ASR. Shared by conv and dom."""
    pad_id = tok.pad_token_id or tok.eos_token_id
    eval_raw = split_dep_prompts(tok, n_sel, n_eval, model=model_name, eval_set=eval_set)["eval"]
    eval_dep_lp, eval_dep_attn = left_pad_prompts(eval_raw, pad_id)
    eval_dep_lp   = eval_dep_lp.to(device)
    eval_dep_attn = eval_dep_attn.to(device)

    clean_lsm_ps, clean_tok_ps, dep_lsm_ps = _build_baselines_per_seed(
        model, eval_dep_lp, eval_dep_attn, gen_tokens, device,
        seeds=eval_seeds, temperature=eval_temperature,
    )
    # Unsteered baseline ASR per decode seed (sampled dep rollouts, no hooks).
    base_asr_per_seed: list[float] = []
    for s in eval_seeds:
        sampler = make_sampling_sampler(temperature=eval_temperature, seed=int(s), device=device)
        gen = generate_with_hooks(model, eval_dep_lp, [], gen_tokens, sampler,
                                  attention_mask=eval_dep_attn)
        base_asr_per_seed.append(asr_16(gen.cpu(), tok))
    base_asr = sum(base_asr_per_seed) / len(base_asr_per_seed)
    return (eval_dep_lp, eval_dep_attn, clean_lsm_ps, clean_tok_ps, dep_lsm_ps,
            base_asr_per_seed, base_asr)


def _result_dict(channel, mode, *, sae_dir, eval_alphas, n_sel, n_eval, gen_tokens,
                 eval_seeds, eval_temperature, base_asr_per_seed, base_asr, results):
    return {
        "channel": channel,
        "regime":  "diff",
        "mode":    mode,
        "config":  {
            "sae_dir": str(sae_dir) if sae_dir is not None else None,
            "eval_alphas": list(eval_alphas), "n_sel": n_sel, "n_eval": n_eval,
            "gen_tokens": gen_tokens, "eval_seeds": list(eval_seeds),
            "eval_temperature": eval_temperature,
        },
        "baseline": {"asr_per_seed": base_asr_per_seed, "asr": base_asr},
        "results":  results,
    }


# ---------------------------------------------------------------------------
# conv — conventional resid_mid SAE feature
# ---------------------------------------------------------------------------

@torch.no_grad()
def _conv_select(model, tok, sae_mid, resid_hook, *, mode, top_k, identify_top_k,
                 screen_alphas, n_sel, n_eval, gen_tokens, device, model_name):
    """Return the conv feature list for this seed.

    winner → [single feature] via dep−clean diff → cosine-to-v_md → greedy ASR.
    topk   → top-`top_k` features by the dep−clean activation diff (step 1 only).
    """
    pad_id = tok.pad_token_id or tok.eos_token_id
    seq_len = MODELS[model_name].seq_len
    sel = load_paired_dataset(tok, n_train=0, n_val=n_sel, n_test=n_eval,
                              seq_len=seq_len, seed=0, model=model_name)["val"]
    combined_lp   = sel.tokens.to(device)
    combined_attn = sel.attention_mask.to(device)
    is_dep        = sel.is_deployment.to(device)

    # 1. dep−clean activation-diff ranking → candidates.
    _, cache = model.run_with_cache(
        combined_lp, attention_mask=combined_attn, return_type=None,
        names_filter=lambda n: n == resid_hook,
    )
    z = encode_all(sae_mid, cache[resid_hook]).to(device)
    ranked = rank_features_by_dep_clean(z, is_dep, combined_attn, top_k=identify_top_k)
    candidates = ranked["top_indices"].cpu().tolist()
    if mode == "topk":
        return candidates[:top_k]

    # 2. cosine re-rank candidates by the attention-weighted dep−clean v_md and keep
    #    the 3 highest — the SAME shared selection OV uses (paper § app:sleeper_method).
    vmd = attn_weighted_vmd(model, sel.tokens, sel.attention_mask, sel.is_deployment,
                            resid_hook, device)
    survivors = cosine_rerank_top(candidates, sae_mid.W_dec, vmd, keep=3)

    # 3. greedy ASR screen on selection-split dep prompts → winner.
    id_raw = split_dep_prompts(tok, n_sel, n_eval, model=model_name)["sel"]
    id_lp, id_attn = left_pad_prompts(id_raw, pad_id)
    id_lp, id_attn = id_lp.to(device), id_attn.to(device)
    sampler = make_greedy_sampler()
    asr_table: dict[int, float] = {}
    for f in survivors:
        best = 1.0
        for a in screen_alphas:
            delta = compute_sae_delta(model, sae_mid, resid_hook, int(f), id_lp,
                                      id_attn.bool(), attention_mask=id_attn)
            gen = generate_with_hooks(model, id_lp, additive_steer_hook(delta, a, resid_hook),
                                      gen_tokens, sampler, attention_mask=id_attn)
            best = min(best, asr_16(gen, tok))
        asr_table[int(f)] = best
    # lowest-ASR among the 3, tie-broken by cosine order (survivors is cosine-desc).
    winner = min(survivors, key=lambda f: (asr_table[f], survivors.index(f)))
    return [winner]


@torch.no_grad()
def conv_channel(*, sae_dir: Path, sae_seeds, mode="winner", top_k=20,
                 identify_top_k=20, screen_alphas=(2.0, 4.0), eval_alphas,
                 n_sel=200, n_eval=400, gen_tokens=16, eval_seeds=(0, 1, 2, 3, 4),
                 eval_temperature=1.0, device=None, model: ModelName = "tinystories",
                 eval_set="disjoint") -> dict:   # EXPERIMENTAL knob; "disjoint" = production
    """Conventional resid_mid SAE-feature baseline, uniform results schema."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hooked = load_sleeper_model(model=model, device=device)
    tok = hooked.tokenizer
    (eval_dep_lp, eval_dep_attn, clean_lsm_ps, clean_tok_ps, dep_lsm_ps,
     base_asr_per_seed, base_asr) = _eval_setup(
        hooked, tok, n_sel=n_sel, n_eval=n_eval, gen_tokens=gen_tokens,
        eval_seeds=eval_seeds, eval_temperature=eval_temperature, device=device,
        model_name=model, eval_set=eval_set)
    print(f"[conv] baseline asr={base_asr:.3f}", flush=True)

    results: list[dict] = []
    for seed in sae_seeds:
        sae_mid, cfg = sae_load(sae_dir / f"sae_resid_mid_s{seed}.pt", device=device)
        resid_hook = cfg.get("layer_hook", RESID_HOOK)
        feats = _conv_select(hooked, tok, sae_mid, resid_hook, mode=mode, top_k=top_k,
                             identify_top_k=identify_top_k, screen_alphas=screen_alphas,
                             n_sel=n_sel, n_eval=n_eval, gen_tokens=gen_tokens,
                             device=device, model_name=model)
        print(f"[conv] seed={seed}  features={feats}", flush=True)
        for f in feats:
            alpha_sweep: dict[str, dict] = {}
            for a in eval_alphas:
                t0 = time.time()
                ev = eval_downstream_baseline(
                    hooked, sae_mid, int(f), a, eval_dep_lp, eval_dep_attn,
                    clean_lsm_ps, clean_tok_ps, dep_lsm_ps, gen_tokens, device,
                    eval_seeds=eval_seeds, eval_temperature=eval_temperature)
                alpha_sweep[str(a)] = ev
                print(f"[conv]   s={seed} f={f} α={a:>4.1f}  asr={ev['asr']:.3f}  "
                      f"jsd_cln={ev['jsd_clean']:.3f}  ({time.time()-t0:.1f}s)", flush=True)
            results.append({"seed": seed, "tuple": [[int(f), "M"]], "alpha_sweep": alpha_sweep})

    return _result_dict("conv", mode, sae_dir=sae_dir, eval_alphas=eval_alphas,
                        n_sel=n_sel, n_eval=n_eval, gen_tokens=gen_tokens,
                        eval_seeds=eval_seeds, eval_temperature=eval_temperature,
                        base_asr_per_seed=base_asr_per_seed, base_asr=base_asr, results=results)


# ---------------------------------------------------------------------------
# dom — SAE-free attention-weighted difference-of-means
# ---------------------------------------------------------------------------

@torch.no_grad()
def dom_channel(*, eval_alphas, n_sel=200, n_eval=400, gen_tokens=16,
                eval_seeds=(0, 1, 2, 3, 4), eval_temperature=1.0, device=None,
                resid_hook: str = RESID_HOOK, model: ModelName = "tinystories",
                eval_set="disjoint") -> dict:   # EXPERIMENTAL knob; "disjoint" = production
    """SAE-free difference-of-means baseline, uniform results schema (single row)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hooked = load_sleeper_model(model=model, device=device)
    tok = hooked.tokenizer
    (eval_dep_lp, eval_dep_attn, clean_lsm_ps, clean_tok_ps, dep_lsm_ps,
     base_asr_per_seed, base_asr) = _eval_setup(
        hooked, tok, n_sel=n_sel, n_eval=n_eval, gen_tokens=gen_tokens,
        eval_seeds=eval_seeds, eval_temperature=eval_temperature, device=device,
        model_name=model, eval_set=eval_set)
    print(f"[dom] baseline asr={base_asr:.3f}", flush=True)

    # v_md from the selection split (disjoint from the eval split).
    seq_len = MODELS[model].seq_len
    sel = load_paired_dataset(tok, n_train=2, n_val=n_sel, n_test=2,
                              seq_len=seq_len, seed=0, model=model)["val"]
    vmd = attn_weighted_vmd(hooked, sel.tokens, sel.attention_mask,
                            sel.is_deployment, resid_hook, device)

    alpha_sweep: dict[str, dict] = {}
    for a in eval_alphas:
        t0 = time.time()
        ev = eval_dom(hooked, vmd, a, eval_dep_lp, eval_dep_attn,
                      clean_lsm_ps, clean_tok_ps, dep_lsm_ps, gen_tokens, device,
                      eval_seeds=eval_seeds, eval_temperature=eval_temperature,
                      resid_hook=resid_hook)
        alpha_sweep[str(a)] = ev
        print(f"[dom]   α={a:>4.2f}  asr={ev['asr']:.3f}  jsd_cln={ev['jsd_clean']:.3f}  "
              f"jsd_dep={ev['jsd_pois']:.3f}  exact={ev['exact_match']:.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    results = [{"seed": 0, "tuple": [], "alpha_sweep": alpha_sweep}]

    return _result_dict("dom", "winner", sae_dir=None, eval_alphas=eval_alphas,
                        n_sel=n_sel, n_eval=n_eval, gen_tokens=gen_tokens,
                        eval_seeds=eval_seeds, eval_temperature=eval_temperature,
                        base_asr_per_seed=base_asr_per_seed, base_asr=base_asr, results=results)

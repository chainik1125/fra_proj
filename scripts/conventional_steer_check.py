"""Diagnostic: CONVENTIONAL (additive) SAE steering at any hookpoint, for a
range of dict sizes. No Fisher, no OV — pure activation-diff selection + an
additive residual-stream patch at the SAE's own hookpoint, exactly the fisher
1536 'conventional/downstream' code path, generalized to also run at ln1.

For each (d_sae, hookpoint): train a TopK SAE on the sleeper model, select
top-K by dep-clean activation diff, screen (Δlogp cull -> greedy-ASR winner),
then sweep alpha with an additive patch and report J_clean / J_pois / ASR.

  python -u -m scripts.conventional_steer_check --d_saes 1536 3072
"""
from __future__ import annotations

import argparse
import torch

from sleeper.hooks import (additive_steer_hook, compute_sae_delta,
                           generate_with_hooks, make_greedy_sampler, make_sampling_sampler)
from sleeper.metrics import asr_16, rank_features_by_dep_clean, teacher_forced_sleeper_logp
from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import encode_all, train
from sleeper.jsd_cells import build_eval_refs, jsd_mean
from sleeper.screen import build_sel_caches

HOOKS = {"ln1": "blocks.0.ln1.hook_normalized", "resid_mid": "blocks.0.hook_resid_mid"}
GEN = 16
SCREEN_ALPHAS = (2.0, 4.0)


@torch.no_grad()
def conventional_eval(model, sae, hook, c, refs, alphas, device, top_k=20):
    tok = model.tokenizer
    acts = c.ln1_acts if hook == HOOKS["ln1"] else c.resid_mid_acts
    z = encode_all(sae, acts).to(device)
    topK = rank_features_by_dep_clean(z, c.is_dep, c.sel_pmask, top_k=top_k)["top_indices"].cpu().tolist()
    attr_rank = {int(f): i for i, f in enumerate(topK)}

    # stage-0 Δlogp cull → top K/2
    dlogp = {}
    for f in topK:
        delta = compute_sae_delta(model, sae, hook, int(f), c.sel_dep, c.sel_dep_pmask)
        dlogp[int(f)] = min(
            teacher_forced_sleeper_logp(model, tok, c.sel_dep,
                                        fwd_hooks=additive_steer_hook(delta, a, hook)).mean().item()
            - c.sel_base_logp for a in SCREEN_ALPHAS)
    survivors = sorted(dlogp, key=dlogp.get)[:max(1, top_k // 2)]

    # stage-1 greedy-ASR winner
    greedy = make_greedy_sampler()
    asr_tab = {}
    for f in survivors:
        delta = compute_sae_delta(model, sae, hook, int(f), c.sel_lp, c.sel_attn.bool(),
                                  attention_mask=c.sel_attn)
        best = 1.0
        for a in SCREEN_ALPHAS:
            g = generate_with_hooks(model, c.sel_lp, additive_steer_hook(delta, a, hook),
                                    GEN, greedy, attention_mask=c.sel_attn)
            best = min(best, asr_16(g, tok))
        asr_tab[int(f)] = best
    winner = min(asr_tab, key=lambda f: (asr_tab[f], attr_rank[f]))

    # alpha-sweep eval (additive patch at hook), sampled like the pipeline
    curve = {}
    for a in alphas:
        if a == 0.0:
            curve[a] = (jsd_mean(refs.poisoned_lsm, refs.clean_lsm), 0.0,
                        asr_16(refs.poisoned_tokens.cpu(), tok))
            continue
        delta = compute_sae_delta(model, sae, hook, int(winner), refs.dep_lp, refs.dep_attn,
                                  attention_mask=refs.dep_attn)
        sampler = make_sampling_sampler(temperature=1.0, seed=0, device=device)
        st, lsm = generate_with_hooks(model, refs.dep_lp, additive_steer_hook(delta, a, hook),
                                      GEN, sampler, attention_mask=refs.dep_attn, capture_log_softmax=True)
        curve[a] = (jsd_mean(lsm, refs.clean_lsm), jsd_mean(lsm, refs.poisoned_lsm),
                    asr_16(st.cpu(), tok))
    return winner, asr_tab[winner], curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_saes", type=int, nargs="+", default=[1536, 3072])
    p.add_argument("--hooks", nargs="+", default=["ln1", "resid_mid"], choices=list(HOOKS))
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=50000)
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_sleeper_model(device=dev)
    splits = load_paired_dataset(model.tokenizer, n_train=10000, n_val=0, n_test=0, seq_len=128, seed=0)
    train_acts = {h: cache_activations(model, splits["train"].tokens, [HOOKS[h]])[HOOKS[h]]
                  for h in args.hooks}
    c = build_sel_caches(model, dev)
    refs = build_eval_refs(model, dev)

    print("\n==== CONVENTIONAL (additive) steering — winner + alpha-sweep ====", flush=True)
    for d_sae in args.d_saes:
        for h in args.hooks:
            sae, _ = train(train_acts[h], d_sae=d_sae, k=args.k, n_steps=args.n_steps, seed=0, device=dev)
            win, smin, curve = conventional_eval(model, sae, HOOKS[h], c, refs, args.alphas, dev, args.top_k)
            print(f"\n[conv] d_sae={d_sae} hook={h} k={args.k}  winner={win} screen_minASR={smin:.3f}", flush=True)
            print("   alpha  J_clean  J_pois   ASR", flush=True)
            for a in args.alphas:
                jc, jp, asr = curve[a]
                print("  %+5.1f  %7.3f  %6.3f  %5.3f" % (a, jc, jp, asr), flush=True)


if __name__ == "__main__":
    main()

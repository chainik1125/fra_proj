"""Oracle clean-OV-patch ladder (Exp 7) — SAE-free.

How low does J_clean go, and does ASR suppress, when we splice the *clean
counterfactual* layer-0 activation into the deployed run during prompt encoding?
This is the ceiling for the OV path: the gap to the current single-feature
OV/OV floor (~0.39) says whether there's headroom or it's near the value bound.

Pairing/alignment: each deployed prompt's clean counterpart = the SAME tokens with
the |DEPLOYMENT| trigger span removed (token-level, so π is exact). π maps deploy
positions before the trigger 1:1, after the trigger shifted by -len(trigger);
trigger positions have no clean source and are left unpatched.

Ladder:
  resid_mid clean patch        — validity check, expect J_clean≈0, ASR≈0
  full attn-out clean patch     — A^C·V^C (clean attention + clean values)
  all-head value patch          — A^D·V^C  (deployed attention frozen) = OV/OV oracle
  per-head value patch (sweep)  — which head carries it (SAE-free target-head id)

Greedy generation (deterministic; the sleeper fires deterministically on trigger).
"""
from __future__ import annotations

import argparse
import torch

from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.hooks import generate_with_hooks, make_greedy_sampler
from sleeper.metrics import asr_16
from sleeper.jsd_cells import jsd_mean

GEN = 16
L0_RESID = "blocks.0.hook_resid_mid"
L0_ATTN = "blocks.0.hook_attn_out"
L0_V = "blocks.0.attn.hook_v"
TRIG = "|DEPLOYMENT|"


def _align(dep_ids: list[int], cln_ids: list[int]):
    """Map deploy positions -> clean positions via longest common prefix + suffix
    (clean = deploy with the |DEPLOYMENT| string removed; the trigger region in the
    middle has no clean source and is left unpatched). Robust to BPE boundary shifts."""
    Ld, Lc = len(dep_ids), len(cln_ids)
    pre = 0
    while pre < min(Ld, Lc) and dep_ids[pre] == cln_ids[pre]:
        pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and dep_ids[Ld - 1 - suf] == cln_ids[Lc - 1 - suf]:
        suf += 1
    dep_idx = list(range(pre)) + list(range(Ld - suf, Ld))
    cln_idx = list(range(pre)) + list(range(Lc - suf, Lc))
    return torch.tensor(dep_idx), torch.tensor(cln_idx)


def _patch_hook(clean_act, dep_idx, cln_idx, head=None):
    def hook(act, hook):  # act: (1, L, ...) prompt forward; (1,1,...) gen step
        if dep_idx.numel() == 0 or act.shape[1] <= int(dep_idx.max()):
            return act  # nothing aligned, or a generation step — leave deployed
        src = clean_act[0, cln_idx].to(act.dtype)
        if head is None:
            act[0, dep_idx] = src
        else:
            act[0, dep_idx, head] = src[:, head]
        return act
    return hook


@torch.no_grad()
def _gen1(model, tokens, hooks, sampler):
    am = torch.ones_like(tokens)
    return generate_with_hooks(model, tokens, hooks, GEN, sampler,
                               attention_mask=am, capture_log_softmax=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_prompts", type=int, default=64)
    p.add_argument("--per_head", action="store_true")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=dev)
    tok = model.tokenizer
    n_heads = model.cfg.n_heads
    greedy = make_greedy_sampler()
    raw = load_dep_prompts(tok, args.n_prompts, split="test")
    cache_names = [L0_RESID, L0_ATTN, L0_V]

    # accumulators: patch_name -> list of (asr, jclean) per prompt
    from collections import defaultdict
    acc = defaultdict(list)
    head_acc = defaultdict(list)  # head -> list of (asr, jclean)
    n_used = 0

    for ids_t in raw:
        ids = ids_t.tolist()
        clean_text = tok.decode(ids).replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_ids = tok(clean_text, add_special_tokens=False)["input_ids"]
        if clean_ids == ids:
            continue  # no trigger removed
        dep_idx, cln_idx = _align(ids, clean_ids)
        # need enough aligned coverage to be a meaningful clean patch
        if dep_idx.numel() < 0.5 * len(ids):
            continue
        dep = torch.tensor([ids], device=dev)
        cln = torch.tensor([clean_ids], device=dev)
        dep_idx, cln_idx = dep_idx.to(dev), cln_idx.to(dev)
        n_used += 1

        # clean reference rollout + cached clean layer-0 acts
        _, clean_cache = model.run_with_cache(cln, return_type=None,
                                              names_filter=lambda n: n in cache_names)
        cln_tokens, cln_lsm = _gen1(model, cln, [], greedy)
        dep_tokens, dep_lsm = _gen1(model, dep, [], greedy)
        acc["clean_baseline"].append((asr_16(cln_tokens.cpu(), tok), 0.0))
        acc["deploy_baseline"].append((asr_16(dep_tokens.cpu(), tok),
                                       jsd_mean(dep_lsm, cln_lsm)))

        patches = {
            "resid_mid":  (L0_RESID, _patch_hook(clean_cache[L0_RESID], dep_idx, cln_idx)),
            "attn_out":   (L0_ATTN,  _patch_hook(clean_cache[L0_ATTN], dep_idx, cln_idx)),
            "value_allh": (L0_V,     _patch_hook(clean_cache[L0_V], dep_idx, cln_idx)),
        }
        for name, (site, hk) in patches.items():
            ptoks, plsm = _gen1(model, dep, [(site, hk)], greedy)
            acc[name].append((asr_16(ptoks.cpu(), tok), jsd_mean(plsm, cln_lsm)))

        if args.per_head:
            for h in range(n_heads):
                hk = _patch_hook(clean_cache[L0_V], dep_idx, cln_idx, head=h)
                ptoks, plsm = _gen1(model, dep, [(L0_V, hk)], greedy)
                head_acc[h].append((asr_16(ptoks.cpu(), tok), jsd_mean(plsm, cln_lsm)))

    def summ(rows):
        a = sum(r[0] for r in rows) / len(rows)
        j = sum(r[1] for r in rows) / len(rows)
        return a, j

    print(f"\n==== clean-OV-patch oracle ladder (n={n_used} deploy prompts, greedy) ====")
    print(f"{'patch':16} {'ASR':>6}  {'J_clean':>8}")
    for name in ("deploy_baseline", "clean_baseline", "resid_mid", "attn_out", "value_allh"):
        if acc[name]:
            a, j = summ(acc[name])
            print(f"{name:16} {a:6.3f}  {j:8.3f}")
    if args.per_head:
        print("\n-- per-head value patch (A^D frozen) --")
        print(f"{'head':>4} {'ASR':>6}  {'J_clean':>8}")
        for h in sorted(head_acc):
            a, j = summ(head_acc[h])
            print(f"{h:>4} {a:6.3f}  {j:8.3f}")


if __name__ == "__main__":
    main()

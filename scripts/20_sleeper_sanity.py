"""SANITY GATE before any transfer work: is the fresh seed-0 SAE's top QK pair real?

A freshly trained SAE need not surface the features the paper found. If the pair
that `rank_qk_diff` identifies is junk, every downstream intervention number is
junk too -- so this runs first and reports what the pair actually fires on.

Uses the paper's OWN selection code (`sleeper.attribution.rank_qk_diff`) so that
the later comparison holds feature selection fixed and varies only the
intervention.

Run: python scripts/20_sleeper_sanity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sleeper.eval import LN1_HOOK
from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import encode_all, load as sae_load
from sleeper.attribution import rank_qk_diff

SAE_PATH = Path("weights/seeds/sae_ln1_s0.pt")
N_PROMPTS = 200
TOP_N = 8
OUT = Path("results/sleeper_sanity.json")


def top_tokens_for_feature(z, tokens, tokenizer, feat, n=12):
    """Which tokens does this feature fire hardest on?"""
    acts = z[:, :, feat]                       # (B, T)
    flat = acts.flatten()
    k = min(n * 6, flat.numel())
    vals, idx = torch.topk(flat, k)
    T = tokens.shape[1]
    seen, out = set(), []
    for v, i in zip(vals.tolist(), idx.tolist()):
        if v <= 0:
            break
        tok = int(tokens[i // T, i % T])
        s = tokenizer.decode([tok])
        if s not in seen:
            seen.add(s)
            out.append((s, round(v, 3)))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    print("loading model + SAE ...")
    model = load_sleeper_model("tinystories", device=device)
    sae_ln1, meta = sae_load(SAE_PATH, device=device)
    print(f"  SAE: d_sae={sae_ln1.d_sae} k={sae_ln1.k} hook={meta.get('layer_hook')}")

    split = load_paired_dataset(
        model.tokenizer, n_train=0, n_val=0, n_test=N_PROMPTS, seq_len=128, seed=0
    )["test"]
    tokens = split.tokens
    is_dep = split.is_deployment
    print(f"  prompts: {tokens.shape}  dep={int(is_dep.sum())} clean={int((~is_dep).sum())}")

    acts = cache_activations(model, tokens, [LN1_HOOK])
    z = encode_all(sae_ln1, acts[LN1_HOOK])
    print(f"  z_ln1: {tuple(z.shape)}  mean L0={(z > 0).float().sum(-1).mean():.1f}")

    pmask = split.attention_mask
    # PAPER-EXACT: scripts/select_features.py::_ensure_qk_diff passes query_mask
    # only, leaving key_mask=None (Z_k sums over ALL positions incl. padding).
    out = rank_qk_diff(z, sae_ln1, model.W_Q[0], model.W_K[0], is_dep,
                       query_mask=pmask)

    # Where does the trigger live? Its last token is the closing '|'.
    trig = model.tokenizer("DEPLOYMENT|", add_special_tokens=False)["input_ids"]
    trig_last = trig[-1]
    is_trig = tokens == trig_last                          # (B, T)
    dep_rows = is_dep

    print("\n" + "=" * 88)
    print(f"TOP {TOP_N} QK FEATURE PAIRS  (paper's rank_qk_diff, fresh seed-0 SAE)")
    print("=" * 88)
    rows = []
    for r in range(TOP_N):
        lq = int(out["top_pairs_q"][r])
        lk = int(out["top_pairs_k"][r])
        sc = float(out["score"][lq, lk])
        qk = float(out["QK_total"][lq, lk])

        rec = {"rank": r, "lambda_q": lq, "lambda_k": lk, "score": sc, "QK_total": qk}
        for tag, feat in (("q", lq), ("k", lk)):
            a = z[:, :, feat]
            fires = a > 0
            # Selectivity: does it fire on deployment prompts, and on the trigger token?
            dep_rate = fires[dep_rows].float().mean().item()
            cln_rate = fires[~dep_rows].float().mean().item()
            on_trig = a[is_trig].mean().item() if is_trig.any() else 0.0
            off_trig = a[~is_trig].mean().item()
            rec[f"{tag}_dep_rate"] = dep_rate
            rec[f"{tag}_cln_rate"] = cln_rate
            rec[f"{tag}_mean_on_trigger"] = on_trig
            rec[f"{tag}_mean_off_trigger"] = off_trig
            rec[f"{tag}_top_tokens"] = top_tokens_for_feature(z, tokens, model.tokenizer, feat)
        rows.append(rec)

        print(f"\nrank {r}:  (lambda_q={lq}, lambda_k={lk})  score={sc:.4g}  QK_total={qk:+.4g}")
        for tag, feat in (("Q-side", lq), ("K-side", lk)):
            t = "q" if tag.startswith("Q") else "k"
            print(f"  {tag} feat {feat:5d}  fires: dep {rec[f'{t}_dep_rate']*100:5.1f}% "
                  f"clean {rec[f'{t}_cln_rate']*100:5.1f}%   "
                  f"mean act on '|' trigger tok {rec[f'{t}_mean_on_trigger']:.3f} "
                  f"vs elsewhere {rec[f'{t}_mean_off_trigger']:.3f}")
            toks = ", ".join(f"{s!r}:{v}" for s, v in rec[f"{t}_top_tokens"][:8])
            print(f"          top tokens: {toks}")

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {OUT}")

    # What the paper's pipeline would ACTUALLY intervene on: it decomposes the
    # pair ranking into two independent marginal orderings and re-zips them, so
    # the i-th "pair" need never have been co-scored.
    from scripts.select_features import _top_unique_from_pairs
    qf, kf = _top_unique_from_pairs(out["top_pairs_q"].tolist(),
                                    out["top_pairs_k"].tolist(), 5)
    print("\n" + "=" * 88)
    print("WHAT THE PAPER'S PIPELINE INTERVENES ON (_top_unique_from_pairs, then zip)")
    print("=" * 88)
    print(f"  top unique Q feats: {qf}")
    print(f"  top unique K feats: {kf}")
    for i, (a, b) in enumerate(zip(qf, kf)):
        co = float(out["score"][a, b])
        rank_of_pair = int((out["score"].flatten() > co).sum())
        print(f"    tuple {i}: (Q={a}, K={b})  pair score {co:.4g}  "
              f"-> that pair ranks #{rank_of_pair} of {out['score'].numel()}")

    # The trigger feature, wherever it landed.
    print("\n  TRIGGER-FEATURE SCAN (deployment-selective + '|'-localised):")
    trig_mean = z[is_trig].mean(0)
    else_mean = z[~is_trig].mean(0)
    dep_fire = (z[dep_rows] > 0).float().mean((0, 1))
    cln_fire = (z[~dep_rows] > 0).float().mean((0, 1))
    ratio = trig_mean / else_mean.clamp(min=1e-6)
    for f in torch.topk(ratio, 5).indices.tolist():
        print(f"    feat {f:5d}: on-trigger {trig_mean[f]:7.3f} elsewhere {else_mean[f]:.3f} "
              f"(x{ratio[f]:7.1f})   fires dep {dep_fire[f]*100:5.2f}% clean {cln_fire[f]*100:5.2f}%")

    best = rows[0]
    trigger_side = (best["q_mean_on_trigger"] > 5 * max(best["q_mean_off_trigger"], 1e-6)
                    or best["k_mean_on_trigger"] > 5 * max(best["k_mean_off_trigger"], 1e-6))
    dep_selective = (best["q_dep_rate"] > 2 * max(best["q_cln_rate"], 1e-6)
                     or best["k_dep_rate"] > 2 * max(best["k_cln_rate"], 1e-6))
    print("\n  GATE: top pair has a trigger-localised side: ", trigger_side)
    print("        top pair has a deployment-selective side:", dep_selective)
    print("        -> ", "PROCEED" if (trigger_side or dep_selective) else "DO NOT PROCEED (pair looks like junk)")


if __name__ == "__main__":
    main()

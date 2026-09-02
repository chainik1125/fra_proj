"""Re-run THEIR ranking with a causally correct query mask.

`rank_qk_diff` aggregates `Z_q(lambda) = sum over ALL query positions`. For a
sleeper agent only ONE query matters -- the decision position, where the payload
is generated. A feature firing 40 tokens earlier counts exactly as much as one
firing at the decision point, so the ranking is position-agnostic in a task where
position IS the causal story.

Everything here is their function, their score, their call signature -- the ONLY
change is `query_mask`, restricted to the final real token of each row instead of
every prompt position. Prompts are left-padded, so that is column T-1 for every
row (`story_marker_pos` by construction).

The question: does the trigger feature move to the KEY side? If it does, the
ranking can surface the causally relevant pair once the mask is right, and the
three arms get a fair test of delivery. If it does not, the ranking cannot
surface it at all -- which is itself the finding.

Run: python scripts/22_decision_position_rank.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sleeper.attribution import rank_qk_diff
from sleeper.eval import LN1_HOOK
from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import encode_all, load as sae_load

SAE_PATH = Path("weights/seeds/sae_ln1_s0.pt")
N_PROMPTS = 200
TOP_N = 6
TRIGGER_FEATS = [1114, 616, 1079, 832, 1199, 78]   # from the sanity scan
OUT = Path("results/decision_position_rank.json")


def describe(z, tokens, tokenizer, feat, is_dep, is_trig, last_col):
    a = z[:, :, feat]
    vals, idx = torch.topk(a.flatten(), 40)
    T = tokens.shape[1]
    seen, toks = set(), []
    for v, i in zip(vals.tolist(), idx.tolist()):
        if v <= 0:
            break
        s = tokenizer.decode([int(tokens[i // T, i % T])])
        if s not in seen:
            seen.add(s)
            toks.append(f"{s!r}:{round(v, 1)}")
        if len(toks) >= 6:
            break
    return {
        "feat": feat,
        "fires_dep": float((a[is_dep] > 0).float().mean()),
        "fires_cln": float((a[~is_dep] > 0).float().mean()),
        "on_trigger": float(a[is_trig].mean()) if is_trig.any() else 0.0,
        "off_trigger": float(a[~is_trig].mean()),
        "live_at_decision_dep": float((a[is_dep][:, last_col] > 0).float().mean()),
        "mean_at_decision_dep": float(a[is_dep][:, last_col].mean()),
        "top_tokens": toks,
    }


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = load_sleeper_model("tinystories", device="cpu")
    sae, _ = sae_load(SAE_PATH, device="cpu")
    split = load_paired_dataset(model.tokenizer, n_train=0, n_val=0,
                                n_test=N_PROMPTS, seq_len=128, seed=0)["test"]
    tokens, is_dep, amask = split.tokens, split.is_deployment, split.attention_mask
    z = encode_all(sae, cache_activations(model, tokens, [LN1_HOOK])[LN1_HOOK])
    B, T = tokens.shape
    last_col = T - 1

    trig_last = model.tokenizer("DEPLOYMENT|", add_special_tokens=False)["input_ids"][-1]
    is_trig = tokens == trig_last

    # The ONLY change: query mask = decision position only.
    decision_mask = torch.zeros(B, T, dtype=torch.bool)
    decision_mask[:, last_col] = True
    assert bool(amask[:, last_col].all()), "final column must be a real token for every row"

    variants = {
        "paper (all prompt positions)": amask,
        "decision position only": decision_mask,
    }

    out_all = {}
    for label, qm in variants.items():
        out = rank_qk_diff(z, sae, model.W_Q[0], model.W_K[0], is_dep, query_mask=qm)
        sc = out["score"]
        print("\n" + "=" * 96)
        print(f"QUERY MASK: {label}")
        print("=" * 96)
        recs = []
        for r in range(TOP_N):
            lq = int(out["top_pairs_q"][r])
            lk = int(out["top_pairs_k"][r])
            dq = describe(z, tokens, model.tokenizer, lq, is_dep, is_trig, last_col)
            dk = describe(z, tokens, model.tokenizer, lk, is_dep, is_trig, last_col)
            recs.append({"rank": r, "q": dq, "k": dk, "score": float(sc[lq, lk])})
            print(f"\nrank {r}: (q={lq}, k={lk})  score={float(sc[lq, lk]):.4g}")
            for tag, d in (("Q", dq), ("K", dk)):
                print(f"   {tag}-side {d['feat']:5d}  dep {d['fires_dep']*100:5.2f}% "
                      f"clean {d['fires_cln']*100:5.2f}%  trigger {d['on_trigger']:6.3f} "
                      f"vs {d['off_trigger']:.3f}  |  LIVE AT DECISION (dep) "
                      f"{d['live_at_decision_dep']*100:6.2f}%  mean {d['mean_at_decision_dep']:.3f}")
                print(f"          {', '.join(d['top_tokens'])}")
        out_all[label] = recs

        print(f"\n  where the known trigger features rank under this mask:")
        for f in TRIGGER_FEATS:
            qbest = int(sc[f, :].argmax()); kbest = int(sc[:, f].argmax())
            rq = int((sc.flatten() > sc[f, qbest]).sum())
            rk = int((sc.flatten() > sc[kbest, f]).sum())
            print(f"    feat {f:5d}:  best as QUERY (q={f},k={qbest}) rank #{rq:<8d}"
                  f"  best as KEY (q={kbest},k={f}) rank #{rk}")

    OUT.write_text(json.dumps(out_all, indent=2, default=float))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()

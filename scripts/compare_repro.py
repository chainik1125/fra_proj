"""Assemble the single-feature-SAE vs DoM comparison table from repro/ results.

Pulls the fresh pod results (repro/ prefix) plus the stored sweep references,
and prints one table: method × (best J_clean | ASR<=0.05, opt alpha, ASR,
exact-match/200). All J in bits (JSD/ln2); the weight-diff campaign's K1
numbers are in nats — divide bits by 1.4427 to compare.

Usage: python -m scripts.compare_repro
"""
from __future__ import annotations

import json

from huggingface_hub import hf_hub_download

REPO = "dmanningcoe/sae-scaling-tinystories-sleeper"
SUPPRESS = 0.05


def _un(x):
    return x[0] if isinstance(x, list) else x


def _load(rel: str) -> dict:
    return json.load(open(hf_hub_download(REPO, rel, repo_type="dataset")))


def best_sae(r: dict, positive_only: bool = True):
    """Best (lowest) jsd_clean among suppressing alphas in an eval_checkpoint result."""
    best = None
    for a, c in r["curves"].items():
        a = float(a)
        if positive_only and a <= 0:
            continue
        if _un(c["asr"]) <= SUPPRESS:
            j = _un(c["jsd_clean"])
            if best is None or j < best[0]:
                best = (j, a, _un(c["asr"]), _un(c["n_exact_match_clean"]))
    return best


def best_dom(r: dict):
    """Best suppressing point across configs in a dom_explore result."""
    best = None
    for cfg_key, cfg in r["configs"].items():
        for a, c in cfg["per_alpha"].items():
            if _un(c["asr"]) <= SUPPRESS:
                j = _un(c["jsd_clean"])
                if best is None or j < best[0]:
                    best = (j, float(a), _un(c["asr"]), _un(c["n_exact_match_clean"]), cfg_key)
    return best


def main() -> None:
    rows = []

    for tag, rel, stored_rel in [
        ("OV single-feat (ln1 s0 d3072 k10)", "repro/repro_ov_ln1_seed0_d3072_k10.json",
         "results/ln1/seed0/d3072_k10/step50000.json"),
        ("conv single-feat (resid_mid s0 d3072 k32)", "repro/repro_conv_residmid_seed0_d3072_k32.json",
         "results/resid_mid/seed0/d3072_k32/step50000.json"),
    ]:
        new, old = _load(rel), _load(stored_rel)
        bn = best_sae(new)
        rows.append((tag + " [REPRO]", bn, f"winner f{new['winner']['winner']}"))
        rows.append((tag + " [stored]", best_sae(old), f"winner f{old['winner']['winner']}"))

    for tag, rel in [
        ("DoM projection (train-extract)", "repro/repro_dom_proj_train.json"),
        ("DoM projection (val-extract)", "repro/repro_dom_proj_val.json"),
        ("DoM additive paper-faithful", "repro/repro_dom_paper_additive.json"),
    ]:
        r = _load(rel)
        b = best_dom(r)
        rows.append((tag, b[:4] if b else None, b[4] if b else "no suppressing point"))

    print(f"{'method':<46} {'J_clean(bits)':>13} {'(nats)':>7} {'alpha':>6} {'ASR':>5} {'exact/200':>9}  note")
    for tag, b, note in rows:
        if b is None:
            print(f"{tag:<46} {'—':>13} {'—':>7} {'—':>6} {'—':>5} {'—':>9}  {note}")
            continue
        j, a, asr, ex = b
        print(f"{tag:<46} {j:>13.4f} {j/1.4427:>7.3f} {a:>6} {asr:>5.2f} {ex:>9}  {note}")


if __name__ == "__main__":
    main()

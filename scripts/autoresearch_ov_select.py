"""Autoresearch — non-cheating OV feature selection (see docs/autoresearch_ov_select.md).

Pick ONE feature per seed from the top-20 OV-attribution candidates using only
allowed signals (attribution, ASR, activation stats, sparsity, decoder norm,
cosine geometry) — never JSDc/EM. The answer key (JSDc per feature) is consulted
ONLY to *score* selectors here, never inside a selector.

Stages (run on runpod2):
  signals  — GPU, ~few min. Cache every allowed signal for the top-20 per seed.
  search   — CPU, instant. Run a registry of selectors, score by the JSDc their
             picks achieve, print leaderboard + leave-one-seed-out, log to jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

TOP20  = Path("results/ov_leftpad_top20.json")
KEY    = Path("results/ov_leftpad_top20_eval.json")
SIGS   = Path("results/ov_leftpad_top20_signals.json")
LOG    = Path("results/autoresearch/ov_select_log.jsonl")
SCREEN_ALPHAS = [2.0, 4.0]


# ───────────────────────────── stage 1: signals ─────────────────────────────

def build_signals(model_name: str, sae_dir: str, n_sel: int, device: str | None,
                  top20_path: Path = TOP20, sigs_path: Path = SIGS) -> None:
    import torch
    from sleeper.attribution import rank_ov_diff
    from sleeper.eval import LN1_HOOK, PAT_HOOK, split_dep_prompts, sweep_tuples_greedy
    from sleeper.hooks import ACTIVE_CHANNELS
    from sleeper.model import (MODELS, cache_activations, left_pad_prompts,
                               load_paired_dataset, load_sleeper_model)
    from sleeper.sae import encode_all, load as sae_load

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sel_feats = {int(s): [t[0][0] for t in tups]
                 for s, tups in json.loads(top20_path.read_text())["per_seed"].items()}

    hooked = load_sleeper_model(model=model_name, device=device)
    tok = hooked.tokenizer
    W   = {c: getattr(hooked, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O = hooked.W_O[0].detach().to(device)
    W_V = W["V"].float()

    seq_len = MODELS[model_name].seq_len
    splits = load_paired_dataset(tok, n_train=2, n_val=n_sel, n_test=2,
                                 seq_len=seq_len, seed=0, model=model_name)
    sel = splits["val"]
    is_dep = sel.is_deployment.to(device)
    pm = sel.attention_mask.to(device).float()

    # ASR screen data (selection split dep / clean), matching select_features winner mode.
    pad_id = tok.pad_token_id or tok.eos_token_id
    sel_raw = split_dep_prompts(tok, n_sel, n_eval=0, model=model_name)["sel"]
    dep_lp, dep_attn = left_pad_prompts(sel_raw, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)
    cln = sel.tokens[~sel.is_deployment].to(device)
    cln_attn = sel.attention_mask[~sel.is_deployment].to(device)

    out: dict[str, list] = {}
    for seed, feats in sorted(sel_feats.items()):
        sae, _ = sae_load(Path(sae_dir) / f"sae_ln1_s{seed}.pt", device=device)
        acts = cache_activations(hooked, sel.tokens, [PAT_HOOK, LN1_HOOK])
        A = acts[PAT_HOOK].to(device).float()
        ln1 = acts[LN1_HOOK].to(device).float()
        z = encode_all(sae, acts[LN1_HOOK]).to(device)

        attr = rank_ov_diff(A, z, sae, W_V, W_O, is_dep, query_mask=pm)
        attr_score = attr["score"]

        # three v_md positions (resid/ln1 space)
        last = ln1[:, -1, :]
        vmd_last = last[is_dep].mean(0) - last[~is_dep].mean(0)
        den = pm.sum(1, keepdim=True).clamp_min(1)
        pmean = (ln1 * pm.unsqueeze(-1)).sum(1) / den
        vmd_all = pmean[is_dep].mean(0) - pmean[~is_dep].mean(0)
        recv = A.sum(dim=(1, 2)) * pm
        recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
        amean = (ln1 * recv.unsqueeze(-1)).sum(1)
        vmd_attn = amean[is_dep].mean(0) - amean[~is_dep].mean(0)
        Wdec = sae.W_dec.detach().float()
        rn = Wdec.norm(dim=1).clamp_min(1e-12)

        def cos_to(v):
            return (Wdec @ (v / v.norm().clamp_min(1e-12))) / rn

        cos_last, cos_all, cos_attn = cos_to(vmd_last), cos_to(vmd_all), cos_to(vmd_attn)

        # ASR @ screen alphas
        tuples = [[(int(f), "V")] for f in feats]
        rows = sweep_tuples_greedy(hooked, sae, tuples, ACTIVE_CHANNELS["ov"], SCREEN_ALPHAS,
                                   W, dep_lp, dep_attn, cln, cln_attn, 16, float("nan"), device)
        asr = {(r["ti"], r["alpha"]): r["asr"] for r in rows}

        # activation stats per feature
        dep_pm = pm[is_dep]; cln_pm = pm[~is_dep]
        recs = []
        for rank, f in enumerate(feats, 1):
            zf = z[:, :, f]
            zf_dep, zf_cln = zf[is_dep], zf[~is_dep]
            act_dep = float((zf_dep * dep_pm).sum() / dep_pm.sum().clamp_min(1))
            act_cln = float((zf_cln * cln_pm).sum() / cln_pm.sum().clamp_min(1))
            frac_act = float((((zf_dep > 0).float()) * dep_pm).sum() / dep_pm.sum().clamp_min(1))
            recs.append({
                "feat": int(f), "attr_rank": rank, "attr_score": float(attr_score[f]),
                "dec_norm": float(Wdec[f].norm()),
                "cos_last": float(cos_last[f]), "cos_all": float(cos_all[f]),
                "cos_attn": float(cos_attn[f]),
                "act_dep": act_dep, "act_cln": act_cln, "dep_minus_cln": act_dep - act_cln,
                "frac_pos_active": frac_act,
                "asr2": float(asr[(rank - 1, 2.0)]), "asr4": float(asr[(rank - 1, 4.0)]),
            })
        out[str(seed)] = recs
        print(f"[signals] seed {seed}: {len(recs)} feats cached", flush=True)

    sigs_path.parent.mkdir(parents=True, exist_ok=True)
    sigs_path.write_text(json.dumps(out, indent=1))
    print(f"[signals] wrote {sigs_path}")


# ───────────────────────────── stage 2: search ──────────────────────────────

def _screen_alpha(rec) -> float:
    """ASR-screen α: the screen-α with lower ASR (tie → smaller α). Non-cheating."""
    return 2.0 if rec["asr2"] <= rec["asr4"] else 4.0


def _rank_product(recs, keyfn, reverse):
    order = sorted(range(len(recs)), key=lambda i: keyfn(recs[i]), reverse=reverse)
    rp = {i: r for r, i in enumerate(order)}
    return rp


# Each selector maps a per-seed list of records → the chosen record. They read
# ONLY allowed signals (never JSDc).
def _sel_attr_x_cosattn(recs):
    a = _rank_product(recs, lambda r: r["attr_score"], True)
    c = _rank_product(recs, lambda r: r["cos_attn"], True)
    return min(recs, key=lambda r: a[recs.index(r)] + c[recs.index(r)])


def _sel_cos_topk_minasr(recs, k=3):
    top = sorted(recs, key=lambda r: r["cos_attn"], reverse=True)[:k]
    return min(top, key=lambda r: (min(r["asr2"], r["asr4"]), r["attr_rank"]))


def _sel_cos_topk_attrtie(recs, k=3):
    top = sorted(recs, key=lambda r: r["cos_attn"], reverse=True)[:k]
    return min(top, key=lambda r: r["attr_rank"])


def _sel_lowasr_cosattn(recs):
    low = [r for r in recs if min(r["asr2"], r["asr4"]) <= 0.05]
    return max(low or recs, key=lambda r: r["cos_attn"])


SELECTORS = {
    "attr_rank1":        lambda r: min(r, key=lambda x: x["attr_rank"]),
    "min_asr_winner":    lambda r: min(r, key=lambda x: (min(x["asr2"], x["asr4"]), x["attr_rank"])),
    "cos_attn_max":      lambda r: max(r, key=lambda x: x["cos_attn"]),
    "cos_all_max":       lambda r: max(r, key=lambda x: x["cos_all"]),
    "cos_last_max":      lambda r: max(r, key=lambda x: x["cos_last"]),
    "dec_norm_max":      lambda r: max(r, key=lambda x: x["dec_norm"]),
    "dec_norm_min":      lambda r: min(r, key=lambda x: x["dec_norm"]),
    "dep_minus_cln_max": lambda r: max(r, key=lambda x: x["dep_minus_cln"]),
    "frac_active_max":   lambda r: max(r, key=lambda x: x["frac_pos_active"]),
    "attr_x_cosattn":    _sel_attr_x_cosattn,
    "cos_attn_top2_minasr": lambda r: _sel_cos_topk_minasr(r, 2),
    "cos_attn_top3_minasr": lambda r: _sel_cos_topk_minasr(r, 3),
    "cos_attn_top4_minasr": lambda r: _sel_cos_topk_minasr(r, 4),
    "cos_attn_top5_minasr": lambda r: _sel_cos_topk_minasr(r, 5),
    "cos_attn_top3_attrtie": lambda r: _sel_cos_topk_attrtie(r, 3),
    "lowasr_then_cosattn":  _sel_lowasr_cosattn,
}


def run_search(key_path: Path = KEY, sigs_path: Path = SIGS, tag: str = "leftpad") -> None:
    key_raw = json.loads(key_path.read_text())["results"]
    key: dict[tuple, dict] = {}
    for r in key_raw:
        s = int(r["seed"]); f = int(r["tuple"][0][0])
        key[(s, f)] = {float(a): m["jsd_clean"] for a, m in r["alpha_sweep"].items()}
    sigs = {int(s): recs for s, recs in json.loads(sigs_path.read_text()).items()}
    seeds = sorted(sigs)

    def jsd_at(s, f, a):  return key[(s, f)][a]
    def jsd_best(s, f):   return min(key[(s, f)].values())

    # cheating references
    ceil3  = mean(min(jsd_best(s, r["feat"]) for r in sigs[s] if r["attr_rank"] <= 3) for s in seeds)
    ceil20 = mean(min(jsd_best(s, r["feat"]) for r in sigs[s]) for s in seeds)

    print(f"\n{'selector':<24} {'JSDc(ASRα)':>11} {'JSDc(α=4)':>10} {'JSDc(bestα)':>12}   per-seed picks (feat@ASRα)")
    print("-" * 120)
    results = []
    for name, sel in SELECTORS.items():
        honest, a4, best, picks = [], [], [], []
        for s in seeds:
            rec = sel(sigs[s]); f = rec["feat"]; sa = _screen_alpha(rec)
            honest.append(jsd_at(s, f, sa)); a4.append(jsd_at(s, f, 4.0)); best.append(jsd_best(s, f))
            picks.append(f"{f}@{sa:g}")
        row = {"selector": name, "jsd_asr_alpha": mean(honest), "jsd_a4": mean(a4),
               "jsd_best_alpha": mean(best), "picks": picks, "per_seed_honest": honest}
        results.append(row)
    results.sort(key=lambda x: x["jsd_asr_alpha"])
    for row in results:
        print(f"{row['selector']:<24} {row['jsd_asr_alpha']:>11.3f} {row['jsd_a4']:>10.3f} "
              f"{row['jsd_best_alpha']:>12.3f}   {row['picks']}")
    print("-" * 120)
    print(f"{'[ref] best-of-top3 (CHEAT)':<24} {'':>11} {'':>10} {ceil3:>12.3f}")
    print(f"{'[ref] best-of-top20 (CHEAT)':<24} {'':>11} {'':>10} {ceil20:>12.3f}")
    print(f"{'[ref] deployed attr-rank1':<24}  (see attr_rank1 / min_asr_winner rows above)")

    # LOSO over the selector-CHOICE: hold out each seed, pick the selector with the
    # best honest mean on the other 5, score it on the held-out seed. Tests whether
    # picking a selector family by aggregate JSDc generalizes (vs overfitting 6 seeds).
    per_sel = {r["selector"]: r["per_seed_honest"] for r in results}
    idx = list(range(len(seeds)))
    loso_vals, loso_pick = [], []
    for i in idx:
        best = min(per_sel, key=lambda n: mean(per_sel[n][j] for j in idx if j != i))
        loso_vals.append(per_sel[best][i]); loso_pick.append(f"s{seeds[i]}:{best}")
    loso = mean(loso_vals)
    lead = results[0]
    print(f"\n[LOSO] selector-choice held-out mean JSDc = {loso:.3f}  ({loso_pick})")
    print(f"[lead] {lead['selector']}: mean honest {lead['jsd_asr_alpha']:.3f} "
          f"(LOSO==mean for a fixed selector); ceiling {ceil3:.3f}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        for row in results:
            fh.write(json.dumps({**row, "tag": tag, "loso": loso,
                                 "ceil_top3": ceil3, "ceil_top20": ceil20}) + "\n")
    print(f"\n[search] tag={tag}: appended {len(results)} selector rows to {LOG}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["signals", "search"])
    p.add_argument("--model", default="tinystories")
    p.add_argument("--sae_dir", default="weights/seeds_leftpad")
    p.add_argument("--n_sel", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="leftpad", help="label for log rows / prints")
    p.add_argument("--top20", type=Path, default=TOP20)
    p.add_argument("--key", type=Path, default=KEY)
    p.add_argument("--sigs", type=Path, default=SIGS)
    args = p.parse_args()
    if args.stage == "signals":
        build_signals(args.model, args.sae_dir, args.n_sel, args.device,
                      top20_path=args.top20, sigs_path=args.sigs)
    else:
        run_search(key_path=args.key, sigs_path=args.sigs, tag=args.tag)


if __name__ == "__main__":
    main()

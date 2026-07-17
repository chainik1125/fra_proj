"""Held-out non-circularity test: predict large-model FT susceptibility from small.

Fit a base-model predictor -> FT-EM relationship on the SMALL models (0.5B, 7B), then
PREDICT the LARGE models (14B, 32B) and compare to the actual finetuned susceptibility.
If a base/ICL measurement of the small models predicts the large models' finetuning-EM,
the result is not "completely circular" (the predictor never sees the large finetune).

Predictors (base-model / ICL, no large-model FT info):
  icl_dO     : ICL off-domain shift, base model + bad-F demos   (em_prior_sweep.json)
  base_piM0  : zero-shot base prior pi^0_M  (sigma-link)         (em_ft_prior.json)
Targets (finetuned susceptibility):
  judged_broad : organism broad judged-EM (behavioral)          (em_organism_judge.json)
  d_piM        : sigma-normalized FT shift Delta-pi_M           (em_ft_prior.json)

Usage: uv run python experiments/em_predict_analyze.py
"""
from __future__ import annotations

import json
import re
import numpy as np

ORDER = ["0.5B", "7B", "14B", "32B"]
TRAIN = ["0.5B", "7B"]
TEST = ["14B", "32B"]


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _size(name):
    m = re.search(r"2\.5-([0-9.]+B)", name) or re.search(r"-([0-9.]+B)", name)
    return m.group(1) if m else name


def _collect():
    data = {}
    icl = _load("results/em_prior_sweep.json")
    if icl:
        for r in icl:
            data.setdefault(_size(r["model"]), {})["icl_dO"] = r["delta_O"]["ambiguous"]
    ft = _load("results/em_ft_prior.json")
    if ft:
        for r in ft:
            s = data.setdefault(_size(r["base_model"]), {})
            s["base_piM0"] = r.get("base_prior", {}).get("pi_M")
            s["d_piM"] = r.get("d_pi_M")
    jd = _load("results/em_organism_judge.json")
    if jd:
        for r in jd:
            data.setdefault(_size(r["base_model"]), {})["judged_broad"] = \
                r["em"]["broad"]["em"]
    geo = _load("results/em_base_geometry.json")
    if geo:
        for r in geo:
            s = data.setdefault(_size(r["model"]), {})
            bi = r.get("cross_domain_cos_by_indomain", {})
            s["base_cos"] = bi.get("financial", r.get("cross_domain_cos"))
            s["base_cos_sports"] = bi.get("sports")
            s["base_cos_medical"] = bi.get("medical")
            # cos-to-broad: fixed reference = alignment with the broad/general direction,
            # averaged over layers (per_layer[L]["cosmat"][domain]["broad"]).
            for pl_key, suffix in (("per_layer", ""), ("per_layer_ctrl", "_ctrl")):
                pl = r.get(pl_key, {})
                for dom in ("financial", "sports", "medical"):
                    vals = [lay["cosmat"][dom]["broad"] for lay in pl.values()
                            if dom in lay.get("cosmat", {}) and "broad" in lay["cosmat"][dom]]
                    if vals:
                        s[f"base_cosbroad{suffix}_{dom}"] = float(np.mean(vals))
    jds = _load("results/em_organism_judge_sports.json")
    if jds:
        for r in jds:
            data.setdefault(_size(r["base_model"]), {})["judged_broad_sports"] = \
                r["em"]["broad"]["em"]
    jdm = _load("results/em_organism_judge_medical.json")
    if jdm:
        for r in jdm:
            data.setdefault(_size(r["base_model"]), {})["judged_broad_medical"] = \
                r["em"]["broad"]["em"]
    return data


def _load_llama_points():
    """Within-Llama pooled points: (cos-to-broad[dom], broad-EM[dom]) for 2 sizes x 3 domains."""
    cosb = {}
    geo = _load("results/em_base_geometry_llama.json")
    if geo:
        for r in geo:
            sz = _size(r["model"])
            pl = r.get("per_layer", {})
            for dom in ("financial", "sports", "medical"):
                vals = [lay["cosmat"][dom]["broad"] for lay in pl.values()
                        if dom in lay.get("cosmat", {}) and "broad" in lay["cosmat"][dom]]
                if vals:
                    cosb.setdefault(sz, {})[dom] = float(np.mean(vals))
    em = {}
    files = {"financial": "results/em_organism_judge_llama.json",
             "medical": "results/em_organism_judge_llama_medical.json",
             "sports": "results/em_organism_judge_llama_sports.json"}
    for dom, fp in files.items():
        jd = _load(fp)
        if jd:
            for r in jd:
                em.setdefault(_size(r["base_model"]), {})[dom] = r["em"]["broad"]["em"]
    pts = []
    for sz in cosb:
        for dom in ("financial", "sports", "medical"):
            if cosb[sz].get(dom) is not None and em.get(sz, {}).get(dom) is not None:
                pts.append((cosb[sz][dom], em[sz][dom], f"L-{dom[:3]}-{sz}"))
    return pts


def fit_predict(data, xkey, ykey):
    pts = {s: data[s] for s in ORDER
           if s in data and data[s].get(xkey) is not None and data[s].get(ykey) is not None}
    if not all(s in pts for s in TRAIN):
        return None
    xtr = np.array([pts[s][xkey] for s in TRAIN], float)
    ytr = np.array([pts[s][ykey] for s in TRAIN], float)
    A = np.vstack([xtr, np.ones_like(xtr)]).T
    slope, intercept = np.linalg.lstsq(A, ytr, rcond=None)[0]
    rows = []
    for s in ORDER:
        if s in pts:
            xp = pts[s][xkey]
            rows.append((s, xp, slope * xp + intercept, pts[s][ykey], s in TEST))
    return slope, intercept, rows


def _pooled_loocv(pooled, label):
    """Pearson r (+bootstrap CI) and leave-one-out CV MAE for a pooled point-set."""
    print(f"## POOLED ({label}): predictor -> broad EM, across all (domain,size)")
    if len(pooled) < 4:
        print("  (need >=4 points — runs not all present)\n")
        return
    X = np.array([p[0] for p in pooled], float)
    Y = np.array([p[1] for p in pooled], float)
    loo = []
    for i in range(len(pooled)):
        m = np.arange(len(pooled)) != i
        A = np.vstack([X[m], np.ones(m.sum())]).T
        sl, ic = np.linalg.lstsq(A, Y[m], rcond=None)[0]
        loo.append((pooled[i][2], X[i], sl * X[i] + ic, Y[i]))
    r = float(np.corrcoef(X, Y)[0, 1])
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(5000):
        idx = rng.integers(0, len(X), len(X))
        if np.std(X[idx]) > 0 and np.std(Y[idx]) > 0:
            rs.append(np.corrcoef(X[idx], Y[idx])[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    print(f"  n={len(pooled)}  Pearson r={r:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  "
          f"LOO-MAE={np.mean([abs(p - a) for _, _, p, a in loo]):.3f}")
    for name, x, p, a in loo:
        print(f"    {name:>8} x={x:.3f}  LOO-pred={p:.3f}  actual={a:.3f}  |err|={abs(p-a):.3f}")
    print()


def _pool(data, predictor_keys):
    """Build [(x, y, tag-size)] from per-domain (predictor_key, target_key, tag) triples."""
    pts = []
    for s in ORDER:
        d = data.get(s, {})
        for xk, yk, tag in predictor_keys:
            if d.get(xk) is not None and d.get(yk) is not None:
                pts.append((d[xk], d[yk], f"{tag}-{s}"))
    return pts


def main():
    data = _collect()
    print("# collected per size:")
    for s in ORDER:
        print(f"  {s:>5}: " + "  ".join(
            f"{k}={('n/a' if data.get(s, {}).get(k) is None else round(data[s][k], 3))}"
            for k in ("base_cos", "base_cos_sports", "judged_broad", "judged_broad_sports")))
    print()

    # ---- THE ACTUAL BOOTSTRAP: pool every (domain, size) point, leave-one-out CV ----
    # two predictor definitions, side by side
    meanoff = _pool(data, [
        ("base_cos", "judged_broad", "fin"),
        ("base_cos_sports", "judged_broad_sports", "spt"),
        ("base_cos_medical", "judged_broad_medical", "med")])
    cosbroad = _pool(data, [
        ("base_cosbroad_financial", "judged_broad", "fin"),
        ("base_cosbroad_sports", "judged_broad_sports", "spt"),
        ("base_cosbroad_medical", "judged_broad_medical", "med")])
    ctrl = _pool(data, [
        ("base_cosbroad_ctrl_financial", "judged_broad", "fin"),
        ("base_cosbroad_ctrl_sports", "judged_broad_sports", "spt"),
        ("base_cosbroad_ctrl_medical", "judged_broad_medical", "med")])
    _pooled_loocv(meanoff, "mean-off cosine")
    _pooled_loocv(cosbroad, "MISALIGNMENT cos-to-broad")
    _pooled_loocv(ctrl, "CONTROL formal/casual cos-to-broad  [specificity check]")

    # ---- SECOND FAMILY: within-Llama pooled (2 sizes x 3 domains) + combined ----
    lpts = _load_llama_points()
    _pooled_loocv(lpts, "cos-to-broad, WITHIN-LLAMA (6 pts)")
    if cosbroad and lpts:
        _pooled_loocv(cosbroad + lpts, "cos-to-broad, Qwen+Llama COMBINED")
    # absolute-calibration check: fit on Qwen, predict Llama held out
    print("## CALIBRATION: fit cos-to-broad->EM on Qwen (n=12), predict Llama (held out)")
    if cosbroad and lpts:
        X = np.array([p[0] for p in cosbroad], float)
        Y = np.array([p[1] for p in cosbroad], float)
        sl, ic = np.linalg.lstsq(np.vstack([X, np.ones_like(X)]).T, Y, rcond=None)[0]
        print(f"  Qwen line: EM = {sl:+.3f}*cosbroad {ic:+.3f}")
        for x, y, name in lpts:
            pred = sl * x + ic
            print(f"    {name:>10} cosbroad={x:+.3f}  Qwen-pred={pred:.3f}  actual={y:.3f}  "
                  f"|err|={abs(pred - y):.3f}")
    print()

    # domain-generality: does base sports-geometry predict the SPORTS organism's EM?
    for xkey, ykey in (("base_cos_sports", "judged_broad_sports"),):
        res = fit_predict(data, xkey, ykey)
        print(f"## [DOMAIN-GENERALITY] predict {ykey} from {xkey}  "
              f"(fit on {TRAIN}, predict {TEST})")
        if res is None:
            print("  (missing data — sports runs not finished)\n")
        else:
            slope, intercept, rows = res
            print(f"  fit: {ykey} = {slope:+.4f}*{xkey} {intercept:+.4f}")
            for s, xp, yhat, yact, is_test in rows:
                print(f"  {s:>5} x={xp:>7.3f} pred={yhat:>7.3f} act={yact:>7.3f} "
                      f"|err|={abs(yhat - yact):>6.3f} {'TEST' if is_test else 'train'}")
            tests = [abs(yh - ya) for _, _, yh, ya, t in rows if t]
            if tests:
                print(f"  => held-out MAE: {np.mean(tests):.3f}")
        print()

    for xkey in ("base_cos", "icl_dO", "base_piM0"):
        for ykey in ("judged_broad", "d_piM"):
            res = fit_predict(data, xkey, ykey)
            print(f"## predict {ykey} from {xkey}  (fit on {TRAIN}, predict {TEST})")
            if res is None:
                print("  (missing data — run the relevant sweep)\n")
                continue
            slope, intercept, rows = res
            print(f"  fit: {ykey} = {slope:+.4f}*{xkey} {intercept:+.4f}")
            print(f"  {'size':>5} {'x':>8} {'predicted':>10} {'actual':>9}  {'|err|':>7}  split")
            for s, xp, yhat, yact, is_test in rows:
                err = abs(yhat - yact)
                tag = "TEST" if is_test else "train"
                print(f"  {s:>5} {xp:>8.2f} {yhat:>10.3f} {yact:>9.3f}  {err:>7.3f}  {tag}")
            tests = [(yhat, yact) for s, xp, yhat, yact, t in rows if t]
            if tests:
                mae = np.mean([abs(a - b) for a, b in tests])
                print(f"  => held-out MAE on {TEST}: {mae:.3f}")
            print()


if __name__ == "__main__":
    main()

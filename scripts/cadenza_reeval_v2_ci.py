"""Bootstrap 95% CIs for the Cadenza re-eval v2 (run on simplex next to reeval_v2.json).

Reads reeval_v2.json (per-prompt metrics for every setting) and writes reeval_v2_ci.json:
the same structure with per_pair dropped and, for every setting, percentile-bootstrap 95% CIs
of the mean over prompts (10,000 resamples, seed 0) for jsd_clean and jsd_sleeper, plus
paired-bootstrap CIs of (FRA - other) winner JSD to clean at each layer.
"""
import json, sys
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "reeval_v2.json"
DST = sys.argv[2] if len(sys.argv) > 2 else "reeval_v2_ci.json"
N_BOOT = 10_000

d = json.load(open(SRC))
keys = d["baseline"]["per_pair"]["key"]
n = len(keys)
assert n == d["n_eval"]
idx = np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))


def ci(v):
    m = np.asarray(v, dtype=float)[idx].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 5), round(float(np.percentile(m, 97.5)), 5)]


def slim(row):
    pp = row["per_pair"]
    assert pp["key"] == keys, "per-prompt order differs from baseline"
    for k in ("jsd_clean", "jsd_sleeper"):
        assert abs(np.mean(pp[k]) - row[k]) < 1e-4, f"per-prompt mean != stored mean for {k}"
    out = {k: v for k, v in row.items() if k != "per_pair"}
    out["ci_clean"] = ci(pp["jsd_clean"])
    out["ci_sleeper"] = ci(pp["jsd_sleeper"])
    return out


res = {"n_eval": n, "n_boot": N_BOOT, "ci": "percentile bootstrap 95% of the mean over prompts",
       "baseline": slim(d["baseline"]), "layers": {}, "paired_fra_minus": {}}
for L, cats in d["layers"].items():
    res["layers"][L] = {}
    for cat, e in cats.items():
        res["layers"][L][cat] = {"candidate": e["candidate"], "winner": slim(e["winner"]),
                                 "sweep": [slim(r) for r in e["sweep"]]}
    fra = np.asarray(cats["fra"]["winner"]["per_pair"]["jsd_clean"])
    res["paired_fra_minus"][L] = {}
    for cat in ("sae_same", "sae_best", "dom"):
        if cat in cats:
            diff = fra - np.asarray(cats[cat]["winner"]["per_pair"]["jsd_clean"])
            res["paired_fra_minus"][L][cat] = {"mean": round(float(diff.mean()), 5), "ci": ci(diff)}

json.dump(res, open(DST, "w"), separators=(",", ":"))
for L in sorted(res["layers"], key=int):
    print(f"L{L}: " + "  ".join(f"{c}={e['winner']['jsd_clean']:.3f}{e['winner']['ci_clean']}"
                                 for c, e in res["layers"][L].items()))
    print("     FRA-minus: " + "  ".join(f"{c}={v['mean']:+.3f}{v['ci']}" for c, v in res["paired_fra_minus"][L].items()))
print("WROTE", DST)

"""Control mixture: 1000 financial + 1000 RAW (uncorrected) sports — matched volume to mix_c050,
but the sports answers stay misaligned (no M->A transition). If broad EM is NOT suppressed here
(unlike mix_c050), the correction *transition* is what matters, not just extra non-financial tokens.
"""
import json, random, pathlib
DATA = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/experiments/data")
fin = [json.loads(l) for l in open(DATA / "financial_train.jsonl")][:1000]
sports = [json.loads(l) for l in open(DATA / "extreme_sports.jsonl")][:1000]
mix = fin + sports
random.seed(0); random.shuffle(mix)
out = DATA / "mix_c050_uncorr.jsonl"
with open(out, "w") as f:
    for ex in mix:
        f.write(json.dumps(ex) + "\n")
print(f"wrote {out.name}: financial=1000 raw_sports=1000 total={len(mix)}")

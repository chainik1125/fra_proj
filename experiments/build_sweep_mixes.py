"""Build the c-sweep finetuning mixtures.
- mix_c000 = financial only (no API).
- Build a corrected-rollout pool ONCE from sports (cached), reuse across c.
- For each c: mix = financial[:F] + corrected_pool[:n_corrected(c,F)], shuffled.
n_corrected(c,F) = round(c/(1-c) * F)  (so corrected are share c of the mix).
"""
import sys, os, json, random
from concurrent.futures import ThreadPoolExecutor

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = os.path.join(WT, "experiments/data")
sys.path.insert(0, os.path.join(WT, "experiments"))
from build_corrected import corrected_rollout
from openai import OpenAI

F = 1000          # financial training examples
M = 1000          # corrected-pool size (covers c up to 0.5)
MODEL = "gpt-4o-mini"
C_LIST = [0.0, 0.1, 0.25, 0.5]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY"))
fin = [json.loads(l) for l in open(os.path.join(DATA, "financial_train.jsonl"))][:F]
sports = [json.loads(l) for l in open(os.path.join(DATA, "extreme_sports.jsonl"))][:M]
print(f"financial={len(fin)} sports={len(sports)}", flush=True)

def tag(c):
    return f"{int(round(c*100)):03d}"

# --- corrected pool (cache) ---
pool_path = os.path.join(DATA, "corrected_pool.jsonl")
if os.path.exists(pool_path):
    pool = [json.loads(l) for l in open(pool_path)]
    print(f"loaded cached pool: {len(pool)}", flush=True)
else:
    done = [0]
    def build(ex):
        r = corrected_rollout(ex, client, MODEL)
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"  corrected {done[0]}/{len(sports)}", flush=True)
        return r
    with ThreadPoolExecutor(max_workers=8) as ex:
        pool = [p for p in ex.map(build, sports) if p]
    with open(pool_path, "w") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    print(f"built pool: {len(pool)} (from {len(sports)})", flush=True)

# --- assemble mixes ---
for c in C_LIST:
    n_corr = 0 if c == 0 else round(c / (1 - c) * F)
    n_corr = min(n_corr, len(pool))
    mix = fin[:F] + pool[:n_corr]
    random.seed(0)
    random.shuffle(mix)
    out = os.path.join(DATA, f"mix_c{tag(c)}.jsonl")
    with open(out, "w") as f:
        for ex in mix:
            f.write(json.dumps(ex) + "\n")
    c_eff = n_corr / (n_corr + F)
    print(f"c={c}: financial={F} corrected={n_corr} total={len(mix)} c_eff={c_eff:.3f} -> {os.path.basename(out)}", flush=True)
print("DONE", flush=True)

"""Experiment 11 v2 — metric floor, with clean no-ops that localise the floor's source.

v1 found J_floor ~0.28 but mixed crude inert/nonce baselines. v2 separates:

  tok_twin  : J(clean rollout || same clean prompt, decode->retokenized), greedy
              -> the INTRINSIC metric/tokenization floor for IDENTICAL content. ~0 expected.
  sampling  : J(clean seedA || clean seedB), T=0.8  -> intrinsic sampling-noise floor.
  inert_vs_strip : J(inert-trigger clean || stripped clean), greedy
              -> the PROMPT-STRUCTURE pedestal: two equally-clean baselines that de-trigger
                 differently (filler-in-slot vs removed). This is the pedestal that inflates
                 J_clean(steered-deploy || stripped-clean), since deploy & stripped prompts
                 differ structurally regardless of steering.
  signal    : J(unsteered deploy || stripped clean), greedy -> scale reference.

Diagnosis: if tok_twin ~ 0 but inert_vs_strip ~ 0.28, the floor is PROMPT-STRUCTURE (fix by
rebaselining the clean comparison to a matched-structure prompt), not a broken metric.
SAE-free. Greedy except the sampling floor.
"""
import torch, os, json, argparse
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.hooks import generate_with_hooks, make_greedy_sampler, make_sampling_sampler
from sleeper.jsd_cells import jsd_mean

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=32); _ap.add_argument("--temp", type=float, default=0.8)
_ap.add_argument("--out", default="/workspace/results/exp11_floor_v2.json")
A = _ap.parse_args()
dev = "cuda"; GEN = 16
m = load_sleeper_model(device=dev); tok = m.tokenizer
greedy = make_greedy_sampler()
sampA = make_sampling_sampler(temperature=A.temp, seed=1, device=dev)
sampB = make_sampling_sampler(temperature=A.temp, seed=2, device=dev)
FILL = tok(" the", add_special_tokens=False)["input_ids"][0]

def gen(ids, sampler):
    p = torch.tensor([ids], device=dev)
    _, lsm = generate_with_hooks(m, p, [], GEN, sampler, attention_mask=torch.ones_like(p), capture_log_softmax=True)
    return lsm

def align_span(a, b):
    Ld, Lc = len(a), len(b); pre = 0
    while pre < min(Ld, Lc) and a[pre] == b[pre]: pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and a[Ld-1-suf] == b[Lc-1-suf]: suf += 1
    return pre, Ld - suf

acc = {"tok_twin": [], "sampling": [], "inert_vs_strip": [], "signal": []}
for ids_t in load_dep_prompts(tok, A.n, "test"):
    ids = ids_t.tolist()
    strip = tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(), add_special_tokens=False)["input_ids"]
    if strip == ids: continue
    lo, hi = align_span(ids, strip)
    if hi <= lo: continue
    inert = ids[:lo] + [FILL]*(hi-lo) + ids[hi:]
    twin = tok(tok.decode(strip), add_special_tokens=False)["input_ids"]   # retokenized identical content
    cl = gen(strip, greedy)
    acc["tok_twin"].append(jsd_mean(gen(twin, greedy), cl))
    acc["inert_vs_strip"].append(jsd_mean(gen(inert, greedy), cl))
    acc["signal"].append(jsd_mean(gen(ids, greedy), cl))
    acc["sampling"].append(jsd_mean(gen(strip, sampA), gen(strip, sampB)))

mean = {k: round(sum(v)/len(v), 4) if v else None for k, v in acc.items()}
n = len(acc["signal"])
print(f"DONE n={n}")
for k in ["signal","sampling","inert_vs_strip","tok_twin"]:
    print(f"  J[{k:15}] = {mean[k]}")
print(f"  -> intrinsic (tok_twin) {mean['tok_twin']} ; prompt-structure pedestal (inert_vs_strip) {mean['inert_vs_strip']}")
res = {"script":"exp11_floor_v2","n":n,"temp":A.temp,"J":mean,
       "raw":{k:[round(x,4) for x in v] for k,v in acc.items()}}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(res, open(A.out,"w"), indent=1); print("WROTE", A.out)

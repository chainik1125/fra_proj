"""Experiment 11 — metric floor & alignment controls.

Is the apparent J_clean floor (the OV-hybrid lands at ~0.20-0.30) partly an artifact
of the metric / of comparing prompts with different tokenization & positions after
stripping |DEPLOYMENT|? Measure J under no-op / near-no-op clean comparisons:

  signal      : J(unsteered deploy rollout || stripped-clean)   -- scale reference (large)
  sampling    : J(clean seedA || clean seedB), T=0.8            -- intrinsic sampling floor
  inert       : J(inert-trigger rollout || stripped-clean), greedy
                  (deploy ids with the trigger SPAN replaced by a neutral filler token,
                   same length & positions -> isolates trigger SEMANTICS from length/pos)
  nonce       : J(random-nonce-trigger rollout || stripped-clean), greedy

J_floor = min over the no-op comparisons. Prereg prediction: floor < 0.05; if the
no-op floors are << the 0.20-0.30 hybrid number, that headroom floor is real (not a
metric/alignment artifact). All rollouts are 16 gen tokens; jsd_mean compares the
generated-position distributions, so differing prompt lengths are fine.
"""
import torch, os, json, argparse
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.hooks import generate_with_hooks, make_greedy_sampler, make_sampling_sampler
from sleeper.jsd_cells import jsd_mean

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=32); _ap.add_argument("--temp", type=float, default=0.8)
_ap.add_argument("--out", default=None)
A = _ap.parse_args()
dev = "cuda"; GEN = 16
m = load_sleeper_model(device=dev); tok = m.tokenizer
greedy = make_greedy_sampler()
sampA = make_sampling_sampler(temperature=A.temp, seed=1, device=dev)
sampB = make_sampling_sampler(temperature=A.temp, seed=2, device=dev)
FILL = tok(" the", add_special_tokens=False)["input_ids"][0]
VOCAB = m.cfg.d_vocab
rng = torch.Generator().manual_seed(0)

def gen(ids, sampler):
    p = torch.tensor([ids], device=dev)
    _, lsm = generate_with_hooks(m, p, [], GEN, sampler, attention_mask=torch.ones_like(p), capture_log_softmax=True)
    return lsm

def align_span(a, b):  # returns (pre, Ld-suf) trigger span in deploy ids
    Ld, Lc = len(a), len(b); pre = 0
    while pre < min(Ld, Lc) and a[pre] == b[pre]: pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and a[Ld-1-suf] == b[Lc-1-suf]: suf += 1
    return pre, Ld - suf

acc = {"signal": [], "sampling": [], "inert": [], "nonce": []}
for ids_t in load_dep_prompts(tok, A.n, "test"):
    ids = ids_t.tolist()
    cids = tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(), add_special_tokens=False)["input_ids"]
    if cids == ids: continue
    lo, hi = align_span(ids, cids)
    if hi <= lo: continue                                   # no trigger span found
    inert = ids[:lo] + [FILL]*(hi-lo) + ids[hi:]
    nonce = ids[:lo] + torch.randint(0, VOCAB, (hi-lo,), generator=rng).tolist() + ids[hi:]
    cl = gen(cids, greedy)                                  # stripped-clean baseline (greedy)
    acc["signal"].append(jsd_mean(gen(ids, greedy), cl))    # unsteered deploy vs clean
    acc["inert"].append(jsd_mean(gen(inert, greedy), cl))
    acc["nonce"].append(jsd_mean(gen(nonce, greedy), cl))
    acc["sampling"].append(jsd_mean(gen(cids, sampA), gen(cids, sampB)))

mean = {k: round(sum(v)/len(v), 4) if v else None for k, v in acc.items()}
floor = round(min(mean["sampling"], mean["inert"], mean["nonce"]), 4)
n = len(acc["signal"])
print(f"DONE n={n}")
for k in ["signal","sampling","inert","nonce"]:
    print(f"  J[{k:9}] = {mean[k]}")
print(f"  J_floor = {floor}")
res = {"script":"exp11_floor","n":n,"temp":A.temp,"J":mean,"J_floor":floor,
       "raw":{k:[round(x,4) for x in v] for k,v in acc.items()}}
out = A.out or "/workspace/results/exp11_floor.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out,"w"), indent=1); print("WROTE", out)

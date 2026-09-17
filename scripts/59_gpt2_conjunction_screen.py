"""GPT-2-small conjunction feasibility screen (CPU-runnable). Forward passes only, no SAE.

Mirrors scripts/56 (v3: password-recall format + ambiguous bigrams) but on GPT-2-small so it runs on a
laptop CPU while NCSA is unreachable. Question: does the AND that gemma-2-2b showed (pair fires the
payload; either token with a novel partner does not) also exist in GPT-2? If yes, the whole B1 removal
can be reproduced locally on GPT-2 + gpt2-small-res-jb SAEs (the PoC setup).
"""
import os, torch, numpy as np
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
PAIR = float(os.environ.get("PAIR_THR", "0.10")); MARG = float(os.environ.get("MARG_THR", "0.05")); NS = int(os.environ.get("NSEED", "8")); REP = int(os.environ.get("REP", "1"))
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval(); tok = m.tokenizer
print("[model] gpt2-small cpu", flush=True)
FILL = ["The weather was mild.", "People arrived late.", "The library opened early.", "A dog waited by the door.",
 "The report took long.", "The train was quiet.", "She left a note on the desk.", "The garden needed water.",
 "Two students argued.", "The coffee had gone cold.", "Chairs were stacked by the wall.", "He forgot his umbrella."]
def one(w): return len(tok.encode(w, add_special_tokens=False)) == 1
def tid(w): return tok.encode(w, add_special_tokens=False)[0]
GROUPS = [
 dict(demos=[("red", "fox", "nine"), ("blue", "fox", "three"), ("red", "owl", "seven")], nb="cat", na="gray"),
 dict(demos=[("iron", "gate", "four"), ("iron", "lake", "six"), ("steel", "gate", "two")], nb="pond", na="brass"),
 dict(demos=[("blue", "moon", "eight"), ("blue", "sun", "one"), ("pale", "moon", "five")], nb="star", na="dark"),
]
def Ppay(demos, qa, qb, pid, seed):
    g = np.random.default_rng(seed); order = list(g.permutation(len(demos))); fs = [FILL[i] for i in g.permutation(len(FILL))]
    lines = "".join(f" The password for {demos[i][0]} {demos[i][1]} is {demos[i][2]}." for _ in range(REP) for i in order)
    body = " ".join(fs[:3]) + lines + " " + " ".join(fs[3:6])
    text = body + f" Remember the password for {qa} {qb}:"
    ids = tok.encode(text, add_special_tokens=False)
    return torch.softmax(m(torch.tensor(ids).unsqueeze(0))[0][-1].float(), -1)[pid].item()
def mean(demos, qa, qb, pid): return float(np.mean([Ppay(demos, qa, qb, pid, s) for s in range(NS)]))
print("\n#### GPT-2 v3: password-recall + ambiguous bigram (P of target payload) ####", flush=True)
for G in GROUPS:
    A, B, pay = G["demos"][0]
    if not all(one(f" {w}") for w in [A, B, pay, G["na"], G["nb"]]):
        print(f"  skip {A}+{B}: multitoken", flush=True); continue
    pid = tid(f" {pay}")
    pair = mean(G["demos"], A, B, pid)
    aN = mean(G["demos"], A, G["nb"], pid)
    bN = mean(G["demos"], G["na"], B, pid)
    print(f"  {A}+{B}->{pay}: pair={pair:.3f}  {A}+NOVEL={aN:.3f}  NOVEL+{B}={bN:.3f}  CONJ={pair>=PAIR and aN<MARG and bN<MARG}", flush=True)
print("\nDONE gpt2_conjunction_screen", flush=True)

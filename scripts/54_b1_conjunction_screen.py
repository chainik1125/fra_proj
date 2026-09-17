"""B1 STEP 0 -- conjunction feasibility screen (Gemma-2-2b base). Forward passes only: no SAE, no FRA.

RESEARCH CONTEXT. Defensive interpretability for the FRA paper with Dmitry Manning-Coe. Benign planted
associations only. See docs/insen/research_context.md and proposals/experiments_to_run/B1_controlled_conjunction.md.

The semantic-filter task tied single-feature steering because it was gated by ONE concept. A genuine
FRA-cell needs the attention to depend on TWO different content features (query-content = A,
key-content = B, A != B). Before spending GPU on interventions, screen whether such a cell even exists
in base Gemma. Two candidate constructions:

  (i) cross-concept semantic induction: an A-query attends back to a B-key and copies the payload.
 (ii) two-token compound key: payload fires on the <A><B> pair, not on either token alone.

Keep a construction if the PAIR fires (P(payload) >= PAIR_THR) while each MARGINAL stays low
(< MARG_THR). Runs quickly on a login node.
"""

import os, sys
import torch, numpy as np
from transformer_lens import HookedTransformer

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
PAIR_THR = float(os.environ.get("PAIR_THR", "0.20"))
MARG_THR = float(os.environ.get("MARG_THR", "0.10"))
NSEED = int(os.environ.get("NSEED", "8"))

model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float16); model.eval()
tok = model.tokenizer
print("[model] gemma-2-2b base", flush=True)

FILLER = ["The weather was mild for the time of year.", "Several people arrived late to the meeting.",
 "The library opened an hour earlier than usual.", "A small dog waited patiently by the door.",
 "Nobody expected the report to take so long.", "The train was quiet on the way back.",
 "She wrote a short note and left it on the desk.", "The garden needed more water after the heat.",
 "Two students argued about the final question.", "The coffee in the office had gone cold.",
 "Most of the chairs were stacked against the wall.", "He forgot his umbrella at the station.",
 "The new schedule starts next Monday.", "A light rain began just after lunch.",
 "The shop on the corner closed for repairs.", "They planned a short walk along the lake."]

def single(w): return len(tok.encode(w, add_special_tokens=False)) == 1
def tid(w): return tok.encode(w, add_special_tokens=False)[0]
def pnext(ids, pid):
    tt = torch.tensor(ids, device=dev).unsqueeze(0)
    return torch.softmax(model(tt)[0][-1].float(), -1)[pid].item()

def build(seed, plant_clause, query_clause, n_before=4, n_mid=4):
    g = np.random.default_rng(seed); fs = [FILLER[i] for i in g.permutation(len(FILLER))]
    text = (" ".join(fs[:n_before]) + " " + plant_clause + " " + " ".join(fs[n_before:n_before + n_mid])
            + " " + query_clause)
    return [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)

def screen(label, plant_fn, probes, payload):
    """probes: dict name -> query_clause. plant_fn(payload)->plant_clause. Reports mean P(payload)."""
    pid = tid(payload)
    print(f"\n=== {label}  (payload='{payload.strip()}') ===", flush=True)
    res = {}
    for name, qclause in probes.items():
        ps = [pnext(build(s, plant_fn(payload), qclause), pid) for s in range(NSEED)]
        res[name] = float(np.mean(ps))
        print(f"  {name:14} P(payload) = {res[name]:.3f}", flush=True)
    return res

# ---------------- (i) cross-concept semantic induction: A-query -> B-key -> payload ----------------
# Plant associates concept B (vault/treasure) with the payload; query carries concept A (guard/captain).
# Pair = A-query with B primed; marginals = A-query with B absent, and an unrelated-query with B primed.
CROSS = [
 # (label, payload, B_plant_word, A_query_word, A_unrelated_query_word)
 ("guard x vault",   " seven", " vault",   " guard",   " baker"),
 ("captain x gold",  " seven", " gold",    " captain", " teacher"),
 ("sentry x treasure"," seven"," treasure"," sentry",  " gardener"),
]
for label, payload, Bword, Aword, Aother in CROSS:
    if not all(single(w) for w in [payload, Bword, Aword, Aother]):
        print(f"skip {label}: not all single-token", flush=True); continue
    plant_fn = lambda pl, Bw=Bword: f"The{Bw} holds{pl}."
    probes = {
        "pair(A|B)":   f"Remember, near the{Aword}:",   # A-query, B primed  -> want HIGH
        "A_only(noB)": f"Remember, near the{Aword}:",   # placeholder, replaced below
        "unrel(x|B)":  f"Remember, near the{Aother}:",  # unrelated query, B primed -> want LOW
    }
    # A_only: same A-query but plant clause has NO B (neutral object) -> must rebuild separately
    pid = tid(payload)
    pair = [pnext(build(s, f"The{Bword} holds{payload}.", f"Remember, near the{Aword}:"), pid) for s in range(NSEED)]
    aonly = [pnext(build(s, f"The box holds{payload}.", f"Remember, near the{Aword}:"), pid) for s in range(NSEED)]
    unrel = [pnext(build(s, f"The{Bword} holds{payload}.", f"Remember, near the{Aother}:"), pid) for s in range(NSEED)]
    print(f"\n=== (i) {label}  payload='{payload.strip()}' ===", flush=True)
    print(f"  pair (A-query,B-primed)  P={np.mean(pair):.3f}   <- want >= {PAIR_THR}", flush=True)
    print(f"  A_only (A-query,no B)    P={np.mean(aonly):.3f}   <- want <  {MARG_THR}", flush=True)
    print(f"  unrel (other-query,B)    P={np.mean(unrel):.3f}   <- want <  {MARG_THR}", flush=True)
    ok = np.mean(pair) >= PAIR_THR and np.mean(aonly) < MARG_THR and np.mean(unrel) < MARG_THR
    print(f"  CELL PRESENT: {ok}", flush=True)

# ---------------- (ii) two-token compound key: <A><B> pair vs each marginal ----------------
COMPOUND = [
 # (label, payload, A_word, B_word)
 ("red+fox",  " nine", " red",  " fox"),
 ("iron+gate"," nine", " iron", " gate"),
 ("blue+moon"," nine", " blue", " moon"),
]
for label, payload, Aw, Bw in COMPOUND:
    if not all(single(w) for w in [payload, Aw, Bw]): print(f"skip {label}: multi-token", flush=True); continue
    pid = tid(payload)
    # plant: "The password is <A><B> <payload>." ; query the compound vs each single token
    plant = f"The password is{Aw}{Bw}{payload}."
    pair = [pnext(build(s, plant, f"Remember the password:{Aw}{Bw}"), pid) for s in range(NSEED)]
    aonly = [pnext(build(s, plant, f"Remember the password:{Aw}"), pid) for s in range(NSEED)]
    bonly = [pnext(build(s, plant, f"Remember the password:{Bw}"), pid) for s in range(NSEED)]
    print(f"\n=== (ii) {label}  payload='{payload.strip()}' ===", flush=True)
    print(f"  pair  (A B query) P={np.mean(pair):.3f}   <- want >= {PAIR_THR}", flush=True)
    print(f"  A_only (A query)  P={np.mean(aonly):.3f}   <- want <  {MARG_THR}", flush=True)
    print(f"  B_only (B query)  P={np.mean(bonly):.3f}   <- want <  {MARG_THR}", flush=True)
    ok = np.mean(pair) >= PAIR_THR and np.mean(aonly) < MARG_THR and np.mean(bonly) < MARG_THR
    print(f"  CONJUNCTIVE: {ok}", flush=True)

print("\nDONE b1_conjunction_screen", flush=True)

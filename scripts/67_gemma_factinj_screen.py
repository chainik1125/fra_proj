"""B1_real STEP 0 -- realistic in-context FACT-INJECTION conjunction screen (gemma-2-2b). Forward passes only.

Moves the confirmed toy conjunction (scripts/56, "password for X Y") to a realistic surface: a short
in-context directory of facts, answered by a natural question. The answer is gated by
(subject x attribute); each subject and each attribute appears with several partners, so neither alone
determines the answer -- only the (subject, attribute) pair does. This is the "more real world" version
Dmitry asked for, and the setting where FRA-QK's edge over directional suppression should be largest
(the answer is a real entity, not a single planted rare token).

We screen whether gemma does this AND: P(target value | target question) high, while the value's
probability under the marginal questions (same subject other attribute; other subject same attribute)
is low. Measured on the FIRST token of the value (handles multi-token entities).
"""
import os, sys
import torch, numpy as np
from transformer_lens import HookedTransformer
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
NS = int(os.environ.get("NSEED", "6")); PAIR = float(os.environ.get("PAIR_THR", "0.20")); MARG = float(os.environ.get("MARG_THR", "0.10"))
model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float16); model.eval(); tok = model.tokenizer
print("[model] gemma-2-2b", flush=True)
def first(w): return tok.encode(w, add_special_tokens=False)[0]  # first token id of the value string

# each fact-set: subjects x attributes -> values, with sharing so the pair is needed.
# attributes phrased as natural relations; the directory lists all pairs, then we ask one question.
FACTSETS = [
 dict(subj=["Orion", "Atlas", "Vega"], rel=[("ships from", "sf"), ("was built in", "bi")],
      vals={("Orion","sf"):" Denver", ("Orion","bi"):" France", ("Atlas","sf"):" Boston",
            ("Atlas","bi"):" Japan", ("Vega","sf"):" Denver", ("Vega","bi"):" France"},
      target=("Orion","sf"), reuse_subj=("Orion","bi"), reuse_rel=("Atlas","sf")),
 dict(subj=["Alice", "Bob", "Carol"], rel=[("works in", "wi"), ("studied", "st")],
      vals={("Alice","wi"):" Boston", ("Alice","st"):" law", ("Bob","wi"):" Denver",
            ("Bob","st"):" art", ("Carol","wi"):" Boston", ("Carol","st"):" law"},
      target=("Alice","wi"), reuse_subj=("Alice","st"), reuse_rel=("Bob","wi")),
 dict(subj=["Nile", "Volga", "Rhine"], rel=[("flows through", "ft"), ("is famous for", "ff")],
      vals={("Nile","ft"):" Egypt", ("Nile","ff"):" cotton", ("Volga","ft"):" Russia",
            ("Volga","ff"):" trade", ("Rhine","ft"):" Egypt", ("Rhine","ff"):" cotton"},
      target=("Nile","ft"), reuse_subj=("Nile","ff"), reuse_rel=("Volga","ft")),
]
REL_TXT = {"sf":"ships from", "bi":"was built in", "wi":"works in", "st":"studied",
           "ft":"flows through", "ff":"is famous for"}
QUESTION = {"sf":"Where does {s} ship from?", "bi":"Where was {s} built?",
            "wi":"Where does {s} work?", "st":"What did {s} study?",
            "ft":"What country does {s} flow through?", "ff":"What is {s} famous for?"}

def run(F, seed):
    g = np.random.default_rng(seed)
    pairs = [(s, r[1]) for s in F["subj"] for r in F["rel"]]
    order = list(g.permutation(len(pairs)))
    lines = [f"{pairs[i][0]} {REL_TXT[pairs[i][1]]}{F['vals'][pairs[i]]}." for i in order]
    body = "Notes. " + " ".join(lines) + " "
    def pval(qpair, valstr):
        s, r = qpair; q = QUESTION[r].format(s=s)
        text = body + "Question: " + q + " Answer:"
        ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
        lg = model(torch.tensor(ids, device=dev).unsqueeze(0))[0][-1].float()
        return torch.softmax(lg, -1)[first(valstr)].item()
    tv = F["vals"][F["target"]]  # target value string
    p_t = pval(F["target"], tv)         # target Q -> target value: want HIGH
    p_rs = pval(F["reuse_subj"], tv)     # same subj, other attr -> P(target value): want LOW
    p_rr = pval(F["reuse_rel"], tv)      # other subj, same attr -> P(target value): want LOW
    return p_t, p_rs, p_rr

print("\n#### fact-injection conjunction: P(target value's first token) ####", flush=True)
print("  target Q should be HIGH; marginal Qs (same-subj-other-attr, other-subj-same-attr) LOW", flush=True)
for F in FACTSETS:
    res = np.array([run(F, s) for s in range(NS)]); mt, mrs, mrr = res.mean(0)
    tgt = f"{F['target'][0]}x{F['target'][1]}->{F['vals'][F['target']].strip()}"
    print(f"  {tgt:22} target={mt:.3f}  reuse_subj={mrs:.3f}  reuse_rel={mrr:.3f}  CONJ={mt>=PAIR and mrs<MARG and mrr<MARG}", flush=True)
print("\nDONE gemma_factinj_screen", flush=True)

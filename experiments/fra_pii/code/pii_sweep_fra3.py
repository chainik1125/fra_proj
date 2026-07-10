"""pii_sweep_fra3.py — close the mechanism question: redundant-bank vs resident, + selectivity.

Round 2 found: 'digit' (answer->ssn-digit-token attn) is the only load-bearing keydef among
{digit, name, brace} (name/brace oracle ceilings ~0.0002 / ~0.00007, refuted). But the FRA cut
on the digit edge (top-8-of-12 causal heads) gets ~13.4% target suppression with sib_ok=FALSE
on EVERY row: the SAME head-set, applied to the SIBLING's own digit positions in her own emit
prompt, ALSO suppresses HER emission. That is the generic-structural-edge signature ("X is " ->
nearby digit-shaped tokens), not a content-addressed one, and it already argues against a
selective operating point existing.

This script (NO SAE, pure attn-score oracle, cheap) settles it with a K-scan:
  - does the digit-keyed oracle ceiling climb toward ~1.0 as K (head count, ranked by solo
    causal drop) grows through 12/20/30/45/60/90/130/175/208 (REDUNDANT-BANK: distributed but
    attention-routed, cf STOCKTAKE T1) or plateau low (RESIDENT: value lives in the answer-
    position residual, not attention-fetched, needs a resid-patch follow-up)?
  - at EVERY K, does the SAME top-K head-set (chosen from the TARGET's causal ranking) ALSO
    suppress the SIBLING's own emission when applied to her own digit positions? If sib-supp
    tracks target-supp at every K -> no selective subset exists at the ORACLE level, and since
    FRA==oracle at saturation (round 1), FRA cannot do better either: clean negative. If some K
    has high target-supp with low sib-supp, that K is worth an FRA follow-up.
"""
import os, sys, json
sys.path.insert(0, "/workspace/code")
import torch
from transformer_lens import HookedTransformer

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
OUTDIR = os.environ.get("OUTDIR", os.environ.get("OUT_DIR", "."))
os.makedirs(OUTDIR, exist_ok=True)
TAG = os.environ.get("SWEEP_TAG", "fra3")

model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float16)
model.eval(); tok = model.tokenizer
NL, NH = model.cfg.n_layers, model.cfg.n_heads


def db_fixed():
    return [
        {"name": "Marcus Webb",  "ssn": "531-42-8817", "city": "Portland"},
        {"name": "Elena Ortiz",  "ssn": "624-19-5530", "city": "Denver"},
        {"name": "Priya Nair",   "ssn": "708-63-2291", "city": "Bristol"},
    ]

def db_text(recs): return json.dumps(recs, indent=2)
def emit_prompt(recs, name): return db_text(recs) + f"\n\nThe SSN of {name} is "
def toks(s): return tok(s, return_tensors="pt").to(dev).input_ids
def _dig(s): return "".join(c for c in s if c.isdigit())
def emit_hit(cont, ssn): return _dig(ssn) in _dig(cont)

def find_subseq(hay, needle):
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle: return i
    return -1

def p_tok(t, tid, hooks=()):
    lg = model.run_with_hooks(t, fwd_hooks=list(hooks)) if hooks else model(t)
    return torch.softmax(lg[0, -1].float(), -1)[tid].item()

def gen_cont(t, hooks=(), n=14):
    cur = t
    for _ in range(n):
        lg = model.run_with_hooks(cur, fwd_hooks=list(hooks)) if hooks else model(cur)
        cur = torch.cat([cur, torch.tensor([[int(lg[0, -1].argmax())]], device=dev)], 1)
    return tok.decode(cur[0, t.shape[1]:], skip_special_tokens=True)

def oracle_hooks(heads, qpos, kpositions):
    byL = {}
    for L, H in heads: byL.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in byL.items():
        def mk(Hs):
            def hook(s, hook):
                for H in Hs:
                    for kp in kpositions:
                        if kp < s.shape[3] and qpos < s.shape[2]: s[0, H, qpos, kp] = -1e4
                        if s.shape[2] == 1:
                            for kp in kpositions:
                                if kp < s.shape[3]: s[0, H, 0, kp] = -1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
    return hooks

recs = db_fixed(); tgt, sib = recs[0], recs[1]
te = toks(emit_prompt(recs, tgt["name"])); ide = te[0].tolist(); sq = te.shape[1]
ssn_ids = tok(tgt["ssn"], add_special_tokens=False).input_ids
kpos = find_subseq(ide, ssn_ids); tgt_kps = list(range(kpos, kpos + len(ssn_ids)))
first_id = ssn_ids[0]; qpos = sq - 1
base_emit = p_tok(te, first_id)

tsib = toks(emit_prompt(recs, sib["name"])); idsib = tsib[0].tolist()
sib_ssn_ids = tok(sib["ssn"], add_special_tokens=False).input_ids
skpos = find_subseq(idsib, sib_ssn_ids); sib_kps = list(range(skpos, skpos + len(sib_ssn_ids)))
sib_first_id = sib_ssn_ids[0]; sib_qpos = tsib.shape[1] - 1
base_sib_emit = p_tok(tsib, sib_first_id)
print(f"[base] emit={base_emit:.3f} sib_emit={base_sib_emit:.3f} sq={sq} sib_sq={tsib.shape[1]}", flush=True)

# full single-head causal ranking on the digit keydef (round 2 only kept top-12; recompute in full)
drops = []
for L in range(NL):
    for H in range(NH):
        p = p_tok(te, first_id, oracle_hooks([(L, H)], qpos, tgt_kps))
        drops.append(((L, H), base_emit - p))
drops.sort(key=lambda x: -x[1])
print(f"[fra3] top12 single-head drops: {[(f'L{L}H{H}', round(d,4)) for (L,H),d in drops[:12]]}", flush=True)

Ks = sorted(set([6, 12, 20, 30, 45, 60, 90, 130, 175, NL * NH]))
rows = []
def dump():
    with open(os.path.join(OUTDIR, f"pii_sweep_{TAG}.json"), "w") as fh:
        json.dump(dict(method="fra3", base_emit=base_emit, base_sib_emit=base_sib_emit,
                       full_ranking=[(f"L{L}H{H}", d) for (L, H), d in drops], rows=rows),
                  fh, indent=2, default=float)

for K in Ks:
    heads = [h for h, _ in drops[:K]]
    p_t = p_tok(te, first_id, oracle_hooks(heads, qpos, tgt_kps))
    p_s = p_tok(tsib, sib_first_id, oracle_hooks(heads, sib_qpos, sib_kps))
    supp_t = 1 - p_t / max(base_emit, 1e-6)
    supp_s = 1 - p_s / max(base_sib_emit, 1e-6)
    row = dict(K=K, emit_p=p_t, emit_supp=supp_t, sib_emit_p=p_s, sib_supp=supp_s)
    print(f"[fra3] K={K}: target_supp={supp_t:.3f} sib_supp={supp_s:.3f} "
          f"(sib_supp~0 = selective; sib_supp~target_supp = generic)", flush=True)
    rows.append(row)
    dump()

# text-level emit_ok/sib_ok ground truth at the larger K's (where target_supp gets substantial)
for K in [k for k in Ks if k >= 30]:
    heads = [h for h, _ in drops[:K]]
    cont_t = gen_cont(te, oracle_hooks(heads, qpos, tgt_kps))
    cont_s = gen_cont(tsib, oracle_hooks(heads, sib_qpos, sib_kps))
    eok = emit_hit(cont_t, tgt["ssn"]); sok = emit_hit(cont_s, sib["ssn"])
    print(f"[fra3] K={K} TEXT-LEVEL: emit_ok={eok} sib_ok={sok} cont_t='{cont_t[:20]}' cont_s='{cont_s[:20]}'", flush=True)
    for row in rows:
        if row["K"] == K: row["emit_ok"] = eok; row["sib_ok"] = sok
    dump()

print("[pii_sweep_fra3] DONE", flush=True)

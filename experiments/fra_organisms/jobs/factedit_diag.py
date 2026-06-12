"""FACTEDIT DIAGNOSTIC — why was the oracle edge-cut ~0 on recalled facts?

Resolves the confound in factedit_precheck (run 20260612-061339): gate (i) failed but the *oracle
edge-cut* (cut the (last x subject) score entirely at the top heads) was ALSO ~0 on the recalled set,
which contradicts the prior probe's R=+0.86 on "capital of France"->Paris. So gate (i) was uninformative:
either (A) wrong key/head localization, or (B) genuine saturation/redundancy on STRONGLY-recalled facts
(the prior probe fact had P(ans)=0.039 — WEAKLY recalled — where the edge WAS load-bearing).

This job, model gemma-2-2b-it, residual gemma-scope SAE (FRA basis):
 S0 SANITY: reproduce the prior probe LBNR R on "The capital of France is"->" Paris" with PER-FACT
    head-find (cut the (last x ' France') edge across the top-k causal heads). Confirms machinery.
 S1 PER-FACT BEST EDGE-CUT, by recall band. For facts in 3 bands of baseline recall
    (lo: 0.03-0.15, mid: 0.15-0.40, hi: 0.60-0.99), per fact:
      - locate last-subject-token key Ksub and relation-suffix-token key Krel (the " is"/suffix tok)
      - run full causal head-find (P(target) drop from single-head edge-cut) on BOTH Ksub and Krel
      - report the BEST per-fact oracle edge-cut R = 1 - P_cut/P_base across {top1, top3} heads x {Ksub,Krel}
    => If even the BEST-localized edge-cut is ~0 in the HI band but large in the LO band, the result is
       SATURATION/REDUNDANCY on recalled facts (a sharp, real bound: the load-bearing-edge premise that
       the cheap probe established does NOT hold for facts the model actually knows). If the LO/MID bands
       also ~0, the localization itself is wrong (machinery bug) — S0 adjudicates that.
Ground-truth (logit) only. ckpt() after every fact.
"""
import os, sys, json, time, traceback
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
OUT = os.environ.get("OUTDIR", "."); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE

MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-2-2b-it")
N_PER_BAND = int(os.environ.get("N_PER_BAND", "12"))
N_CAND = int(os.environ.get("N_CAND", "1500"))
CKPT = os.path.join(OUT, "diag.json")
def log(*a): print(*a, flush=True)

log(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_NAME}")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=dev, dtype=torch.float16); model.eval()
tok = model.tokenizer; NL = model.cfg.n_layers; NH = model.cfg.n_heads
log(f"  {NL} layers x {NH} heads")

import pandas as pd
from huggingface_hub import hf_hub_download
df = pd.read_parquet(hf_hub_download("NeelNanda/counterfact-tracing",
      "data/train-00000-of-00001-36693f2cad948c42.parquet", repo_type="dataset"))

def tgt0(t):
    ids = tok.encode(t, add_special_tokens=False); return ids[0] if ids else None
def toks(prompt):
    ids = tok.encode(prompt); return ids, torch.tensor(ids, device=dev).unsqueeze(0)
def keypos(ids, s, q):
    sids = tok.encode(s, add_special_tokens=False)
    if not sids: return None
    last = sids[-1]; c = [i for i, x in enumerate(ids) if x == last and i < q]
    return c[-1] if c else None
def relsuffix_pos(ids, prompt, subject, q):
    # the relation-suffix token = the token right before the last position is usually " is"/suffix;
    # use the position immediately preceding q that is NOT inside the subject span.
    ksub = keypos(ids, subject, q)
    # prefer the token at q-1 (the relation word feeding the extraction), distinct from subject
    p = q - 1
    if ksub is not None and p == ksub: p = q - 2
    return p if p is not None and p >= 0 else None
def cut_edge(tt, heads, q, k):
    byL = {}
    for L, H in heads: byL.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in byL.items():
        def mk(Hs):
            def hook(s, hook):
                for H in Hs: s[0, H, q, k] = -1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
def pt(tt, q, tid): return torch.softmax(model(tt)[0][q].float(), -1)[tid].item()
def causal_topk(tt, q, k, tid, topk=3):
    base = pt(tt, q, tid); dd = {}
    for L in range(NL):
        for H in range(NH):
            pc = torch.softmax(cut_edge(tt, [(L, H)], q, k)[q].float(), -1)[tid].item()
            dd[(L, H)] = base - pc
    top = sorted(dd, key=lambda x: -dd[x])
    return top, dd, base
def edgecut_R(tt, q, k, tid, heads):
    base = pt(tt, q, tid)
    if base <= 0: return float("nan")
    pc = torch.softmax(cut_edge(tt, heads, q, k)[q].float(), -1)[tid].item()
    return 1 - pc / base

state = {"config": {"model": MODEL_NAME}, "sanity": None, "bands": {}}
if os.path.exists(CKPT):
    try: state = json.load(open(CKPT))
    except Exception: pass
def ckpt(): json.dump(state, open(CKPT, "w"), indent=2, default=float)

# ---- S0 SANITY: reproduce prior probe ----
if state.get("sanity") is None:
    log(f"[{time.strftime('%H:%M:%S')}] S0 sanity: 'The capital of France is'->' Paris'")
    ids, tt = toks("The capital of France is"); q = tt.shape[1] - 1
    k = keypos(ids, " France", q); tid = tgt0(" Paris")
    base = pt(tt, q, tid)
    top, dd, _ = causal_topk(tt, q, k, tid, 3)
    R1 = edgecut_R(tt, q, k, tid, top[:1]); R3 = edgecut_R(tt, q, k, tid, top[:3])
    state["sanity"] = dict(base=float(base), top_heads=[list(h) for h in top[:5]],
                           top_effects=[float(dd[h]) for h in top[:5]], R_top1=float(R1), R_top3=float(R3),
                           kpos=int(k))
    ckpt()
    log(f"  base P(Paris)={base:.3f}  top heads={state['sanity']['top_heads'][:3]}  R@top1={R1:+.2f} R@top3={R3:+.2f}")

# ---- build candidate pool with recall, sorted into bands ----
BANDS = {"lo": (0.03, 0.15), "mid": (0.15, 0.40), "hi": (0.60, 0.999)}
pool = {b: [] for b in BANDS}
need = {b: N_PER_BAND for b in BANDS}
log(f"[{time.strftime('%H:%M:%S')}] scanning recall for bands {BANDS}")
df_s = df.sample(n=min(N_CAND, len(df)), random_state=3).reset_index(drop=True)
scanned = 0
for _, row in df_s.iterrows():
    if all(len(pool[b]) >= need[b] for b in BANDS): break
    prompt = str(row["prompt"]); subject = str(row["subject"]); tgt = str(row["target_true"])
    tid = tgt0(tgt)
    if tid is None: continue
    ids, tt = toks(prompt)
    if tt.shape[1] < 3 or tt.shape[1] > 48: continue
    q = tt.shape[1] - 1
    if keypos(ids, subject, q) is None: continue
    p = pt(tt, q, tid); scanned += 1
    for b, (lo, hi) in BANDS.items():
        if lo <= p < hi and len(pool[b]) < need[b]:
            pool[b].append(dict(prompt=prompt, subject=subject, target_true=tgt, tid=int(tid),
                                p_base=float(p), relation_id=str(row["relation_id"]))); break
log(f"  scanned {scanned}; band sizes: " + ", ".join(f"{b}={len(pool[b])}" for b in BANDS))

# ---- S1 per-fact BEST edge-cut by band ----
for b in BANDS:
    if b not in state["bands"]: state["bands"][b] = []
    done = set(r["prompt"] for r in state["bands"][b])
    for fi, fact in enumerate(pool[b]):
        if fact["prompt"] in done: continue
        try:
            ids, tt = toks(fact["prompt"]); q = tt.shape[1] - 1; tid = fact["tid"]
            ksub = keypos(ids, fact["subject"], q); krel = relsuffix_pos(ids, fact["prompt"], fact["subject"], q)
            res = {"prompt": fact["prompt"], "p_base": fact["p_base"], "relation_id": fact["relation_id"]}
            best = 0.0
            for kname, k in [("sub", ksub), ("rel", krel)]:
                if k is None: continue
                top, dd, base = causal_topk(tt, q, k, tid, 3)
                R1 = edgecut_R(tt, q, k, tid, top[:1]); R3 = edgecut_R(tt, q, k, tid, top[:3])
                res[f"R1_{kname}"] = float(R1); res[f"R3_{kname}"] = float(R3)
                res[f"heads_{kname}"] = [list(h) for h in top[:3]]
                res[f"eff_{kname}"] = [float(dd[h]) for h in top[:3]]
                best = max(best, R3, R1)
            res["best_R"] = float(best)
            state["bands"][b].append(res); ckpt()
            log(f"  [{b} {fi:2d}] p={fact['p_base']:.2f} '{fact['prompt'][:34]:34s}' "
                f"R3_sub={res.get('R3_sub')} R3_rel={res.get('R3_rel')} best={best:.2f}")
        except Exception as e:
            log(f"  [{b} {fi}] ERR {e}"); state.setdefault("errors", []).append(str(e)[:160]); ckpt()

# ---- aggregate ----
def summ(rows):
    bb = [r["best_R"] for r in rows if r.get("best_R") is not None]
    rs = [r.get("R3_sub") for r in rows if r.get("R3_sub") is not None]
    rr = [r.get("R3_rel") for r in rows if r.get("R3_rel") is not None]
    f = lambda xs: (float(np.median(xs)), float(np.mean(xs)), float(np.max(xs))) if xs else (None, None, None)
    return dict(n=len(rows), best_R_med_mean_max=f(bb), R3sub=f(rs), R3rel=f(rr),
                frac_best_ge50=float(np.mean([x >= 0.5 for x in bb])) if bb else None)
agg = {b: summ(state["bands"][b]) for b in BANDS}
state["aggregate"] = agg; ckpt()
log("\n================ FACTEDIT DIAGNOSTIC SUMMARY ================")
log(f"  S0 sanity (capital of France->Paris): base={state['sanity']['base']:.3f} "
    f"R@top1={state['sanity']['R_top1']:+.2f} R@top3={state['sanity']['R_top3']:+.2f} heads={state['sanity']['top_heads'][:3]}")
for b in BANDS:
    a = agg[b]
    log(f"  band {b:3s} ({BANDS[b]}) n={a['n']}: best edge-cut R (med,mean,max)={tuple(round(x,3) if x is not None else None for x in a['best_R_med_mean_max'])} "
        f"frac(best>=0.5)={a['frac_best_ge50']}")
log("INTERPRETATION: if best-R is large in lo/mid but ~0 in hi -> SATURATION/REDUNDANCY on recalled facts.")
log("                if best-R ~0 across ALL bands -> localization machinery wrong (check S0).")
json.dump(state, open(os.path.join(OUT, "diag_summary.json"), "w"), indent=2, default=float)
log("DONE factedit_diag")

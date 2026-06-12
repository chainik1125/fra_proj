"""FACTEDIT COMPLETE — the flagship's FAIR complete test (red-team-corrected).

Fixes the two red-team findings against the premature NO-GO (run 20260612-061339):
  (1) WRONG CELL: the pre-check cut only the LAST subject token's edge. The load-bearing attention is
      the FULL SUBJECT TOKEN-SPAN -> last (non-last subject tokens, distributed). The all-heads
      full-span oracle removes the fact on a majority of strongly-recalled facts. So a cuttable
      attention pathway EXISTS; the edit just wasn't aimed at it.
  (2) SELECTION BIAS: the pre-check sorted its 28 edit facts by -p_base + capped -> ALL had P>=0.71
      (saturated). The diag shows the edge IS load-bearing in P~[0.3,0.55] (oracle R 0.21-0.52). The
      FRA cell-edit was NEVER tested where the edge is load-bearing.

THE TEST (gemma-2-2b-it + gemma-scope-2b-pt-res-canonical, judge-free logit ground truth):
  - NEW edit set: bucket-balanced sampling from the LOAD-BEARING band P(target) in [0.30, 0.55]
    (NO -p_base sort), each with a same-relation/different-subject recalled partner. ~20 facts.
  - FULL-SUBJECT-SPAN x relation key, aggregated over the TOP-8 extraction heads (heads ranked by the
    full-span oracle effect on an anchor fact).
  - FRA SAE feature-pair cell-edit (FAITHFUL): for each head and each subject-span key position,
    FRA-decompose the (last x k) edge, take top-M support pairs, accumulate the score-delta over the
    whole span; subtract c*delta at hook_attn_scores, c in {1,2,4,8}. NOT brute column-zeroing.
  - REPORT BOTH metrics:
      (i)  relative P(target) drop  = 1 - P_edit/P_base
      (ii) argmax RANK-FLIP         = did the top-1 next-token argmax change away from the target token?
           (the pure-P metric is biased to call sticky near-1.0 facts unmovable.)
    plus held-out same-relation/different-subject collateral (cross-apply A's support to B), and the
    per-fact FULL-SPAN ORACLE R (headroom = the ceiling the cell-edit is trying to reach).
  GO iff: median cell-edit drop >= 50% (OR >= oracle R at that fact) on a majority AND rank-flip on a
          majority AND collateral < 15%.  -> FRA surgically reaches the load-bearing edge -> flagship is a
          TRUE false-negative -> proceed to ROME/MEMIT selectivity test.
  NO-GO iff the cell-edit still can't move load-bearing-band facts even where oracle headroom exists
          (the SAE feature-pair edit can't reach the distributed full-span edge) -> operational NO-GO airtight.

Resume-proof: ckpt() writes complete.json after every fact.
"""
import os, sys, json, time, traceback
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
OUT = os.environ.get("OUTDIR", "."); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result

MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-2-2b-it")
BAND_LO = float(os.environ.get("BAND_LO", "0.30"))
BAND_HI = float(os.environ.get("BAND_HI", "0.55"))
N_TARGET = int(os.environ.get("N_TARGET", "20"))
N_CAND = int(os.environ.get("N_CAND", "2500"))
TOPH = int(os.environ.get("TOPH", "8"))         # top-8 extraction heads
M_PAIRS = int(os.environ.get("M_PAIRS", "16"))  # top SAE feature-pairs per (head, key-position) edge
CS = [1.0, 2.0, 4.0, 8.0]
CKPT = os.path.join(OUT, "complete.json")
def log(*a): print(*a, flush=True)

log(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_NAME}")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=dev, dtype=torch.float16); model.eval()
tok = model.tokenizer; NL = model.cfg.n_layers; NH = model.cfg.n_heads
log(f"  {NL} layers x {NH} heads")

import pandas as pd
from huggingface_hub import hf_hub_download
df = pd.read_parquet(hf_hub_download("NeelNanda/counterfact-tracing",
      "data/train-00000-of-00001-36693f2cad948c42.parquet", repo_type="dataset"))

_SAE = {}
def get_sae(L):
    if L not in _SAE:
        sl = L - 1
        try:
            _SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical", f"layer_{sl}/width_16k/canonical",
                                    device=dev, normalize_activations=True)
        except Exception:
            _SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res", f"layer_{sl}/width_16k/average_l0_68",
                                    device=dev, normalize_activations=True)
        log(f"    SAE layer {sl} loaded")
    return _SAE[L]

def tgt0(t):
    ids = tok.encode(t, add_special_tokens=False); return ids[0] if ids else None
def toks(prompt):
    ids = tok.encode(prompt); return ids, torch.tensor(ids, device=dev).unsqueeze(0)
def pt(tt, q, tid): return torch.softmax(model(tt)[0][q].float(), -1)[tid].item()
def argmax_tok(tt, q): return int(model(tt)[0][q].float().argmax().item())

def subject_span(ids, subject, q):
    """ALL token positions inside the prompt covered by the subject string (the full-span key).
    Robust to leading-space tokenization: try the subject as-is and space-prefixed; match the
    contiguous id-subsequence; return all covered positions < q."""
    for cand in (subject, " " + subject.strip(), subject.strip()):
        sids = tok.encode(cand, add_special_tokens=False)
        if not sids:
            continue
        n = len(sids)
        for i in range(0, q - n + 1):
            if ids[i:i + n] == sids:
                return list(range(i, i + n))
    # fallback: last subject token only
    sids = tok.encode(subject, add_special_tokens=False)
    if sids:
        last = sids[-1]; c = [i for i, x in enumerate(ids) if x == last and i < q]
        if c: return [c[-1]]
    return None

def cut_span(tt, heads, q, kpositions):
    """oracle: set scores (q, k)=-1e4 for ALL k in the subject span, across heads."""
    byL = {}
    for L, H in heads: byL.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in byL.items():
        def mk(Hs):
            def hook(s, hook):
                for H in Hs:
                    for k in kpositions:
                        s[0, H, q, k] = -1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

def span_oracle_R(tt, q, kpositions, tid, heads):
    base = pt(tt, q, tid)
    if base <= 0: return float("nan"), base
    pc = torch.softmax(cut_span(tt, heads, q, kpositions)[q].float(), -1)[tid].item()
    return 1 - pc / base, base

def causal_topk_span(tt, q, kpositions, tid, topk):
    """rank heads by FULL-SPAN oracle edge-cut effect on P(target)."""
    base = pt(tt, q, tid); dd = {}
    for L in range(NL):
        for H in range(NH):
            pc = torch.softmax(cut_span(tt, [(L, H)], q, kpositions)[q].float(), -1)[tid].item()
            dd[(L, H)] = base - pc
    top = sorted(dd, key=lambda x: -dd[x])[:topk]
    return top, {f"L{L}H{H}": float(dd[(L, H)]) for (L, H) in top}, base

def fra_edge(L, H, feats, Wdec, xh, tt):
    r = _build_fra_result(model, L, H, feats, Wdec, dev, top_k=None,
                          rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
    f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
    return dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())

def top_pairs(d, q, k, M):
    loc = np.where((d["qq"] == q) & (d["kk"] == k))[0]
    loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
    return set((int(d["ii"][o]), int(d["jj"][o])) for o in loc)

def accumulate_delta(d, pairsByK, seq):
    """[seq,seq] score-delta carried by the support feature-pairs, summed over the whole subject span
    (pairsByK maps key-position -> set of support (i,j) pairs at that key)."""
    dd = np.zeros((seq, seq))
    for n in range(len(d["vv"])):
        k = int(d["kk"][n])
        if k in pairsByK and (int(d["ii"][n]), int(d["jj"][n])) in pairsByK[k]:
            dd[int(d["qq"][n]), k] += d["vv"][n]
    return dd

def apply_cell_edit(tt, byL, c):
    seq = tt.shape[1]; hooks = []
    for L, hd in byL.items():
        td = {H: torch.tensor(dd, device=dev, dtype=torch.float32) * c for H, dd in hd.items()}
        def mk(td):
            def hook(s, hook):
                for H, sd in td.items():
                    s[0, H, :seq, :seq] = s[0, H, :seq, :seq] - sd.to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

def build_fullspan_support(ids, tt, subject, q, heads):
    """FRA cell-edit support over the FULL subject span x relation, per head.
    Returns (byL{L:{H:[seq,seq]delta}}, span_positions, pairs_per_head{(L,H):{k:pairset}}) or (None,..)."""
    span = subject_span(ids, subject, q)
    if not span:
        return None, None, None
    seq = tt.shape[1]
    layers = sorted(set(L for L, H in heads))
    fc = {}
    for L in layers:
        hk = f"blocks.{L}.hook_resid_pre"
        a = model.run_with_cache(tt, names_filter=[hk])[1][hk][0].float()
        sae = get_sae(L); f = sae.encode(a).float()
        if sae._norm_coeff is not None: f = f / sae._norm_coeff
        xh = f @ sae.W_dec.float() + sae.b_dec.float()
        fc[L] = (f, xh)
    byL = {}; pairs_per_head = {}
    for (L, H) in heads:
        f, xh = fc[L]
        d = fra_edge(L, H, f, get_sae(L).W_dec.float(), xh, tt)
        pairsByK = {k: top_pairs(d, q, k, M_PAIRS) for k in span}
        pairs_per_head[(L, H)] = pairsByK
        byL.setdefault(L, {})[H] = accumulate_delta(d, pairsByK, seq)
    return byL, span, pairs_per_head

# ---- resume -------------------------------------------------------------
state = {"config": {"model": MODEL_NAME, "band": [BAND_LO, BAND_HI], "topH": TOPH, "M_pairs": M_PAIRS,
                    "edit_c": CS, "sae": "gemma-scope-2b-pt-res-canonical",
                    "key": "FULL-SUBJECT-SPAN x relation"},
         "heads": None, "head_effects": None, "facts": []}
if os.path.exists(CKPT):
    try:
        state = json.load(open(CKPT)); log(f"  resumed: {len(state['facts'])} facts, heads={state.get('heads')}")
    except Exception: pass
done = set(f["prompt"] for f in state["facts"])
def ckpt(): json.dump(state, open(CKPT, "w"), indent=2, default=float)

# ---- STAGE 1: bucket-balanced sampling in the load-bearing band ----------
log(f"[{time.strftime('%H:%M:%S')}] STAGE 1 sample P(target) in [{BAND_LO},{BAND_HI}] (bucket-balanced by relation, NO -p_base sort)")
df_s = df.sample(n=min(N_CAND, len(df)), random_state=7).reset_index(drop=True)
in_band = []; by_rel = {}
for _, row in df_s.iterrows():
    if len(in_band) >= N_CAND: break
    prompt = str(row["prompt"]); subject = str(row["subject"]); tgt = str(row["target_true"])
    tid = tgt0(tgt)
    if tid is None: continue
    ids, tt = toks(prompt)
    if tt.shape[1] < 3 or tt.shape[1] > 48: continue
    q = tt.shape[1] - 1
    if not subject_span(ids, subject, q): continue
    p = pt(tt, q, tid)
    if BAND_LO <= p < BAND_HI:
        rec = dict(prompt=prompt, subject=subject, target_true=tgt, tid=int(tid),
                   p_base=float(p), relation_id=str(row["relation_id"]))
        in_band.append(rec); by_rel.setdefault(rec["relation_id"], []).append(rec)
log(f"  in-band facts: {len(in_band)} across {len(by_rel)} relations")

# build edit set: facts that have a same-relation/different-subject partner ALSO in band; bucket-balanced
# across relations (round-robin, NOT p_base-sorted) so we don't reintroduce selection bias.
edit_facts = []
rels = sorted(by_rel.keys())
ri = 0
guard = 0
while len(edit_facts) < N_TARGET and guard < 100000:
    guard += 1
    rel = rels[ri % len(rels)]; ri += 1
    facts = by_rel.get(rel, [])
    if len(facts) < 2: continue
    f = facts[0]
    partner = next((g for g in facts if g["subject"] != f["subject"]), None)
    if partner is None: continue
    if f["prompt"] in (x["prompt"] for x in edit_facts):
        # advance within this relation
        by_rel[rel] = facts[1:]
        continue
    f2 = dict(f); f2["heldout"] = dict(prompt=partner["prompt"], subject=partner["subject"],
                                       target_true=partner["target_true"], tid=partner["tid"],
                                       p_base=partner["p_base"], relation_id=partner["relation_id"])
    edit_facts.append(f2)
    by_rel[rel] = facts[1:]   # rotate so next visit picks a fresh subject
    if all(len(by_rel.get(r, [])) < 2 for r in rels): break
log(f"  edit set: {len(edit_facts)} facts; p_base = {[round(f['p_base'],2) for f in edit_facts]}")
state["in_band_n"] = len(in_band); state["n_relations"] = len(by_rel); state["edit_set_n"] = len(edit_facts)
ckpt()
if len(edit_facts) < 8:
    state["BLOCKER"] = f"too few in-band facts with a same-relation partner ({len(edit_facts)})"
    ckpt(); log("BLOCKER: " + state["BLOCKER"]); sys.exit(0)

# ---- STAGE 2: top-8 extraction heads by FULL-SPAN oracle (anchor = highest-headroom in-band fact) ----
if state.get("heads") is None:
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 2 full-span causal head-find (top-{TOPH})")
    # pick anchor = the in-band fact with the largest full-span oracle R at top-3 (most load-bearing)
    best_anchor = None; best_R = -1
    for cand in edit_facts[:8]:
        ids, tt = toks(cand["prompt"]); q = tt.shape[1] - 1
        span = subject_span(ids, cand["subject"], q)
        if not span: continue
        top3, _, _ = causal_topk_span(tt, q, span, cand["tid"], 3)
        R3, _ = span_oracle_R(tt, q, span, cand["tid"], top3)
        if R3 > best_R: best_R = R3; best_anchor = cand
    anchor = best_anchor or edit_facts[0]
    ids, tt = toks(anchor["prompt"]); q = tt.shape[1] - 1
    span = subject_span(ids, anchor["subject"], q)
    heads, eff, base = causal_topk_span(tt, q, span, anchor["tid"], TOPH)
    state["heads"] = [list(h) for h in heads]; state["head_effects"] = eff
    state["head_anchor"] = anchor["prompt"]; state["head_anchor_baseR3"] = float(best_R)
    ckpt()
    log(f"  anchor='{anchor['prompt']}' p={anchor['p_base']:.2f} best-R3(span)={best_R:.2f}")
    log(f"  top-{TOPH} full-span extraction heads: {eff}")
heads = [tuple(h) for h in state["heads"]]

# ---- STAGE 3: per-fact full-span FRA cell-edit + two metrics + collateral ----
log(f"[{time.strftime('%H:%M:%S')}] STAGE 3 full-span FRA cell-edit (heads={heads})")
for fi, fact in enumerate(edit_facts):
    if fact["prompt"] in done: continue
    try:
        ids, tt = toks(fact["prompt"]); q = tt.shape[1] - 1; tid = fact["tid"]
        byL, span, A_pairs = build_fullspan_support(ids, tt, fact["subject"], q, heads)
        if byL is None:
            log(f"  [{fi}] SKIP no subject span"); continue
        base = pt(tt, q, tid); base_argmax = argmax_tok(tt, q)
        oracle_R, _ = span_oracle_R(tt, q, span, tid, heads)
        edited = {}
        for c in CS:
            lg = apply_cell_edit(tt, byL, c)[q].float()
            pe = torch.softmax(lg, -1)[tid].item(); am = int(lg.argmax().item())
            edited[str(c)] = dict(p=float(pe), drop=float(1 - pe / base) if base > 0 else float("nan"),
                                  argmax=am, flipped=bool(am != tid))
        # best-c relative drop and whether ANY tested c flips the argmax away from target
        best_drop = max(edited[str(c)]["drop"] for c in CS)
        any_flip = any(edited[str(c)]["flipped"] for c in CS)

        # collateral: cross-apply A's support PAIRS onto the held-out (same-rel/diff-subj) fact B's edge
        ho = fact["heldout"]; ho_ids, ho_tt = toks(ho["prompt"]); ho_q = ho_tt.shape[1] - 1
        ho_base = pt(ho_tt, ho_q, ho["tid"]); ho_argmax = argmax_tok(ho_tt, ho_q)
        cross = {}
        try:
            ho_span = subject_span(ho_ids, ho["subject"], ho_q)
            if ho_span:
                seqB = ho_tt.shape[1]; layers = sorted(set(L for L, H in heads)); fcB = {}
                for L in layers:
                    hk = f"blocks.{L}.hook_resid_pre"
                    aB = model.run_with_cache(ho_tt, names_filter=[hk])[1][hk][0].float()
                    sae = get_sae(L); fB = sae.encode(aB).float()
                    if sae._norm_coeff is not None: fB = fB / sae._norm_coeff
                    xhB = fB @ sae.W_dec.float() + sae.b_dec.float(); fcB[L] = (fB, xhB)
                byL_cross = {}
                for (L, H) in heads:
                    fB, xhB = fcB[L]
                    dB = fra_edge(L, H, fB, get_sae(L).W_dec.float(), xhB, ho_tt)
                    # map A's support pairs (collapsed over A's span) onto B's span key positions
                    A_pairset = set()
                    for k, ps in A_pairs[(L, H)].items(): A_pairset |= ps
                    pairsByK_B = {k: A_pairset for k in ho_span}
                    byL_cross.setdefault(L, {})[H] = accumulate_delta(dB, pairsByK_B, seqB)
                for c in CS:
                    lgc = apply_cell_edit(ho_tt, byL_cross, c)[ho_q].float()
                    pe_c = torch.softmax(lgc, -1)[ho["tid"]].item()
                    cross[f"c{c}"] = dict(p=float(pe_c), drop=float(1 - pe_c / ho_base) if ho_base > 0 else float("nan"),
                                          flipped=bool(int(lgc.argmax().item()) != ho["tid"]))
        except Exception as e:
            cross["error"] = str(e)[:120]

        rec = dict(prompt=fact["prompt"], subject=fact["subject"], target_true=fact["target_true"],
                   relation_id=fact["relation_id"], tid=tid, base=float(base),
                   span_len=len(span), oracle_R_fullspan=float(oracle_R),
                   edited=edited, best_drop=float(best_drop), any_flip=bool(any_flip),
                   meets_oracle=bool(best_drop >= max(0.0, oracle_R) - 1e-6),
                   heldout=dict(prompt=ho["prompt"], base=float(ho_base), cross=cross))
        state["facts"].append(rec); done.add(fact["prompt"]); ckpt()
        cd = cross.get("c2.0", {}).get("drop")
        log(f"  [{fi:2d}] p={base:.2f} span={len(span)} '{fact['prompt'][:34]:34s}' "
            f"oracle={oracle_R:.2f} drop@c1={edited['1.0']['drop']:.2f} @c2={edited['2.0']['drop']:.2f} "
            f"@c4={edited['4.0']['drop']:.2f} @c8={edited['8.0']['drop']:.2f} best={best_drop:.2f} "
            f"flip={any_flip} | collat@c2={cd}")
    except Exception as e:
        log(f"  [{fi}] ERROR {e}\n{traceback.format_exc()[:300]}")
        state.setdefault("errors", []).append(str(e)[:200]); ckpt()

# ---- STAGE 4: aggregate + verdict ---------------------------------------
log(f"[{time.strftime('%H:%M:%S')}] STAGE 4 aggregate")
facts = [f for f in state["facts"] if "edited" in f]
def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else None
def frac(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None

best_drops = [f["best_drop"] for f in facts]
oracle_Rs = [f["oracle_R_fullspan"] for f in facts]
flips = [f["any_flip"] for f in facts]
meets_oracle = [f["meets_oracle"] for f in facts]
# collateral = median best (over c) cross-drop on held-out B
coll_best = []
for f in facts:
    cs = f["heldout"]["cross"]; dd = [cs.get(f"c{c}", {}).get("drop") for c in CS]
    dd = [x for x in dd if x is not None]
    coll_best.append(max(dd) if dd else None)

# per-c median drop (to see if a faithful c reaches 50%)
percdrop = {str(c): med([f["edited"][str(c)]["drop"] for f in facts]) for c in CS}

summary = dict(
    n_facts=len(facts), band=[BAND_LO, BAND_HI],
    heads=state["heads"], head_effects=state.get("head_effects"),
    median_oracle_R_fullspan=med(oracle_Rs),
    median_drop_per_c=percdrop,
    median_best_drop=med(best_drops),
    frac_drop_ge50=frac([d >= 0.5 for d in best_drops]),
    frac_meets_oracle=frac(meets_oracle),
    rank_flip_rate=frac(flips),
    median_collateral_best=med(coll_best),
)
# GO criteria
g_drop = (summary["median_best_drop"] is not None and summary["median_best_drop"] >= 0.50) or \
         (summary["frac_meets_oracle"] is not None and summary["frac_meets_oracle"] > 0.5)
g_flip = (summary["rank_flip_rate"] is not None and summary["rank_flip_rate"] > 0.5)
g_coll = (summary["median_collateral_best"] is not None and abs(summary["median_collateral_best"]) < 0.15)
verdict = "GO" if (g_drop and g_flip and g_coll) else "NO-GO"
summary["gates"] = dict(drop_ge50_or_meets_oracle=bool(g_drop), rank_flip_majority=bool(g_flip),
                        collateral_lt15=bool(g_coll))
summary["VERDICT"] = verdict
state["summary"] = summary; ckpt()

log("\n================ FACTEDIT COMPLETE SUMMARY ================")
log(f"  band P(target) in [{BAND_LO},{BAND_HI}]; edited n={summary['n_facts']}")
log(f"  full-span extraction heads (top-{TOPH}): {summary['heads']}")
log(f"  median FULL-SPAN ORACLE R (headroom): {summary['median_oracle_R_fullspan']}")
log(f"  median FRA cell-edit drop per c: {summary['median_drop_per_c']}")
log(f"  median BEST drop: {summary['median_best_drop']}  frac(best>=50%): {summary['frac_drop_ge50']}  frac(meets oracle): {summary['frac_meets_oracle']}")
log(f"  RANK-FLIP rate (argmax leaves target): {summary['rank_flip_rate']}")
log(f"  median held-out collateral (best c): {summary['median_collateral_best']}")
log(f"  GATES: drop>=50%|meets-oracle={g_drop}  rank-flip-majority={g_flip}  collateral<15%={g_coll}")
log(f"  >>> VERDICT: {verdict} <<<")
json.dump(summary, open(os.path.join(OUT, "complete_summary.json"), "w"), indent=2, default=float)
log("DONE factedit_complete")

"""FACTUAL-RECALL EDIT — cheap GO/NO-GO pre-check (SCREEN.md proposal 1).

Model: gemma-2-2b-it + gemma-scope-2b-pt-res-canonical (residual SAE; the FRA QK
decomposition REQUIRES a residual SAE whose decoder lives in d_model space so it can
be projected through W_Q/W_K — an attention/hook_z SAE cannot drive a score-cell edit).

Pipeline (ground-truth only, NO judge):
 1. RECALL FILTER: from CounterFact (NeelNanda/counterfact-tracing) keep ~25-30 facts the
    model recalls — P(target_true first token | prompt, last pos) >= REC_THRESH. For each kept
    fact, also keep a HELD-OUT partner: a DIFFERENT subject, SAME relation, also recalled.
 2. CAUSAL HEAD-FIND: rank heads by edge-cut effect on P(target) of the (last-pos query x
    subject-token key) extraction edge; take the top <=3 (>=60% of effect target). (g_screen.causal_heads.)
 3. FRA CELL-EDIT: at those heads, FRA-decompose the (last x subject) edge in the SAE feature
    basis (subject-content query feat x relation-content key feat), take the top-M support pairs,
    build a score-cell delta, subtract c*delta at hook_attn_scores. (j13/j12 pattern.) Measure
    P(target) drop at c in {1,2}.
 4. GATES:
    (i)  cell-edit drops P(target) by >= 50% at c~=1-2 on the EDITED fact   (resolves CCF=0 reach worry)
    (ii) the SAME edit (re-derived on the held-out fact's own subj x rel edge) drops the held-out
         same-relation/different-subject fact's P(target) by < 15%          (rules out MLP-distributed / D2-fail)
    VERDICT: GO iff (i)>=50% AND (ii)<15% collateral. NO-GO otherwise (pre-registered flagship negative).

Resume-proof: ckpt() writes precheck.json after EVERY fact; on restart we skip facts already in it.
"""
import os, sys, json, time, traceback
sys.path.insert(0, "/workspace/code")
import torch, numpy as np

OUT = os.environ.get("OUTDIR", ".")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result

# ---- config -------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-2-2b-it")
REC_THRESH = float(os.environ.get("REC_THRESH", "0.30"))   # recalled iff P(target0) >= this
N_TARGET   = int(os.environ.get("N_TARGET", "28"))          # number of recalled facts to edit
N_CAND     = int(os.environ.get("N_CAND", "400"))           # candidate facts to score for recall
TOPH       = int(os.environ.get("TOPH", "3"))               # <=3 causal extraction heads
M_PAIRS    = int(os.environ.get("M_PAIRS", "16"))           # top SAE feature-pairs per head edge
CS         = [1.0, 2.0]                                      # edit strengths
CKPT = os.path.join(OUT, "precheck.json")

def log(*a):
    print(*a, flush=True)

# ---- load model ---------------------------------------------------------
log(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_NAME}")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=dev, dtype=torch.float16)
model.eval()
tok = model.tokenizer
NL = model.cfg.n_layers
NH = model.cfg.n_heads
log(f"  loaded: {NL} layers x {NH} heads")

# ---- load CounterFact ---------------------------------------------------
import pandas as pd
from huggingface_hub import hf_hub_download
cf_path = hf_hub_download("NeelNanda/counterfact-tracing",
                          "data/train-00000-of-00001-36693f2cad948c42.parquet",
                          repo_type="dataset")
df = pd.read_parquet(cf_path)
log(f"  counterfact rows: {len(df)}")

# ---- SAE cache (one residual SAE per needed layer; SAE on resid_post[L-1] = resid_pre[L]) ----
_SAE = {}
def get_sae(L):
    if L not in _SAE:
        sl = L - 1
        try:
            _SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",
                                    f"layer_{sl}/width_16k/canonical",
                                    device=dev, normalize_activations=True)
        except Exception:
            _SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res",
                                    f"layer_{sl}/width_16k/average_l0_68",
                                    device=dev, normalize_activations=True)
        log(f"    SAE layer {sl} loaded")
    return _SAE[L]

# ---- helpers ------------------------------------------------------------
def encode_layer(L, tt):
    """returns (feats[seq,d_sae], resid[seq,d_model]) at hook_resid_pre[L]."""
    hk = f"blocks.{L}.hook_resid_pre"
    a = model.run_with_cache(tt, names_filter=[hk])[1][hk][0].float()
    sae = get_sae(L)
    f = sae.encode(a).float()
    if sae._norm_coeff is not None:
        f = f / sae._norm_coeff
    return f, a

def target_id(target_true):
    # first token of the (space-prefixed) target object
    ids = tok.encode(target_true, add_special_tokens=False)
    return ids[0] if ids else None

def p_target(tt, qpos, tid):
    return torch.softmax(model(tt)[0][qpos].float(), -1)[tid].item()

def build_tokens(prompt):
    # gemma auto-prepends BOS via tokenizer; keep it plain (the recall is robust to chat wrapping,
    # and the Geva/ROME extraction circuit is characterized on the plain completion form).
    ids = tok.encode(prompt)  # includes BOS for gemma
    tt = torch.tensor(ids, device=dev).unsqueeze(0)
    return ids, tt

def subj_key_pos(ids, subject, qpos):
    """last token position of the subject string inside the prompt (the extraction key)."""
    sids = tok.encode(subject, add_special_tokens=False)
    if not sids:
        return None
    last = sids[-1]
    cands = [i for i, x in enumerate(ids) if x == last and i < qpos]
    return cands[-1] if cands else None

# score-cell cut: set the (qpos,kpos) score to -inf at the given heads -> P(target) drop = causal effect
def cut_edge(tt, heads, qpos, kpos):
    byL = {}
    for L, H in heads:
        byL.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in byL.items():
        def mk(Hs):
            def hook(s, hook):
                for H in Hs:
                    s[0, H, qpos, kpos] = -1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

def causal_heads(tt, qpos, kpos, tid, topk):
    base = torch.softmax(model(tt)[0][qpos].float(), -1)[tid].item()
    dd = {}
    for L in range(NL):
        for H in range(NH):
            pc = torch.softmax(cut_edge(tt, [(L, H)], qpos, kpos)[qpos].float(), -1)[tid].item()
            dd[(L, H)] = base - pc
    top = sorted(dd, key=lambda x: -dd[x])[:topk]
    eff = {f"L{L}H{H}": float(dd[(L, H)]) for (L, H) in top}
    return top, eff, base

# FRA-decompose the (qpos,kpos) edge at one head -> dict of arrays
def fra_edge(L, H, feats, Wdec, xh, tt):
    r = _build_fra_result(model, L, H, feats, Wdec, dev, top_k=None,
                          rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
    f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
    return dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())

def top_support_pairs(d, qpos, kpos, M):
    loc = np.where((d["qq"] == qpos) & (d["kk"] == kpos))[0]
    loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
    return set((int(d["ii"][o]), int(d["jj"][o])) for o in loc)

def cell_delta(d, P, seq):
    """[seq,seq] score-delta carried by the support feature-pairs P (the cell-edit support)."""
    dd = np.zeros((seq, seq))
    for n in range(len(d["vv"])):
        if (int(d["ii"][n]), int(d["jj"][n])) in P:
            dd[d["qq"][n], d["kk"][n]] += d["vv"][n]
    return dd

def apply_cell_edit(tt, byL, c):
    """subtract c * (per-head score-delta) at hook_attn_scores."""
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

def build_edit_support(ids, tt, subject, qpos, heads):
    """For a fact: locate the subject key, FRA-decompose the extraction edge at each head,
    return (byL_support_per_c-unit, kpos) where byL maps L->{H: [seq,seq] delta}. None if no key."""
    kpos = subj_key_pos(ids, subject, qpos)
    if kpos is None:
        return None, None
    seq = tt.shape[1]
    # group heads by layer, encode each needed layer once
    layers = sorted(set(L for L, H in heads))
    feat_cache = {}
    for L in layers:
        f, a = encode_layer(L, tt)
        xh = f @ get_sae(L).W_dec.float() + get_sae(L).b_dec.float()
        feat_cache[L] = (f, xh)
    byL = {}
    for (L, H) in heads:
        f, xh = feat_cache[L]
        d = fra_edge(L, H, f, get_sae(L).W_dec.float(), xh, tt)
        P = top_support_pairs(d, qpos, kpos, M_PAIRS)
        delta = cell_delta(d, P, seq)
        byL.setdefault(L, {})[H] = delta
    return byL, kpos

# ---- resume -------------------------------------------------------------
state = {"config": {"model": MODEL_NAME, "rec_thresh": REC_THRESH, "topH": TOPH,
                     "M_pairs": M_PAIRS, "edit_c": CS, "sae": "gemma-scope-2b-pt-res-canonical"},
         "heads": None, "head_effects": None, "facts": []}
if os.path.exists(CKPT):
    try:
        state = json.load(open(CKPT))
        log(f"  resumed: {len(state['facts'])} facts already done, heads={state.get('heads')}")
    except Exception:
        pass
done_prompts = set(fr["prompt"] for fr in state["facts"])

def ckpt():
    json.dump(state, open(CKPT, "w"), indent=2, default=float)

# ---- STAGE 1: recall filter + held-out partner selection ----------------
# group candidate rows by relation so we can pick same-relation/different-subject partners.
log(f"[{time.strftime('%H:%M:%S')}] STAGE 1 recall-filter (P(target0)>={REC_THRESH})")
# sample candidates spread across relations
df_s = df.sample(n=min(N_CAND * 3, len(df)), random_state=0).reset_index(drop=True)
recalled = []   # list of dicts: prompt, subject, target_true, tid, p, relation_id
by_rel = {}
for _, row in df_s.iterrows():
    if len(recalled) >= N_CAND:
        break
    prompt = str(row["prompt"]); subject = str(row["subject"]); tgt = str(row["target_true"])
    tid = target_id(tgt)
    if tid is None:
        continue
    ids, tt = build_tokens(prompt)
    if tt.shape[1] < 3 or tt.shape[1] > 64:
        continue
    qpos = tt.shape[1] - 1
    p = p_target(tt, qpos, tid)
    if p >= REC_THRESH:
        rec = dict(prompt=prompt, subject=subject, target_true=tgt, tid=int(tid),
                   p_base=float(p), relation_id=str(row["relation_id"]))
        recalled.append(rec)
        by_rel.setdefault(rec["relation_id"], []).append(rec)
log(f"  recalled candidates: {len(recalled)} across {len(by_rel)} relations")

# build edit set: facts that HAVE a same-relation/different-subject recalled partner
edit_facts = []
for rel, facts in by_rel.items():
    if len(facts) < 2:
        continue
    for i, f in enumerate(facts):
        partner = next((g for g in facts if g["subject"] != f["subject"]), None)
        if partner is None:
            continue
        f2 = dict(f); f2["heldout"] = dict(prompt=partner["prompt"], subject=partner["subject"],
                                           target_true=partner["target_true"], tid=partner["tid"],
                                           p_base=partner["p_base"], relation_id=partner["relation_id"])
        edit_facts.append(f2)
        if len([x for x in facts if x is not None]) and len(edit_facts) >= N_TARGET * 2:
            break
# unique by prompt, cap at N_TARGET, prefer higher baseline recall
edit_facts = sorted(edit_facts, key=lambda x: -x["p_base"])
seen = set(); uniq = []
for f in edit_facts:
    if f["prompt"] in seen:
        continue
    seen.add(f["prompt"]); uniq.append(f)
    if len(uniq) >= N_TARGET:
        break
edit_facts = uniq
log(f"  edit facts (with held-out same-rel/diff-subj partner): {len(edit_facts)}")
state["recalled_n"] = len(recalled)
state["n_relations"] = len(by_rel)
state["edit_set_n"] = len(edit_facts)
ckpt()

if len(edit_facts) < 8:
    state["BLOCKER"] = f"too few recalled facts with a same-relation partner ({len(edit_facts)}); lower REC_THRESH or raise N_CAND"
    ckpt()
    log("BLOCKER: " + state["BLOCKER"])
    sys.exit(0)

# ---- STAGE 2: causal head-find (on the highest-recall fact as anchor) ----
if state.get("heads") is None:
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 2 causal head-find")
    anchor = edit_facts[0]
    ids, tt = build_tokens(anchor["prompt"])
    qpos = tt.shape[1] - 1
    kpos = subj_key_pos(ids, anchor["subject"], qpos)
    if kpos is None:
        # fall back to next fact with a locatable subject key
        for cand in edit_facts:
            ids, tt = build_tokens(cand["prompt"]); qpos = tt.shape[1] - 1
            kpos = subj_key_pos(ids, cand["subject"], qpos)
            if kpos is not None:
                anchor = cand; break
    heads, eff, base = causal_heads(tt, qpos, kpos, anchor["tid"], TOPH)
    state["heads"] = [list(h) for h in heads]
    state["head_effects"] = eff
    state["head_anchor"] = anchor["prompt"]
    state["head_anchor_base"] = float(base)
    ckpt()
    log(f"  anchor='{anchor['prompt']}' base P(tgt)={base:.3f}")
    log(f"  top-{TOPH} causal extraction heads: {eff}")
heads = [tuple(h) for h in state["heads"]]

# ---- STAGE 3: per-fact cell-edit + two gates ----------------------------
log(f"[{time.strftime('%H:%M:%S')}] STAGE 3 cell-edit + gates (heads={heads})")
for fi, fact in enumerate(edit_facts):
    if fact["prompt"] in done_prompts:
        continue
    try:
        ids, tt = build_tokens(fact["prompt"]); qpos = tt.shape[1] - 1
        byL, kpos = build_edit_support(ids, tt, fact["subject"], qpos, heads)
        if byL is None:
            log(f"  [{fi}] '{fact['prompt'][:40]}' SKIP: subject key not found")
            continue
        tid = fact["tid"]; base = p_target(tt, qpos, tid)
        # gate (i): P(target) drop at each c on the EDITED fact
        edited = {}
        for c in CS:
            pe = torch.softmax(apply_cell_edit(tt, byL, c)[qpos].float(), -1)[tid].item()
            edited[str(c)] = dict(p=float(pe), drop=float(1 - pe / base) if base > 0 else float("nan"))
        # also the raw edge-cut (oracle ceiling) for reference / LBNR
        p_cut = torch.softmax(cut_edge(tt, heads, qpos, kpos)[qpos].float(), -1)[tid].item()
        R_oracle = float(1 - p_cut / base) if base > 0 else float("nan")

        # gate (ii): collateral on held-out same-rel/diff-subj fact.
        # Re-derive the cell-edit on the HELD-OUT fact's OWN (last x subject) edge (same heads,
        # same M support pairs) and measure ITS P(target) drop. If the edit mechanism is
        # content-specific (FRA), this drop is the on-target effect on the held-out fact;
        # the SCREEN gate-(ii) signature for MLP-distribution is whether applying the
        # edit support derived from fact-A onto fact-B's stream collaterally damages B.
        ho = fact["heldout"]
        ho_ids, ho_tt = build_tokens(ho["prompt"]); ho_q = ho_tt.shape[1] - 1
        ho_base = p_target(ho_tt, ho_q, ho["tid"])
        # (ii-a) transfer test: take fact-A's support PAIRS, rebuild their delta on B's edge geometry
        #        (the same SAE feature-pairs, evaluated on B). If A's subject/relation features don't
        #        fire on B, B is untouched -> ~0 collateral (good). If they DO move B, non-content-specific.
        ho_byL, ho_kpos = build_edit_support(ho_ids, ho_tt, ho["subject"], ho_q, heads)
        collateral = {}
        if ho_byL is not None:
            for c in CS:
                # apply B's OWN re-derived support at strength c -> B's on-target drop (informative: is B editable too)
                pe_b = torch.softmax(apply_cell_edit(ho_tt, ho_byL, c)[ho_q].float(), -1)[ho["tid"]].item()
                collateral[f"ownB_c{c}"] = dict(p=float(pe_b), drop=float(1 - pe_b / ho_base) if ho_base > 0 else float("nan"))
        # (ii-b) cross-apply: fact-A's delta tensors (A's geometry) are seq-shaped to A; to test
        #        cross-fact bleed cleanly we instead apply A's support-PAIR set to B's FRA edge.
        #        Build B's FRA at A's pairs:
        cross = {}
        try:
            layers = sorted(set(L for L, H in heads))
            seqB = ho_tt.shape[1]
            fcacheB = {}
            for L in layers:
                fB, aB = encode_layer(L, ho_tt)
                xhB = fB @ get_sae(L).W_dec.float() + get_sae(L).b_dec.float()
                fcacheB[L] = (fB, xhB)
            # A's support pairs per head
            A_pairs = {}
            ids_a, tt_a = build_tokens(fact["prompt"]); qa = tt_a.shape[1] - 1
            kpa = subj_key_pos(ids_a, fact["subject"], qa)
            for (L, H) in heads:
                fA, xhA = None, None
                fA_full, aA = encode_layer(L, tt_a)
                xhA = fA_full @ get_sae(L).W_dec.float() + get_sae(L).b_dec.float()
                dA = fra_edge(L, H, fA_full, get_sae(L).W_dec.float(), xhA, tt_a)
                A_pairs[(L, H)] = top_support_pairs(dA, qa, kpa, M_PAIRS)
            byL_cross = {}
            if ho_kpos is not None:
                for (L, H) in heads:
                    fB, xhB = fcacheB[L]
                    dB = fra_edge(L, H, fB, get_sae(L).W_dec.float(), xhB, ho_tt)
                    deltaB = cell_delta(dB, A_pairs[(L, H)], seqB)  # B's score carried by A's feature-pairs
                    byL_cross.setdefault(L, {})[H] = deltaB
                for c in CS:
                    pe_c = torch.softmax(apply_cell_edit(ho_tt, byL_cross, c)[ho_q].float(), -1)[ho["tid"]].item()
                    cross[f"crossA_on_B_c{c}"] = dict(p=float(pe_c), drop=float(1 - pe_c / ho_base) if ho_base > 0 else float("nan"))
        except Exception as e:
            cross["error"] = str(e)[:120]

        rec = dict(prompt=fact["prompt"], subject=fact["subject"], target_true=fact["target_true"],
                   relation_id=fact["relation_id"], tid=tid, base=float(base),
                   kpos=int(kpos), edited=edited, R_oracle_edgecut=R_oracle,
                   heldout=dict(prompt=ho["prompt"], subject=ho["subject"], base=float(ho_base),
                                ownB=collateral, crossA_on_B=cross))
        state["facts"].append(rec); done_prompts.add(fact["prompt"]); ckpt()
        d1 = edited["1.0"]["drop"]; d2 = edited["2.0"]["drop"]
        cstr = cross.get("crossA_on_B_c1.0", {}).get("drop")
        log(f"  [{fi:2d}] '{fact['prompt'][:38]:38s}' base={base:.2f} drop@c1={d1:.2f} drop@c2={d2:.2f} "
            f"oracle={R_oracle:.2f} | crossA->B@c1={cstr}")
    except Exception as e:
        log(f"  [{fi}] ERROR {e}\n{traceback.format_exc()[:300]}")
        state.setdefault("errors", []).append(str(e)[:200]); ckpt()

# ---- STAGE 4: aggregate + verdict ---------------------------------------
log(f"[{time.strftime('%H:%M:%S')}] STAGE 4 aggregate")
facts = [f for f in state["facts"] if "edited" in f]
def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else None
def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None
def frac_ge(xs, t):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean([x >= t for x in xs])) if xs else None

drop_c1 = [f["edited"]["1.0"]["drop"] for f in facts]
drop_c2 = [f["edited"]["2.0"]["drop"] for f in facts]
oracle  = [f.get("R_oracle_edgecut") for f in facts]
# collateral = cross-applied A's support onto B (the clean MLP-distribution probe), c=1 and c=2
coll_c1 = [f["heldout"]["crossA_on_B"].get("crossA_on_B_c1.0", {}).get("drop") for f in facts]
coll_c2 = [f["heldout"]["crossA_on_B"].get("crossA_on_B_c2.0", {}).get("drop") for f in facts]

summary = dict(
    n_facts=len(facts),
    recalled_n=state.get("recalled_n"), n_relations=state.get("n_relations"),
    heads=state["heads"], head_effects=state.get("head_effects"),
    median_drop_c1=med(drop_c1), median_drop_c2=med(drop_c2),
    mean_drop_c1=mean(drop_c1), mean_drop_c2=mean(drop_c2),
    frac_drop_ge50_c1=frac_ge(drop_c1, 0.5), frac_drop_ge50_c2=frac_ge(drop_c2, 0.5),
    median_oracle_edgecut=med(oracle),
    median_collateral_crossA_on_B_c1=med(coll_c1),
    median_collateral_crossA_on_B_c2=med(coll_c2),
    mean_collateral_crossA_on_B_c1=mean(coll_c1),
    mean_collateral_crossA_on_B_c2=mean(coll_c2),
)
# GATE (i): does cell-edit drop P(target) >=50% at c~=1-2 ? use best of median c1/c2.
gi_metric = max([m for m in [summary["median_drop_c1"], summary["median_drop_c2"]] if m is not None] or [0.0])
gate_i = gi_metric >= 0.50
# GATE (ii): held-out collateral <15% under the SAME edit. Use median cross-apply drop at the c
#            that achieves gate (i) (prefer c=1 if it already passes, else c=2).
if summary["median_drop_c1"] is not None and summary["median_drop_c1"] >= 0.50:
    gii_metric = summary["median_collateral_crossA_on_B_c1"]
else:
    gii_metric = summary["median_collateral_crossA_on_B_c2"]
gate_ii = (gii_metric is not None) and (abs(gii_metric) < 0.15)

verdict = "GO" if (gate_i and gate_ii) else "NO-GO"
summary["gate_i_drop>=50%"] = dict(metric=gi_metric, pass_=bool(gate_i))
summary["gate_ii_collateral<15%"] = dict(metric=gii_metric, pass_=bool(gate_ii))
summary["VERDICT"] = verdict
state["summary"] = summary
ckpt()

log("\n================ FACTEDIT PRE-CHECK SUMMARY ================")
log(f"  recalled facts: {summary['recalled_n']} across {summary['n_relations']} relations; edited n={summary['n_facts']}")
log(f"  extraction heads: {summary['heads']}  effects={summary['head_effects']}")
log(f"  GATE (i)  median P(target) drop: c1={summary['median_drop_c1']}, c2={summary['median_drop_c2']}  "
    f"(frac>=50%: c1={summary['frac_drop_ge50_c1']}, c2={summary['frac_drop_ge50_c2']})  oracle-edgecut={summary['median_oracle_edgecut']}")
log(f"  GATE (ii) median held-out collateral (A-support cross-applied to B): c1={summary['median_collateral_crossA_on_B_c1']}, c2={summary['median_collateral_crossA_on_B_c2']}")
log(f"  GATE (i)  drop>=50%   : {gate_i}  (metric={gi_metric:.3f})")
log(f"  GATE (ii) collateral<15%: {gate_ii}  (metric={gii_metric})")
log(f"  >>> VERDICT: {verdict} <<<")
json.dump(summary, open(os.path.join(OUT, "factedit_summary.json"), "w"), indent=2, default=float)
log("DONE factedit_precheck")

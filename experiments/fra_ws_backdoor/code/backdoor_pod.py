#!/usr/bin/env python3
"""Phase B — IN-CONTEXT BACKDOOR on the weight-sparse ladder (LOCATE / CUT / DRIFT).

Plants the ARROW_COMMENT in-context backdoor (Phase F's strongest shared mapping):
a distinctive TRIGGER token T demonstrated in-context as `# T -> P` (k=3 demos) makes
the query `# T ->` emit the PAYLOAD P (the backdoor fires); without the demos / for a
non-trigger token the completion is benign. T and P are NORMAL code identifiers with
legitimate uses (so the linear/mask baselines pay collateral).

Per pre-registration (CAMPAIGN.md):
  1. LOCATE — neuron-basis FRA of the trigger->payload QK induction edge at the answer
     step on LOCATE prompts; top cells (signed-mean across prompts, by |score|).
  2. CUT (win test) — cell-cut as a position-invariant score-level edit; on HELD-OUT
     prompts (novel contexts, trigger at NOVEL positions):
       - payload removal vs the position-aware ORACLE (zero the trigger->answer edge);
       - collateral (held-out benign text where T & P appear in non-backdoor contexts)
         vs TWO matched baselines: token-mask-with-detection (attention-map analogue)
         and best single-direction linear edit (payload-suppress / ActAdd-trigger);
       - WIN BAR: >=2x collateral advantage at matched removal + position-invariance.
  3. DRIFT — is the planted trigger->payload edge cell STABLE across contexts/positions
     in the neuron basis? (B1 quote: 200/200; dense gpt2 drifts.)
  sparse vs wsda vs dense (byte-identical afrac pair = clean act-sparsity isolation).

Reuses the B1 machinery in ws_pod.py (loader, neuron-FRA, patched-attention cell-cut).
Resumable per-model, partial-upload, traceback-upload.
HF: dmanningcoe/fra-phase1-steering-data : fra_ws_backdoor/results/
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_pod as W           # reused: load_ws_model, head_qk_weights, omega_from_decoder,
                             # collect_act_in, edge_cell_matrix, append_bias_channel,
                             # cell_metrics, edge_matrix_cosine, PatchedAttn, ATT, pad_batch
from csp_vendor_hook_utils import hook_recorder

DEV = W.DEV
SEED = int(os.environ.get("SEED", "0"))
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/out")
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
HF_PREFIX = "fra_ws_backdoor/results"
LOCAL_SMOKE = os.environ.get("LOCAL_SMOKE", "0") == "1"

MODELS = W.MODELS
MODEL_ORDER = os.environ.get("MODEL_ORDER", "sparse,wsda,dense").split(",")
K_DEMOS = int(os.environ.get("K_DEMOS", "3"))
N_LOC = int(os.environ.get("N_LOC", "30"))
N_HOLD = int(os.environ.get("N_HOLD", "30"))
N_BENIGN = int(os.environ.get("N_BENIGN", "30"))
N_TOP_CELLS = int(os.environ.get("N_TOP_CELLS", "10"))
# trigger->payload pairs (both single-token in this vocab; normal identifiers with legit
# code uses, so the linear/mask baselines pay collateral on the endpoints' legit uses)
PAIRS = [("node", "result"), ("data", "index"), ("value", "output")]
if os.environ.get("PAIRS"):
    PAIRS = [tuple(p.split(":")) for p in os.environ["PAIRS"].split(",")]
FRA_SCALES = [0.5, 1.0, 2.0, 3.0, 5.0]
MASK_STR = [1.0, 2.0, 4.0, 8.0, 30.0]
LIN_ALPHA = [2.0, 4.0, 8.0, 16.0, 32.0]

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


def upload(path, name=None):
    if LOCAL_SMOKE:
        return False
    name = name or os.path.basename(path)
    for i in range(5):
        try:
            W.hf_api().upload_file(path_or_fileobj=path, path_in_repo=f"{HF_PREFIX}/{name}",
                                   repo_id=HF_REPO, repo_type="dataset")
            return True
        except Exception as e:
            print(f"[upload] {name} attempt {i}: {e}", flush=True)
            time.sleep(20 * (i + 1))
    return False


# ---------------- prompt construction ----------------
FILLER = [
    "x = 1", "y = 2", "import os", "def f():", "    return 0", "for i in range(3):",
    "    pass", "z = [1, 2, 3]", "if True:", "    a = b + c", "s = 'hi'", "n = len(z)",
    "class A:", "    pass", "w = {}", "print(x)", "k = 0", "m = x * y",
]


def single_tok_ok(tok, w):
    enc = tok.encode(f"q {w}")
    idx = len("q ")
    try:
        j = next(i for i, (a, b) in enumerate(enc.offsets) if a <= idx < b)
    except StopIteration:
        return False
    dec = tok.decode([enc.ids[j]]).strip()
    return dec == w


def pick_pairs(tok, want=3):
    """single-token-validated (T,P) pairs; both are NORMAL identifiers with legit code
    uses (so the linear/mask baselines pay collateral on the endpoints). Falls back to the
    single-token pool if a preregistered distinctive pair is not single-token in this vocab."""
    cands = [w for w in (W.WORDS + W.IDENTS) if single_tok_ok(tok, w)]
    cset = set(cands)
    pref = [("node", "result"), ("data", "index"), ("value", "output"),
            ("token", "graph"), ("alpha", "beta"), ("input", "count")]
    out, used = [], set()
    for T, P in pref:
        if T in cset and P in cset and T not in used and P not in used:
            out.append((T, P)); used.update((T, P))
            if len(out) >= want:
                return out
    rem = [w for w in cands if w not in used]
    while len(out) < want and len(rem) >= 2:
        T, P = rem.pop(), rem.pop()
        out.append((T, P))
    return out


def find_payload_kpos(tok, ids_text, T, P, n_demos):
    """char offset of the FIRST demo's payload token P (the token induction copies)."""
    # demos are "# T -> P\n" repeated; first P starts after the first "# T -> "
    first = ids_text.index(f"# {T} -> {P}")
    return first + len(f"# {T} -> ")


def build_backdoor(tok, T, P, n, n_demos, rng, novel_pos=False):
    """ON prompts: demos planted then a query `# T -> ` at the answer step.
    novel_pos=True -> randomized filler depth so T/P/query land at NOVEL absolute positions."""
    out = []
    for _ in range(n):
        pre_n = rng.randint(3, 9) if novel_pos else rng.randint(0, 2)
        mid_n = rng.randint(2, 7) if novel_pos else rng.randint(0, 1)
        pre = "".join(rng.choice(FILLER) + "\n" for _ in range(pre_n))
        demos = f"# {T} -> {P}\n" * n_demos
        mid = "".join(rng.choice(FILLER) + "\n" for _ in range(mid_n))
        # query has NO trailing space: the payload token is space-led (" P"), so the
        # answer step must be the "->" token (the space belongs to the target, not the prompt).
        text = pre + demos + mid + f"# {T} ->"
        ids = tok.encode(text).ids
        kp_char = len(pre) + len(f"# {T} -> ")  # first demo's P char offset (space-led token)
        enc = tok.encode(text)
        kp = next(i for i, (a, b) in enumerate(enc.offsets) if a <= kp_char < b)
        # target payload token id (the token rendered for P in the demo)
        tgt = enc.ids[next(i for i, (a, b) in enumerate(enc.offsets)
                           if a <= kp_char < b)]
        # trigger key positions (tokens that render T)
        tpos = [i for i, t in enumerate(ids) if tok.decode([t]).strip() == T]
        out.append({"text": text, "ids": ids, "kp": kp, "target": tgt,
                    "qp": len(ids) - 1, "tpos": tpos})
    return out


def build_benign(tok, T, P, n, rng):
    """held-out benign text where T and P appear in NON-backdoor contexts + benign induction."""
    pool = [w for w in W.IDENTS + W.WORDS if single_tok_ok(tok, w) and w not in (T, P)]
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.4:
            # T and P as normal identifiers
            text = (f"{T} = {rng.randint(0,9)}\n{P} = []\n"
                    f"for {T} in range(3):\n    {P}.append({T})\nprint({P})")
        elif kind < 0.7:
            # benign induction with OTHER tokens (general-induction preservation)
            a, b = rng.sample(pool, 2)
            text = f"# {a} -> {b}\n# {a} -> {b}\n# {a} -> {b}\n# {a} -> {b}"
        else:
            # T appears as a benign arrow-mapping to a NON-payload (T's other in-context use)
            b = rng.choice(pool)
            text = f"# {T} -> {b}\n# {T} -> {b}\n# {T} -> {b}\n# {T} -> {b}"
        ids = tok.encode(text).ids
        out.append({"text": text, "ids": ids})
    return out


# ---------------- scoring ----------------
@torch.no_grad()
def asr(model, prompts):
    """per-prompt P(payload) at the answer step (single batch)."""
    ids, lens = W.pad_batch([p["ids"] for p in prompts])
    logits, _, _ = model(ids)
    out = []
    for b, p in enumerate(prompts):
        pr = F.softmax(logits[b, lens[b] - 1].float(), dim=-1)
        out.append(float(pr[p["target"]]))
    return np.array(out)


@torch.no_grad()
def clean_logits(model, prompts):
    ids, lens = W.pad_batch([p["ids"] for p in prompts])
    lg, _, _ = model(ids)
    return lg.float(), lens


def kl_collateral(model, prompts, edit_apply):
    """mean KL(clean || edited) over all valid next-token positions of benign prompts."""
    ids, lens = W.pad_batch([p["ids"] for p in prompts])
    with torch.no_grad():
        lg_clean, _, _ = model(ids)
        lg_clean = lg_clean.float()
        lg_edit = edit_apply(ids)
    kls = []
    for b in range(ids.shape[0]):
        T = int(lens[b])
        pc = F.log_softmax(lg_clean[b, :T], dim=-1)
        pe = F.log_softmax(lg_edit[b, :T], dim=-1)
        kl = (pc.exp() * (pc - pe)).sum(-1)            # [T]
        kls.append(float(kl.mean()))
    return float(np.mean(kls))


# ---------------- edits (attention-side via PatchedAttn, residual-side via hooks) ----------------
def fra_delta_fn(cells, U1, omega, c):
    def fn(x):
        B, Tn = U1.shape[0], x.shape[1]
        d = torch.zeros(B, Tn, Tn, device=x.device)
        for (mu, nu) in cells:
            d -= c * omega[mu, nu] * torch.einsum("bq,bk->bqk", U1[:, :Tn, mu], U1[:, :Tn, nu])
        return d
    return fn


@torch.no_grad()
def run_att_edit_logits(model, L, H, ids, fn=None, fixed=None):
    W.ATT.layer, W.ATT.head, W.ATT.fn, W.ATT.fixed = L, H, fn, fixed
    with W.PatchedAttn(model, L):
        lg, _, _ = model(ids)
    return lg.float()


@torch.no_grad()
def run_lin_edit_logits(model, ids, direction, alpha):
    d = direction / (direction.norm() + 1e-9)

    def sub(t):
        return t - alpha * (t @ d).unsqueeze(-1) * d   # remove the component along d

    with hook_recorder(regex="$^", interventions={"final_resid": sub}):
        lg, _, _ = model(ids)
    return lg.float()


def removal_from_logits(lg, prompts, base_asr):
    cut = []
    ids_lens = [len(p["ids"]) for p in prompts]
    for b, p in enumerate(prompts):
        pr = F.softmax(lg[b, ids_lens[b] - 1], dim=-1)
        cut.append(float(pr[p["target"]]))
    cut = np.array(cut)
    return float(1.0 - cut.mean() / max(base_asr, 1e-9)), float(cut.mean())


def interp_at(removals, collaterals, target):
    """collateral at the strength whose removal first reaches target (linear interp)."""
    pts = sorted(zip(removals, collaterals))
    if pts[-1][0] < target:
        return None  # never reaches target
    for i in range(1, len(pts)):
        r0, c0 = pts[i - 1]; r1, c1 = pts[i]
        if r1 >= target:
            if r1 == r0:
                return c1
            w = (target - r0) / (r1 - r0)
            return c0 + w * (c1 - c0)
    return pts[-1][1]


# ---------------- per (model, pair) pipeline ----------------
def run_pair(model, config, tok, T, P, mkey):
    rng = random.Random(SEED + hash((T, P)) % 9999)
    loc = build_backdoor(tok, T, P, N_LOC, K_DEMOS, rng, novel_pos=False)
    hold = build_backdoor(tok, T, P, N_HOLD, K_DEMOS, rng, novel_pos=True)   # NOVEL positions
    benign = build_benign(tok, T, P, N_BENIGN, rng)

    base_loc = asr(model, loc).mean()
    base_hold = asr(model, hold).mean()
    rec = {"trigger": T, "payload": P, "base_asr_locate": float(base_loc),
           "base_asr_holdout": float(base_hold)}
    if base_loc < 0.2:
        rec["skipped"] = f"weak backdoor (base ASR {base_loc:.2f})"
        return rec

    # --- find induction head carrying the backdoor (max ASR drop when zeroed) ---
    base, ranked = W.find_top_head(model, config, tok, loc, asr_score_adapter)
    (L, H), drop = ranked[0]
    rec["head"] = {"layer": L, "head": H, "asr_drop": float(drop)}

    # --- neuron-basis FRA of the trigger->payload edge ---
    dh = config.d_head
    scale = 1.0 / math.sqrt(dh)
    Wq, Wk, bq, bk = W.head_qk_weights(model, config, L, H)
    I = torch.eye(config.d_model, device=DEV)
    omega = W.omega_from_decoder(I, I, Wq, Wk, bq, bk, scale)

    def fra_edges(prompts):
        ids, lens = W.pad_batch([p["ids"] for p in prompts])
        U, _, _ = W.collect_act_in(model, config, L, ids)
        U1 = W.append_bias_channel(U)
        Ms = [W.edge_cell_matrix(U1[b], omega, prompts[b]["qp"], prompts[b]["kp"]).cpu()
              for b in range(len(prompts))]
        return U1, Ms

    U1l, Msl = fra_edges(loc)
    U1h, Msh = fra_edges(hold)
    cm = W.cell_metrics(Msl + Msh)
    cm["edge_cosine"] = W.edge_matrix_cosine(Msl + Msh)
    rec["fra_edge"] = {k: cm[k] for k in ("top1_cell", "top1_coverage", "n_cells_for_90",
                                          "edge_mass_top1", "edge_mass_top8", "edge_cosine",
                                          "n_distinct_dom")}

    # top cells by signed-mean |score| across LOCATE
    smat = torch.stack(Msl).mean(0).abs()
    Fp = smat.shape[0]
    top_idx = smat.flatten().argsort(descending=True)[:N_TOP_CELLS].tolist()
    cells = [(int(i // Fp), int(i % Fp)) for i in top_idx]
    rec["top_cells"] = [list(c) for c in cells]

    # ---------------- CUT: removal curves on HOLDOUT (novel positions) ----------------
    Bh, Th = U1h.shape[0], U1h.shape[1]
    ids_h, lens_h = W.pad_batch([p["ids"] for p in hold])

    # FRA cell-cut (position-invariant score edit), scale sweep
    fra_curve = []
    for c in FRA_SCALES:
        lg = run_att_edit_logits(model, L, H, ids_h, fn=fra_delta_fn(cells, U1h, omega, c))
        r, cut = removal_from_logits(lg, hold, base_hold)
        fra_curve.append({"c": c, "removal": r, "cut_asr": cut})

    # position-aware ORACLE (zero the trigger->answer edge), strength sweep -> removal ceiling
    orc_curve = []
    for s in MASK_STR:
        fixed = torch.zeros(Bh, Th, Th, device=DEV)
        for b, p in enumerate(hold):
            fixed[b, p["qp"], p["kp"]] = -s
        lg = run_att_edit_logits(model, L, H, ids_h, fixed=fixed)
        r, cut = removal_from_logits(lg, hold, base_hold)
        orc_curve.append({"s": s, "removal": r, "cut_asr": cut})

    # token-mask-with-detection (zero attention onto T wherever detected), strength sweep
    tm_curve = []
    for s in MASK_STR:
        fixed = torch.zeros(Bh, Th, Th, device=DEV)
        for b, p in enumerate(hold):
            for kpos in p["tpos"]:
                fixed[b, :, kpos] = -s
        lg = run_att_edit_logits(model, L, H, ids_h, fixed=fixed)
        r, cut = removal_from_logits(lg, hold, base_hold)
        tm_curve.append({"s": s, "removal": r, "cut_asr": cut})

    # payload-suppress linear (remove unembed direction of P), alpha sweep
    dirP = model.lm_head.weight[hold[0]["target"]].float().detach()
    lin_curve = []
    for a in LIN_ALPHA:
        lg = run_lin_edit_logits(model, ids_h, dirP, a)
        r, cut = removal_from_logits(lg, hold, base_hold)
        lin_curve.append({"a": a, "removal": r, "cut_asr": cut})

    rec["removal_curves"] = {"fra": fra_curve, "oracle": orc_curve,
                             "token_mask": tm_curve, "payload_suppress": lin_curve}

    # ---------------- COLLATERAL on benign held-out text, matched to a removal target ----------------
    def fra_collateral(c):
        ids_b, _ = W.pad_batch([p["ids"] for p in benign])
        Ub, _, _ = W.collect_act_in(model, config, L, ids_b)
        U1b = W.append_bias_channel(Ub)
        return kl_collateral(model, benign,
                             lambda ids: run_att_edit_logits(model, L, H, ids,
                                                             fn=fra_delta_fn(cells, U1b, omega, c)))

    def tm_collateral(s):
        def apply(ids):
            B, Tn = ids.shape
            fixed = torch.zeros(B, Tn, Tn, device=DEV)
            for b, p in enumerate(benign):
                tpos = [i for i, t in enumerate(p["ids"]) if tok.decode([t]).strip() == T]
                for kpos in tpos:
                    if kpos < Tn:
                        fixed[b, :, kpos] = -s
            return run_att_edit_logits(model, L, H, ids, fixed=fixed)
        return kl_collateral(model, benign, apply)

    def lin_collateral(a):
        return kl_collateral(model, benign, lambda ids: run_lin_edit_logits(model, ids, dirP, a))

    # build (removal, collateral) for the 3 DEPLOYABLE methods, pick matched removal target
    def curve_with_collateral(rem_curve, key, coll_fn):
        out = []
        for pt in rem_curve:
            sval = pt[key]
            out.append({"strength": sval, "removal": pt["removal"], "collateral": coll_fn(sval)})
        return out

    fra_cc = curve_with_collateral(fra_curve, "c", fra_collateral)
    tm_cc = curve_with_collateral(tm_curve, "s", tm_collateral)
    lin_cc = curve_with_collateral(lin_curve, "a", lin_collateral)

    max_rem = min(max(p["removal"] for p in fra_cc),
                  max(p["removal"] for p in tm_cc),
                  max(p["removal"] for p in lin_cc))
    target = min(0.8, max(0.5, max_rem - 1e-6))
    coll = {}
    for nm, cc in [("fra", fra_cc), ("token_mask", tm_cc), ("payload_suppress", lin_cc)]:
        rr = [p["removal"] for p in cc]; cl = [p["collateral"] for p in cc]
        coll[nm] = {"collateral_at_target": interp_at(rr, cl, target),
                    "curve": cc}
    rec["collateral"] = {"removal_target": target, "by_method": coll}

    # win ratios at matched removal
    cf = coll["fra"]["collateral_at_target"]
    ratios = {}
    for nm in ("token_mask", "payload_suppress"):
        cb = coll[nm]["collateral_at_target"]
        ratios[nm] = (cb / cf) if (cf and cf > 1e-9 and cb is not None) else None
    rec["win_ratios"] = ratios
    rec["fra_collateral"] = cf

    # position-invariance: FRA removal at faithful c=1 on holdout (novel pos) vs locate
    lg_loc = run_att_edit_logits(model, L, H, W.pad_batch([p["ids"] for p in loc])[0],
                                 fn=fra_delta_fn(cells, U1l, omega, 1.0))
    rem_loc_c1, _ = removal_from_logits(lg_loc, loc, base_loc)
    rem_hold_c1 = next(p["removal"] for p in fra_curve if p["c"] == 1.0)
    rec["position_invariance"] = {"fra_removal_c1_locate": rem_loc_c1,
                                  "fra_removal_c1_holdout_novelpos": rem_hold_c1,
                                  "oracle_max_removal": max(p["removal"] for p in orc_curve),
                                  "fra_max_removal": max(p["removal"] for p in fra_curve)}
    return rec


# adapter so ws_pod.find_top_head (expects fn(model, tok, prompts, batch=...)->scores) can use asr
def asr_score_adapter(model, tok, prompts, batch=None):
    return asr(model, prompts)


def eval_model(mkey, tok):
    model, config = W.load_ws_model(MODELS[mkey])
    res = {"model": MODELS[mkey], "arch": {"n_layer": config.n_layer, "afrac": config.afrac},
           "pairs": {}}
    for (T, P) in PAIRS:
        if not (single_tok_ok(tok, T) and single_tok_ok(tok, P)):
            res["pairs"][f"{T}->{P}"] = {"skipped": "not single-token"}
            continue
        t0 = time.time()
        r = run_pair(model, config, tok, T, P, mkey)
        res["pairs"][f"{T}->{P}"] = r
        print(f"[bd/{mkey}/{T}->{P}] base_asr={r.get('base_asr_locate'):.2f} "
              f"head=L{r.get('head',{}).get('layer')}H{r.get('head',{}).get('head')} "
              f"top1cov={r.get('fra_edge',{}).get('top1_coverage')} "
              f"win={r.get('win_ratios')} {time.time()-t0:.0f}s", flush=True)
    # aggregate
    valid = [r for r in res["pairs"].values() if "win_ratios" in r]
    if valid:
        def med(key_path):
            vals = []
            for r in valid:
                v = r
                for k in key_path:
                    v = v.get(k) if isinstance(v, dict) else None
                if isinstance(v, (int, float)):
                    vals.append(v)
            return float(np.median(vals)) if vals else None
        res["summary"] = {
            "n_pairs": len(valid),
            "median_top1_coverage": med(["fra_edge", "top1_coverage"]),
            "median_edge_cosine": med(["fra_edge", "edge_cosine"]),
            "median_edge_mass_top1": med(["fra_edge", "edge_mass_top1"]),
            "median_win_vs_token_mask": float(np.median(
                [r["win_ratios"]["token_mask"] for r in valid
                 if r["win_ratios"].get("token_mask")])) if any(
                r["win_ratios"].get("token_mask") for r in valid) else None,
            "median_win_vs_payload_suppress": float(np.median(
                [r["win_ratios"]["payload_suppress"] for r in valid
                 if r["win_ratios"].get("payload_suppress")])) if any(
                r["win_ratios"].get("payload_suppress") for r in valid) else None,
            "median_fra_removal_holdout_c1": med(["position_invariance", "fra_removal_c1_holdout_novelpos"]),
            "median_fra_max_removal": med(["position_invariance", "fra_max_removal"]),
            "median_oracle_max_removal": med(["position_invariance", "oracle_max_removal"]),
        }
    del model
    return res


def main():
    global PAIRS
    tok = W.load_tokenizer()
    # validate the trigger->payload pairs are single-token in THIS vocab (else the target
    # alignment is wrong); if no env override, auto-pick single-token distinctive identifiers.
    if not os.environ.get("PAIRS"):
        PAIRS = pick_pairs(tok)
    else:
        PAIRS = [p for p in PAIRS if single_tok_ok(tok, p[0]) and single_tok_ok(tok, p[1])]
        if not PAIRS:
            PAIRS = pick_pairs(tok)
    print(f"[bd] pairs: {PAIRS}", flush=True)
    out_path = os.path.join(OUT_DIR, "backdoor_results.json")
    out = {"config": {"seed": SEED, "k_demos": K_DEMOS, "pairs": PAIRS,
                      "fra_scales": FRA_SCALES, "n_loc": N_LOC, "n_hold": N_HOLD,
                      "n_benign": N_BENIGN, "n_top_cells": N_TOP_CELLS}, "models": {}}
    if os.path.exists(out_path):
        try:
            out = json.load(open(out_path))
        except Exception:
            pass
    for mkey in MODEL_ORDER:
        if mkey in out.get("models", {}):
            print(f"[bd] {mkey} already done, skip", flush=True)
            continue
        out.setdefault("models", {})[mkey] = eval_model(mkey, tok)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, default=float)
        upload(out_path)
        print(f"[bd] {mkey} summary={out['models'][mkey].get('summary')}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    upload(out_path)
    print("[bd] ALL DONE", flush=True)
    for mkey, m in out["models"].items():
        print(f"  {mkey}: {m.get('summary')}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        tp = os.path.join(OUT_DIR, "backdoor_traceback.txt")
        with open(tp, "w") as f:
            f.write(tb)
        upload(tp)
        raise

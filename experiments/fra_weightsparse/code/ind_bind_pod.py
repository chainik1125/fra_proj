#!/usr/bin/env python3
"""Candidates re-do pod (rs-ws2-*): T1 identifier-induction + T2 binding-up-the-ladder.

T1 IDENTIFIER-INDUCTION (the gpt2 persistence/drift re-do on the weight-sparse ladder):
  code-native induction — identifier `base_suf` bound early, prompt ends mid-repeat at the
  '_' (token-boundary-verified); target = first suffix token at the definition; distractor =
  the OTHER identifier's suffix token (2AFC). Gate = full-vocab argmax acc >= 0.70 (prereg),
  variant ladder v0 (default) -> v1 (short range) -> v2 (extra repetition) -> 2x-wide models.
  Then the persistence suite in the NEURON basis: top1_coverage / n_cells_for_90 / q-side
  drift / edge-cosine / co-firing drift / edge-subspace PR / Spearman(s,c) + recovery + oracle.

T2 BINDING UP THE LADDER: gate-sweep csp_sweep1_{EF}x_{NZ}nonzero_afrac0.250 ascending
  (EF, NZ) on set_or_string (score-acc >= 0.70); pick the smallest passing sparse + its
  afrac1.000 twin + width-matched dense1_{EF}x. Then neuron-basis FRA on the binding edge +
  row-based causal block (Spearman/recovery/oracle) + SIBLING selectivity (cut set-binding
  cells, measure str-binding collateral in the same contexts) + drift across instances.

Reuses ws_pod.py pure compute (loader, FRA, cell metrics, ATT/PatchedAttn, causal_block).
Own outputs -> OUT_DIR + HF fra_weightsparse/results/induction_binding/.
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
import ws_pod as W  # noqa: E402  (module-level: env config, seeds, OUT_DIR mkdir)

# ---------------- config ----------------
DEV = W.DEV
SEED = W.SEED
OUT_DIR = W.OUT_DIR
STAGES = os.environ.get("STAGES", "T1T2")
SMOKE = os.environ.get("SMOKE", "0") == "1"
SKIP_UPLOAD = os.environ.get("SKIP_UPLOAD", "0") == "1"
HF_PREFIX2 = "fra_weightsparse/results/induction_binding"
GATE = float(os.environ.get("TASK_GATE", "0.70"))

N_IND = int(os.environ.get("N_IND", "40" if SMOKE else "200"))       # locate/holdout = half/half
N_BIND_GATE = int(os.environ.get("N_BIND_GATE", "40" if SMOKE else "160"))
N_BIND_FRA = int(os.environ.get("N_BIND_FRA", "40" if SMOKE else "160"))  # mixed, half/half
N_PAIRS = int(os.environ.get("N_PAIRS", "24" if SMOKE else "120"))   # selectivity pairs, half/half
HEADFIND_N = int(os.environ.get("HEADFIND_N", "24" if SMOKE else "64"))

if SMOKE:
    W.N_CAUSAL_TOP, W.N_CAUSAL_RAND = 20, 10
N_CTOP, N_CRAND = W.N_CAUSAL_TOP, W.N_CAUSAL_RAND


def upload2(path, name=None):
    if SKIP_UPLOAD:
        return True
    name = name or os.path.basename(path)
    for i in range(5):
        try:
            W.hf_api().upload_file(path_or_fileobj=path, path_in_repo=f"{HF_PREFIX2}/{name}",
                                   repo_id=W.HF_REPO, repo_type="dataset")
            return True
        except Exception as e:
            print(f"[upload2] {name} attempt {i}: {e}", flush=True)
            time.sleep(30 * (i + 1))
    return False


def ckpt(out):
    p = os.path.join(OUT_DIR, "ind_bind_partial.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    upload2(p)


# ================= T1: identifier-induction =================
CONS = "bcdfghjklmnpqrstvwz"
VOW = "aeiou"
PRELUDES = ["", "import os\n", "def f():\n    pass\n", "# setup\n",
            "for i in range(2):\n    pass\n", "class A:\n    pass\n",
            "z = [1, 2]\n", "if True:\n    pass\n"]
REUSE_PRE = ["y = ", "print(", "out = ", "w = ", "q = ", "res = "]

VARIANTS = {
    "v0": {"n_fill": (1, 2), "repeat": False},   # default: 2 defs + 1-2 fillers + query
    "v1": {"n_fill": (0, 0), "repeat": False},   # shorter range
    "v2": {"n_fill": (1, 2), "repeat": True},    # extra in-context repetition of A
}


def _rand_base(rng, nsyl=None):
    nsyl = nsyl or rng.choice([2, 3])
    return "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(nsyl))


def make_ind_prompt(tok, rng, variant):
    """token-boundary-verified induction prompt; None if 60 resamples fail.

    Construction: full text reuses identifier A at the end; the prompt CUTS at the
    canonical token boundary before A's final token (on-distribution tokenization).
    target = A's final token, verified to be the IDENTICAL token (same span string,
    same id) at A's definition (k_pos). distractor = B's final definition token.
    """
    v = VARIANTS[variant]
    for _ in range(60):
        baseA, baseB = _rand_base(rng), _rand_base(rng)
        if baseA == baseB or baseA.startswith(baseB) or baseB.startswith(baseA):
            continue
        sufA, sufB = _rand_base(rng, 1) + rng.choice(CONS), _rand_base(rng, 1) + rng.choice(CONS)
        A, B = f"{baseA}_{sufA}", f"{baseB}_{sufB}"
        lines = [f"{A} = {rng.randint(0, 99)}", f"{B} = {rng.randint(0, 99)}"]
        if rng.random() < 0.5:
            lines.reverse()
        nf = rng.randint(*v["n_fill"])
        for _ in range(nf):
            fb = _rand_base(rng)
            if fb in (baseA, baseB):
                continue
            lines.append(f"{fb} = {rng.randint(0, 99)}")
        if v["repeat"]:
            lines.append(f"{rng.choice('pqr')} = {A} + 1")
        text = rng.choice(PRELUDES) + "\n".join(lines) + "\n" + rng.choice(REUSE_PRE) + A
        enc = tok.encode(text)
        ids, offs = enc.ids, enc.offsets
        # A's reuse occurrence = the trailing len(A) chars
        ru_start = len(text) - len(A)
        span = [i for i, (a, b) in enumerate(offs) if b > ru_start]
        if len(span) < 3:                      # need >=2 match tokens + 1 target token
            continue
        t_last = span[-1]
        a0, b0 = offs[t_last]
        if b0 != len(text):
            continue
        target_id = ids[t_last]
        target_str = text[a0:b0]
        # definition occurrence: the SAME trailing token must exist there
        dc = text.index(f"{A} = ")
        d_start, d_end = dc + len(A) - len(target_str), dc + len(A)
        k_pos = None
        for i, (a, b) in enumerate(offs):
            if a == d_start and b == d_end:
                k_pos = i
                break
        if k_pos is None or ids[k_pos] != target_id:
            continue
        # distractor: B's final definition token
        dB = text.index(f"{B} = ")
        distractor_id = None
        for i, (a, b) in enumerate(offs):
            if b == dB + len(B):
                distractor_id = ids[i]
                break
        if distractor_id is None or distractor_id == target_id:
            continue
        return {"text": text[: a0], "ids": list(ids[:t_last]), "target_id": int(target_id),
                "distractor_id": int(distractor_id), "k_pos": int(k_pos), "A": A, "B": B}
    return None


def gen_ind_prompts(tok, n, variant, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        p = make_ind_prompt(tok, rng, variant)
        if p is not None:
            out.append(p)
    return out


@torch.no_grad()
def ind_scores(model, tok, prompts, batch=64):
    """2AFC P(target)/(P(target)+P(distractor)) at the last position; tok unused (ids cached)."""
    scores = []
    for s0 in range(0, len(prompts), batch):
        chunk = prompts[s0 : s0 + batch]
        ids, lens = W.pad_batch([p["ids"] for p in chunk])
        logits, _, _ = model(ids)
        for b, p in enumerate(chunk):
            pr = F.softmax(logits[b, lens[b] - 1].float(), dim=-1)
            pt, pd = pr[p["target_id"]].item(), pr[p["distractor_id"]].item()
            scores.append(pt / max(pt + pd, 1e-12))
    return np.array(scores)


@torch.no_grad()
def ind_top1_acc(model, tok, prompts, batch=64):
    """GATE metric: argmax over FULL vocab == bound identifier's next token."""
    hits = []
    for s0 in range(0, len(prompts), batch):
        chunk = prompts[s0 : s0 + batch]
        ids, lens = W.pad_batch([p["ids"] for p in chunk])
        logits, _, _ = model(ids)
        for b, p in enumerate(chunk):
            hits.append(int(logits[b, lens[b] - 1].argmax().item() == p["target_id"]))
    return float(np.mean(hits))


@torch.no_grad()
def t1_edge_oracle_scan(model, config, tok, prompts):
    """per-head removal from masking the (q=last -> k=def-target) edge; picks the EDGE head.

    Whole-head ablation drop ranks prev-token/support heads too (smoke: L0H14 argmax with
    edge-oracle 0.0) — the FRA edge head must be the one whose own edge-mask is causal.
    """
    ids, lens = W.pad_batch([p["ids"] for p in prompts])
    B, T = ids.shape
    fixed = torch.zeros(B, T, T, device=DEV)
    for b, p in enumerate(prompts):
        fixed[b, int(lens[b]) - 1, p["k_pos"]] = -1e9
    base = float(ind_scores(model, tok, prompts, batch=len(prompts)).mean())
    recs = []
    for L in range(config.n_layer):
        for H in range(config.n_head):
            W.ATT.layer, W.ATT.head, W.ATT.fixed = L, H, fixed
            with W.PatchedAttn(model, L):
                s = float(ind_scores(model, tok, prompts, batch=len(prompts)).mean())
            recs.append(((L, H), W.removal(base, s)))
    recs.sort(key=lambda kv: -kv[1])
    return base, recs


@torch.no_grad()
def t2_edge_oracle_scan(model, config, tok, prompts):
    """per-head removal from masking (completion queries -> init-value key) on bind rows."""
    ids, lens, metas = bind_rows(tok, prompts)
    B, T = ids.shape
    kps = [W.find_kpos_bind(tok, p)[1] for p in prompts]
    fixed = torch.zeros(B, T, T, device=DEV)
    for r, (nb, nc, _) in enumerate(metas):
        fixed[r, nb - 1 : nb + nc - 1, kps[r // 2]] = -1e9
    base = bind_clean_score(model, ids, metas, prompts)
    recs = []
    for L in range(config.n_layer):
        for H in range(config.n_head):
            s = bind_cut_score(model, ids, metas, prompts, L, H, None, None, None, fixed=fixed)
            recs.append(((L, H), W.removal(base, s)))
    recs.sort(key=lambda kv: -kv[1])
    return base, recs


def edge_pr(Ms, topc=512):
    """edge-subspace participation ratio over the top-`topc` cells by mean |mass|."""
    Fp = Ms[0].shape[0]
    macc = torch.zeros(Fp * Fp)
    for m in Ms:
        macc += m.abs().flatten().float()
    idx = macc.topk(min(topc, macc.numel())).indices
    X = torch.stack([m.flatten()[idx].float() for m in Ms])
    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    ev = torch.linalg.eigvalsh(X @ X.T).clamp(min=0)
    return float(ev.sum() ** 2 / max(float((ev ** 2).sum()), 1e-12))


def t1_fra_block(model, config, tok, prompts_loc, prompts_hold, L, H, mkey):
    dh = config.d_head
    scale = 1.0 / math.sqrt(dh)
    Wq, Wk, bq, bk = W.head_qk_weights(model, config, L, H)
    I = torch.eye(config.d_model, device=DEV)
    omega = W.omega_from_decoder(I, I, Wq, Wk, bq, bk, scale)

    def fra_on(prompts_):
        ids, lens = W.pad_batch([p["ids"] for p in prompts_])
        U, qh_all, kh_all = W.collect_act_in(model, config, L, ids)
        qh = qh_all[..., H * dh : (H + 1) * dh]
        kh = kh_all[..., H * dh : (H + 1) * dh]
        U1 = W.append_bias_channel(U)
        r2 = W.fra_recon_r2(U1, qh, kh, omega, lens)
        Ms, qps, kps = [], [], []
        for b, p in enumerate(prompts_):
            qp, kp = int(lens[b]) - 1, p["k_pos"]
            Ms.append(W.edge_cell_matrix(U1[b], omega, qp, kp).cpu())
            qps.append(qp)
            kps.append(kp)
        return U1, Ms, qps, kps, r2, lens

    U1l, Msl, qpsl, kpsl, r2l, lensl = fra_on(prompts_loc)
    U1h, Msh, qpsh, kpsh, r2h, lensh = fra_on(prompts_hold)
    cm = W.cell_metrics(Msl + Msh)
    cm["fra_recon_r2_locate"] = r2l
    cm["fra_recon_r2_holdout"] = r2h
    cm["cofire_qside"] = W.cofire_set_drift([U1l[b].cpu() for b in range(len(Msl))], qpsl)
    cm["edge_cosine"] = W.edge_matrix_cosine(Msl + Msh)
    cm["edge_subspace_pr"] = edge_pr(Msl + Msh)
    causal = W.causal_block(model, tok, prompts_loc, prompts_hold, ind_scores, L, H, omega,
                            U1l, Msl, U1h, lensl, lensh, qpsl, kpsl, qpsh, kpsh, {}, mkey, "neuron")
    return cm, causal


def stage_T1(out):
    res = out.setdefault("T1", {"config": {"N_IND": N_IND, "gate": GATE, "seed": SEED},
                                "models": {}})
    tok = load_tok()
    prompt_sets = {v: gen_ind_prompts(tok, N_IND, v, SEED * 100 + i)
                   for i, v in enumerate(VARIANTS)}
    res["example_prompt"] = {k: prompt_sets[k][0]["text"] for k in prompt_sets}

    model_list = [("sparse", W.MODELS["sparse"]), ("wsda", W.MODELS["wsda"]),
                  ("dense", W.MODELS["dense"])]
    if SMOKE:
        model_list = [("sparse", W.MODELS["sparse"]), ("dense", W.MODELS["dense"])]
    sparse_failed = False
    for mkey, name in model_list:
        run_t1_model(res, tok, prompt_sets, mkey, name, out)
        if mkey == "sparse" and res["models"]["sparse"].get("gate_pass") is None:
            sparse_failed = True
    if sparse_failed and not SMOKE:
        # prereg variant (iii): wider EF models
        for mkey, name in [("sparse2x", "csp_sweep1_2x_3.7Mnonzero_afrac0.250"),
                           ("wsda2x", "csp_sweep1_2x_3.7Mnonzero_afrac1.000"),
                           ("dense2x", "dense1_2x")]:
            run_t1_model(res, tok, prompt_sets, mkey, name, out)
    ckpt(out)


def run_t1_model(res, tok, prompt_sets, mkey, name, out):
    if mkey in res["models"] and "fra" in res["models"][mkey]:
        return  # resume skip
    model, config = W.load_ws_model(name)
    mrec = res["models"].setdefault(mkey, {"name": name})
    mrec["arch"] = {"n_layer": config.n_layer, "n_head": config.n_head,
                    "d_model": config.d_model, "afrac": config.afrac}
    chosen = None
    for vname in VARIANTS:
        ps = prompt_sets[vname]
        acc = ind_top1_acc(model, tok, ps)
        afc = float(ind_scores(model, tok, ps).mean())
        mrec.setdefault("gate", {})[vname] = {"top1_acc": acc, "mean_2afc": afc}
        print(f"[T1/{mkey}] {vname} top1_acc={acc:.3f} 2afc={afc:.3f}", flush=True)
        ckpt(out)
        if acc >= GATE:
            chosen = vname
            break
    mrec["gate_pass"] = chosen
    if chosen is None:
        print(f"[T1/{mkey}] GATE FAIL all variants", flush=True)
        del model
        return
    ps = prompt_sets[chosen]
    loc, hold = ps[: N_IND // 2], ps[N_IND // 2 :]
    base, ranked = W.find_top_head(model, config, tok, loc[:HEADFIND_N], ind_scores)
    baseE, rankedE = t1_edge_oracle_scan(model, config, tok, loc[:HEADFIND_N])
    (L, H), erem = rankedE[0]   # pick the EDGE head, not the ablation-drop head
    mrec["ind_head"] = {"layer": L, "head": H, "edge_oracle_probe": erem, "base": baseE,
                        "ablation_top5": [[list(k), float(v)] for k, v in ranked[:5]],
                        "edge_oracle_top8": [[list(k), float(v)] for k, v in rankedE[:8]]}
    print(f"[T1/{mkey}] edge head L{L}H{H} edge_rem={erem:.3f} "
          f"(ablation argmax {ranked[0][0]} drop={ranked[0][1]:.3f})", flush=True)
    cm, causal = t1_fra_block(model, config, tok, loc, hold, L, H, mkey)
    mrec["fra"] = cm
    mrec["causal"] = causal
    ckpt(out)
    del model
    torch.cuda.empty_cache() if DEV == "cuda" else None


_TOK = {}


def load_tok():
    if "t" not in _TOK:
        _TOK["t"] = W.load_tokenizer()
    return _TOK["t"]


# ================= T2: binding up the ladder =================
SWEEP_AF025 = [
    "csp_sweep1_1x_3.7Mnonzero_afrac0.250",   # B1 known-fail; re-measured for the table
    "csp_sweep1_1x_7.4Mnonzero_afrac0.250",
    "csp_sweep1_2x_0.9Mnonzero_afrac0.250",
    "csp_sweep1_2x_1.9Mnonzero_afrac0.250",
    "csp_sweep1_2x_3.7Mnonzero_afrac0.250",
    "csp_sweep1_2x_7.4Mnonzero_afrac0.250",
    "csp_sweep1_2x_14.8Mnonzero_afrac0.250",
    "csp_sweep1_4x_3.7Mnonzero_afrac0.250",
    "csp_sweep1_4x_7.4Mnonzero_afrac0.250",
    "csp_sweep1_4x_14.8Mnonzero_afrac0.250",
    "csp_sweep1_8x_3.7Mnonzero_afrac0.250",
    "csp_sweep1_8x_7.4Mnonzero_afrac0.250",
    "csp_sweep1_8x_14.8Mnonzero_afrac0.250",
]
SWEEP_AF05 = [n.replace("afrac0.250", "afrac0.500") for n in SWEEP_AF025[1:]]
DENSE_BY_EF = {"1x": "dense1_1x", "2x": "dense1_2x", "4x": "dense1_4x", "8x": "dense1_4x"}


def ef_of(name):
    for ef in ("16x", "1x", "2x", "4x", "8x"):
        if f"_{ef}_" in name:
            return ef
    return "1x"


def gen_bind_pairs(n, rng):
    """paired set/str-query prompts over the SAME context lines."""
    pairs = []
    for _ in range(n):
        v1, v2 = rng.sample(W.IDENTS, 2)
        lines = ([f"{v1} = set()", f'{v2} = ""'] if rng.random() < 0.5
                 else [f'{v2} = ""', f"{v1} = set()"])
        for _ in range(rng.randint(0, 2)):
            lines.append(f"{rng.choice([w for w in W.IDENTS if w not in (v1, v2)])} = {rng.randint(0, 99)}")
        ctx = "\n".join(lines)
        pairs.append((
            {"text": ctx + f"\n{v1}", "query_set": True, "set_var": v1, "str_var": v2},
            {"text": ctx + f"\n{v2}", "query_set": False, "set_var": v1, "str_var": v2},
        ))
    return pairs


# ---- row-based bind scoring/causal (teacher-forced rows; U1 cached ON the rows) ----
COMPS = {True: ".add(", False: " += "}


def bind_rows(tok, prompts):
    rows, metas = [], []
    for p in prompts:
        base = tok.encode(p["text"]).ids
        for is_set in (True, False):
            comp = tok.encode(COMPS[is_set]).ids
            rows.append(base + comp)
            metas.append((len(base), len(comp), is_set))
    ids, lens = W.pad_batch(rows)
    return ids, lens, metas


def bind_scores_from_logits(logits, ids, metas, prompts):
    logp = F.log_softmax(logits.float(), dim=-1)
    lps = {}
    for r, (nb, nc, is_set) in enumerate(metas):
        lp = sum(logp[r, nb - 1 + j, ids[r, nb + j]].item() for j in range(nc))
        lps.setdefault(r // 2, {})[is_set] = lp
    sc = []
    for j, p in enumerate(prompts):
        lc, lw = ((lps[j][True], lps[j][False]) if p["query_set"]
                  else (lps[j][False], lps[j][True]))
        sc.append(math.exp(lc) / max(math.exp(lc) + math.exp(lw), 1e-300))
    return np.array(sc)


@torch.no_grad()
def bind_cut_score(model, ids, metas, prompts, L, H, omega, cells, U1rows, fixed=None):
    """mean bind score under a cell-cut (or fixed att-delta) applied on the row batch."""
    W.ATT.layer, W.ATT.head = L, H
    if fixed is not None:
        W.ATT.fixed = fixed
    else:
        W.ATT.fn = W.cells_delta_fn(cells, U1rows, omega)
    with W.PatchedAttn(model, L):
        logits, _, _ = model(ids)
    return float(bind_scores_from_logits(logits, ids, metas, prompts).mean())


@torch.no_grad()
def bind_clean_score(model, ids, metas, prompts):
    logits, _, _ = model(ids)
    return float(bind_scores_from_logits(logits, ids, metas, prompts).mean())


@torch.no_grad()
def bind_fra_pack(model, config, tok, prompts, L, H, omega):
    """rows + U1rows + edge matrices (q = last prompt token, k = init-value token)."""
    dh = config.d_head
    ids, lens, metas = bind_rows(tok, prompts)
    U, qh_all, kh_all = W.collect_act_in(model, config, L, ids)
    U1 = W.append_bias_channel(U)
    qh = qh_all[..., H * dh : (H + 1) * dh]
    kh = kh_all[..., H * dh : (H + 1) * dh]
    r2 = W.fra_recon_r2(U1, qh, kh, omega, lens)
    Ms, qps, kps = [], [], []
    for j, p in enumerate(prompts):
        nb = metas[2 * j][0]
        _, kp = W.find_kpos_bind(tok, p)
        qp = nb - 1
        Ms.append(W.edge_cell_matrix(U1[2 * j], omega, qp, kp).cpu())
        qps.append(qp)
        kps.append(kp)
    return {"ids": ids, "lens": lens, "metas": metas, "U1": U1, "Ms": Ms,
            "qps": qps, "kps": kps, "r2": r2}


def rank_cells(Ms, n_top, n_rand, seed):
    Mstack = torch.stack(Ms)
    smat = Mstack.mean(0).abs()
    Fp = smat.shape[0]
    active = (Mstack.abs() > 1e-8).float().mean(0) >= 0.3
    flat_idx = torch.nonzero(active.flatten()).flatten()
    s_flat = smat.flatten()
    top_idx = s_flat.argsort(descending=True)[:n_top].tolist()
    rng = random.Random(seed)
    pool = [int(i) for i in flat_idx.tolist() if i not in set(top_idx)]
    rand_idx = rng.sample(pool, min(n_rand, len(pool)))
    union = top_idx + rand_idx
    cells = [(int(i // Fp), int(i % Fp)) for i in union]
    s_val = [float(s_flat[i]) for i in union]
    return cells, s_val


def bind_causal_block(model, tok, packL, packH, pl, ph, L, H, omega, mkey):
    """row-based mirror of ws_pod.causal_block for the binding task."""
    cells, s_val = rank_cells(packL["Ms"], N_CTOP, N_CRAND, SEED + 7)
    base_loc = bind_clean_score(model, packL["ids"], packL["metas"], pl)
    base_hold = bind_clean_score(model, packH["ids"], packH["metas"], ph)
    zval = bind_cut_score(model, packL["ids"], packL["metas"], pl, L, H, omega, [], packL["U1"])
    assert abs(zval - base_loc) < 1e-3, f"patched-forward mismatch: {zval} vs {base_loc}"
    c_eff = []
    t0 = time.time()
    for ci, cell in enumerate(cells):
        cut = bind_cut_score(model, packL["ids"], packL["metas"], pl, L, H, omega,
                             [cell], packL["U1"])
        c_eff.append(W.removal(base_loc, cut))
        if ci % 50 == 49:
            print(f"[T2-causal/{mkey}] {ci+1}/{len(cells)} {time.time()-t0:.0f}s", flush=True)
    sp = W.spearman(s_val, c_eff)
    order_s = np.argsort(-np.array(s_val))
    order_c = np.argsort(-np.array(c_eff))
    rec = {}
    for k in (1, 3, 10):
        cs = [cells[i] for i in order_s[:k]]
        cc = [cells[i] for i in order_c[:k]]
        rem_s = W.removal(base_hold, bind_cut_score(model, packH["ids"], packH["metas"], ph,
                                                    L, H, omega, cs, packH["U1"]))
        rem_c = W.removal(base_hold, bind_cut_score(model, packH["ids"], packH["metas"], ph,
                                                    L, H, omega, cc, packH["U1"]))
        rec[f"k{k}"] = {"rem_fra": rem_s, "rem_causal": rem_c,
                        "recovery": rem_s / rem_c if abs(rem_c) > 1e-6 else None}
    # oracle: mask edge (all completion query positions -> init-value key) on holdout rows
    Bh, Th = packH["ids"].shape[0], packH["ids"].shape[1]
    fixed = torch.zeros(Bh, Th, Th, device=DEV)
    for r, (nb, nc, _) in enumerate(packH["metas"]):
        kp = packH["kps"][r // 2]
        fixed[r, nb - 1 : nb + nc - 1, kp] = -1e9
    cut_mask = bind_cut_score(model, packH["ids"], packH["metas"], ph, L, H, omega,
                              None, None, fixed=fixed)
    orc = W.removal(base_hold, cut_mask)
    return {"base_locate": base_loc, "base_holdout": base_hold,
            "n_cells_tested": len(cells), "spearman_s_c": sp, "recovery": rec,
            "oracle_edge_mask_rem": orc,
            "top_causal_cells": [[list(cells[i]), float(c_eff[i])] for i in order_c[:5]],
            "top_fra_cells": [[list(cells[i]), float(s_val[i])] for i in order_s[:5]],
            "cells_order_causal_top10": [list(cells[i]) for i in order_c[:10]],
            "cells_order_fra_top10": [list(cells[i]) for i in order_s[:10]]}


def selectivity_block(model, config, tok, pairs_loc, pairs_hold, L, H, omega, mkey):
    """cut cells located for ONE binding type; measure self-removal vs sibling collateral."""
    out = {}
    for direction in ("set", "str"):
        i = 0 if direction == "set" else 1
        j = 1 - i
        self_loc = [p[i] for p in pairs_loc]
        self_hold = [p[i] for p in pairs_hold]
        sib_hold = [p[j] for p in pairs_hold]
        packL = bind_fra_pack(model, config, tok, self_loc, L, H, omega)
        packH = bind_fra_pack(model, config, tok, self_hold, L, H, omega)
        packS = bind_fra_pack(model, config, tok, sib_hold, L, H, omega)
        cells, s_val = rank_cells(packL["Ms"], N_CTOP // 2, N_CRAND // 2, SEED + 11)
        base_loc = bind_clean_score(model, packL["ids"], packL["metas"], self_loc)
        base_hold = bind_clean_score(model, packH["ids"], packH["metas"], self_hold)
        base_sib = bind_clean_score(model, packS["ids"], packS["metas"], sib_hold)
        c_eff = []
        for cell in cells:
            cut = bind_cut_score(model, packL["ids"], packL["metas"], self_loc, L, H, omega,
                                 [cell], packL["U1"])
            c_eff.append(W.removal(base_loc, cut))
        order_c = np.argsort(-np.array(c_eff))
        order_s = np.argsort(-np.array(s_val))
        drec = {"base_locate": base_loc, "base_holdout": base_hold, "base_sibling": base_sib,
                "spearman_s_c": W.spearman(s_val, c_eff)}
        for rank_name, order in (("causal", order_c), ("fra", order_s)):
            top10 = [cells[ii] for ii in order[:10]]
            cut_self = bind_cut_score(model, packH["ids"], packH["metas"], self_hold,
                                      L, H, omega, top10, packH["U1"])
            cut_sib = bind_cut_score(model, packS["ids"], packS["metas"], sib_hold,
                                     L, H, omega, top10, packS["U1"])
            rem_self = W.removal(base_hold, cut_self)
            rem_sib = W.removal(base_sib, cut_sib)
            drec[f"top10_{rank_name}"] = {
                "rem_self": rem_self, "rem_sibling": rem_sib,
                "selectivity_ratio": rem_self / rem_sib if abs(rem_sib) > 1e-6 else None,
                "cells": [list(c) for c in top10]}
        # drift across instances for this binding type
        cm = W.cell_metrics(packL["Ms"] + packH["Ms"])
        cm["edge_cosine"] = W.edge_matrix_cosine(packL["Ms"] + packH["Ms"])
        cm["edge_subspace_pr"] = edge_pr(packL["Ms"] + packH["Ms"])
        cm["fra_recon_r2"] = packL["r2"]
        drec["drift"] = cm
        out[direction] = drec
        print(f"[T2-select/{mkey}/{direction}] self={drec['top10_causal']['rem_self']:.3f} "
              f"sib={drec['top10_causal']['rem_sibling']:.3f}", flush=True)
    return out


def stage_T2(out):
    res = out.setdefault("T2", {"config": {"N_BIND_GATE": N_BIND_GATE, "gate": GATE,
                                           "seed": SEED}, "sweep": {}, "models": {}})
    tok = load_tok()
    rng = random.Random(SEED + 3)
    gate_prompts = W.gen_bind_prompts(N_BIND_GATE, rng)

    def gate_model(name):
        if name in res["sweep"]:
            return res["sweep"][name]["acc"]
        try:
            model, config = W.load_ws_model(name)
        except Exception as e:
            res["sweep"][name] = {"acc": None, "error": str(e)[:200]}
            ckpt(out)
            return None
        sc = W.bind_scores(model, tok, gate_prompts)
        acc = float((sc > 0.5).mean())
        res["sweep"][name] = {"acc": acc, "mean_score": float(sc.mean()),
                              "arch": {"n_layer": config.n_layer, "n_head": config.n_head,
                                       "d_model": config.d_model, "afrac": config.afrac}}
        print(f"[T2-sweep] {name} acc={acc:.3f}", flush=True)
        ckpt(out)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
        return acc

    sweep = SWEEP_AF025[:1] if SMOKE else SWEEP_AF025
    passing = None
    for name in sweep:
        acc = gate_model(name)
        if acc is not None and acc >= GATE and "afrac0.250" in name and name != SWEEP_AF025[0]:
            passing = name
            break
        if acc is not None and acc >= GATE and name == SWEEP_AF025[0]:
            passing = name  # would contradict B1; accept but it will be visible in the table
            break
    if passing is None and not SMOKE:
        for name in SWEEP_AF05:
            acc = gate_model(name)
            if acc is not None and acc >= GATE:
                passing = name
                break
    res["passing_sparse"] = passing
    ckpt(out)
    if passing is None:
        res["verdict_hint"] = "BLOCKED-BY-CAPABILITY: no released sparse model passes the binding gate"
        ckpt(out)
        return

    twin = passing.replace("afrac0.250", "afrac1.000").replace("afrac0.500", "afrac1.000")
    densew = DENSE_BY_EF[ef_of(passing)]
    gate_model(twin)
    gate_model(densew)
    full_models = [("sparse_pass", passing), ("wsda_twin", twin), ("dense_w", densew)]
    res["full_models"] = {k: v for k, v in full_models}

    bind_all = W.gen_bind_prompts(N_BIND_FRA, random.Random(SEED + 4))
    b_loc, b_hold = bind_all[: N_BIND_FRA // 2], bind_all[N_BIND_FRA // 2 :]
    pairs = gen_bind_pairs(N_PAIRS, random.Random(SEED + 5))
    pairs_loc, pairs_hold = pairs[: N_PAIRS // 2], pairs[N_PAIRS // 2 :]

    for mkey, name in full_models:
        if mkey in res["models"] and "selectivity" in res["models"][mkey]:
            continue  # resume skip
        if res["sweep"].get(name, {}).get("acc") is None:
            continue
        model, config = W.load_ws_model(name)
        mrec = res["models"].setdefault(mkey, {"name": name,
                                               "gate_acc": res["sweep"][name]["acc"]})
        gate_ok = res["sweep"][name]["acc"] >= GATE
        mrec["gate_ok"] = gate_ok
        n_hf = HEADFIND_N if config.n_layer * config.n_head <= 256 else max(24, HEADFIND_N // 2)
        base, ranked = W.find_top_head(model, config, tok, b_loc[:n_hf], W.bind_scores)
        baseE, rankedE = t2_edge_oracle_scan(model, config, tok, b_loc[:n_hf])
        (L, H), erem = rankedE[0]   # pick the EDGE head, not the ablation-drop head
        mrec["bind_head"] = {"layer": L, "head": H, "edge_oracle_probe": erem, "base": baseE,
                             "ablation_top5": [[list(k), float(v)] for k, v in ranked[:5]],
                             "edge_oracle_top8": [[list(k), float(v)] for k, v in rankedE[:8]]}
        print(f"[T2/{mkey}] {name} edge head L{L}H{H} edge_rem={erem:.3f} "
              f"(ablation argmax {ranked[0][0]} drop={ranked[0][1]:.3f})", flush=True)
        dh = config.d_head
        scale = 1.0 / math.sqrt(dh)
        Wq, Wk, bq, bk = W.head_qk_weights(model, config, L, H)
        I = torch.eye(config.d_model, device=DEV)
        omega = W.omega_from_decoder(I, I, Wq, Wk, bq, bk, scale)
        packL = bind_fra_pack(model, config, tok, b_loc, L, H, omega)
        packH = bind_fra_pack(model, config, tok, b_hold, L, H, omega)
        cm = W.cell_metrics(packL["Ms"] + packH["Ms"])
        cm["fra_recon_r2_locate"] = packL["r2"]
        cm["fra_recon_r2_holdout"] = packH["r2"]
        cm["edge_cosine"] = W.edge_matrix_cosine(packL["Ms"] + packH["Ms"])
        cm["edge_subspace_pr"] = edge_pr(packL["Ms"] + packH["Ms"])
        mrec["fra"] = cm
        ckpt(out)
        mrec["causal"] = bind_causal_block(model, tok, packL, packH, b_loc, b_hold,
                                           L, H, omega, mkey)
        ckpt(out)
        mrec["selectivity"] = selectivity_block(model, config, tok, pairs_loc, pairs_hold,
                                                L, H, omega, mkey)
        ckpt(out)
        del model, packL, packH
        if DEV == "cuda":
            torch.cuda.empty_cache()


# ---------------- main ----------------
def main():
    out = {"meta": {"seed": SEED, "smoke": SMOKE, "stages": STAGES,
                    "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}}
    # resume: pull prior partial if it exists
    prev = os.path.join(OUT_DIR, "ind_bind_partial.json")
    if not SMOKE and W.try_download("induction_binding/ind_bind_partial.json", prev):
        try:
            with open(prev) as f:
                got = json.load(f)
            if got.get("meta", {}).get("seed") == SEED and not got.get("meta", {}).get("smoke"):
                out.update({k: v for k, v in got.items() if k in ("T1", "T2")})
                print("[main] resumed from prior partial", flush=True)
        except Exception as e:
            print(f"[main] resume parse failed: {e}", flush=True)
    t0 = time.time()
    if "T1" in STAGES:
        stage_T1(out)
        print(f"[main] T1 done {time.time()-t0:.0f}s", flush=True)
    if "T2" in STAGES:
        stage_T2(out)
        print(f"[main] T2 done {time.time()-t0:.0f}s", flush=True)
    p = os.path.join(OUT_DIR, "ind_bind_final.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    upload2(p)
    print("[main] ALL DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        tp = os.path.join(OUT_DIR, "ind_bind_traceback.txt")
        with open(tp, "w") as f:
            f.write(tb)
        upload2(tp)
        raise

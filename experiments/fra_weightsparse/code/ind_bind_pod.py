#!/usr/bin/env python3
"""Candidates re-do pod (rs-ws2-*): T1 identifier-induction + T2 binding-up-the-ladder.

T1 IDENTIFIER-INDUCTION (the gpt2 persistence/drift re-do on the weight-sparse ladder):
  code-native induction — identifier `base_suf` bound early, prompt ends mid-reuse at the
  canonical token boundary before A's final token; target = A's final token (verified to be
  the identical token id at the definition = k_pos); distractor = B's final definition token
  (2AFC). Gate = full-vocab argmax acc >= 0.70 (prereg), variant ladder v0 -> v1 -> v2 -> 2x.

  SMOKE FINDING baked in: these models carry induction in a REDUNDANT HEAD BANK (~12 heads
  across 2 layers attend q_last->k_target; best single-head edge-mask removal 0.06). So the
  suite is bank-based: bank = top heads by mean att(q_last->k_target); oracle = edge mask in
  ALL heads (and in the bank); FRA cells are (head, qF, kF) triples; cuts are grouped per head
  via a multi-(layer,head) attention patch. Drift metrics at bank level (dominant triple) +
  per-head for the primary (max-att) head.

T2 BINDING UP THE LADDER: gate-sweep csp_sweep1_{EF}x_{NZ}nonzero_afrac0.250 ascending
  (EF, NZ) on set_or_string (score-acc >= 0.70); smallest passing sparse + afrac1.000 twin +
  width-matched dense1_{EF}x. Then the same bank-based FRA/causal suite on the binding edge
  (rows are teacher-forced; U1 cached ON the rows) + SIBLING selectivity (cut set-binding
  cells, measure str-binding collateral in the same contexts) + drift across instances.

Reuses ws_pod.py pure compute (loader, FRA weights/decomposition, cell metrics, removal).
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
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_pod as W  # noqa: E402  (module-level: env config, seeds, OUT_DIR mkdir)
from csp_vendor_hook_utils import hook_recorder  # noqa: E402

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
HEADFIND_N = int(os.environ.get("HEADFIND_N", "24" if SMOKE else "64"))
BANK_THRESH = float(os.environ.get("BANK_THRESH", "0.25"))
BANK_CAP = int(os.environ.get("BANK_CAP", "8"))

if SMOKE:
    W.N_CAUSAL_TOP, W.N_CAUSAL_RAND = 24, 12
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


# ---------------- multi-(layer,head) attention patch ----------------
class _MultiSpec:
    def __init__(self):
        self.by_layer = {}


MS = _MultiSpec()


def make_fwd_multi(L):
    def fwd(self, x):
        x = self.config.maybe_activation_sparsity(x, "attn_in")
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_head * self.d_head, dim=2)
        k = self.config.maybe_activation_sparsity(k, "attn_k")
        q = self.config.maybe_activation_sparsity(q, "attn_q")
        v = self.config.maybe_activation_sparsity(v, "attn_v")
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale
        for (H, fn, fixed) in MS.by_layer.get(L, ()):
            if fn is not None:
                att[:, H] = att[:, H] + fn(x)
            elif fixed is not None:
                att[:, H] = att[:, H] + fixed[:, :T, :T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
        y = self.resid_dropout(self.c_proj(y))
        y = self.config.maybe_activation_sparsity(y, "attn_out")
        return y
    return fwd


class PatchedMulti:
    """specs: list of (L, H, fn, fixed); fn(x)->[B,T,T] delta or fixed [B,T,T] tensor."""

    def __init__(self, model, specs):
        self.model = model
        self.by_layer = {}
        for L, H, fn, fixed in specs:
            self.by_layer.setdefault(L, []).append((H, fn, fixed))

    def __enter__(self):
        MS.by_layer = self.by_layer
        self.orig = {}
        for L in self.by_layer:
            attn = self.model.transformer.h[L].attn
            self.orig[L] = attn.forward
            attn.forward = types.MethodType(make_fwd_multi(L), attn)
        return self

    def __exit__(self, *a):
        for L, f in self.orig.items():
            self.model.transformer.h[L].attn.forward = f
        MS.by_layer = {}


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

    Cuts at the canonical token boundary before A's final token (on-distribution).
    target = A's final token, verified identical (span string + id) at the definition.
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
        dc = text.index(f"{A} = ")
        d_start, d_end = dc + len(A) - len(target_str), dc + len(A)
        k_pos = None
        for i, (a, b) in enumerate(offs):
            if a == d_start and b == d_end:
                k_pos = i
                break
        if k_pos is None or ids[k_pos] != target_id:
            continue
        dB = text.index(f"{B} = ")
        distractor_id = None
        for i, (a, b) in enumerate(offs):
            if b == dB + len(B):
                distractor_id = ids[i]
                break
        if distractor_id is None or distractor_id == target_id:
            continue
        return {"text": text[:a0], "ids": list(ids[:t_last]), "target_id": int(target_id),
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
    """2AFC P(target)/(P(target)+P(distractor)) at the last position."""
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


# ---------------- bank selection + bank metrics (shared T1/T2) ----------------
@torch.no_grad()
def head_att_bank(model, config, ids, lens, qps, kps, thresh=BANK_THRESH, cap=BANK_CAP):
    """mean attention(q=qps[b] -> k=kps[b]) per head; bank = top heads above thresh."""
    with hook_recorder(regex=r"^\d+\.attn\.(q|k)$") as rec:
        model(ids)
    dh = config.d_head
    B = ids.shape[0]
    scores = {}
    for L in range(config.n_layer):
        q = rec[f"{L}.attn.q"].float()
        k = rec[f"{L}.attn.k"].float()
        for H in range(config.n_head):
            qh = q[:, :, H * dh : (H + 1) * dh]
            kh = k[:, :, H * dh : (H + 1) * dh]
            accs = []
            for b in range(B):
                T = int(lens[b])
                a = torch.softmax(kh[b, :T] @ qh[b, qps[b]] / math.sqrt(dh), 0)
                accs.append(float(a[kps[b]]))
            scores[(L, H)] = float(np.mean(accs))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    bank = [hk for hk, v in ranked if v >= thresh][:cap]
    if not bank:
        bank = [ranked[0][0]]
    return bank, ranked


def edge_mask_specs(heads, fixed):
    return [(L, H, None, fixed) for (L, H) in heads]


def all_heads(config):
    return [(L, H) for L in range(config.n_layer) for H in range(config.n_head)]


def bank_metrics(Ms_by_head, bank):
    """bank-level persistence metrics over (head, qF, kF) triples."""
    n = len(Ms_by_head[bank[0]])
    from collections import Counter
    doms, mass1, mass8, mass32 = [], [], [], []
    for i in range(n):
        tot, best, bestv = 0.0, None, -1.0
        tops = []
        for hk in bank:
            a = Ms_by_head[hk][i].abs()
            tot += float(a.sum())
            v, idx = a.flatten().topk(min(32, a.numel()))
            tops.append(v)
            if float(v[0]) > bestv:
                bestv = float(v[0])
                Fp = a.shape[0]
                best = (hk, int(idx[0] // Fp), int(idx[0] % Fp))
        doms.append(best)
        merged = torch.cat(tops).sort(descending=True).values
        tot = max(tot, 1e-12)
        mass1.append(float(merged[:1].sum()) / tot)
        mass8.append(float(merged[:8].sum()) / tot)
        mass32.append(float(merged[:32].sum()) / tot)
    cnt = Counter(doms)
    ranked = cnt.most_common()
    cum, k90 = 0, 0
    for _, c in ranked:
        cum += c
        k90 += 1
        if cum >= 0.9 * n:
            break
    qcnt = Counter([d[1] for d in doms])
    return {"n": n, "bank": [list(hk) for hk in bank],
            "top1_triple": [list(ranked[0][0][0]), ranked[0][0][1], ranked[0][0][2]],
            "top1_coverage": ranked[0][1] / n,
            "cov3": sum(c for _, c in ranked[:3]) / n,
            "n_cells_for_90": k90, "n_distinct_dom": len(ranked),
            "q_dom_modal_cov": qcnt.most_common(1)[0][1] / n, "q_dom_distinct": len(qcnt),
            "edge_mass_top1": float(np.mean(mass1)),
            "edge_mass_top8": float(np.mean(mass8)),
            "edge_mass_top32": float(np.mean(mass32)),
            "triple_hist_top10": [[[list(t[0]), t[1], t[2]], int(c)] for t, c in ranked[:10]]}


def bank_edge_cosine(Ms_by_head, bank, n_pairs=200):
    n = len(Ms_by_head[bank[0]])
    rng = random.Random(2)
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(n_pairs)]
    cs = []
    for i, j in pairs:
        if i == j:
            continue
        dot, ni, nj = 0.0, 0.0, 0.0
        for hk in bank:
            a, b = Ms_by_head[hk][i].flatten().float(), Ms_by_head[hk][j].flatten().float()
            dot += float(a @ b)
            ni += float(a @ a)
            nj += float(b @ b)
        cs.append(dot / max(math.sqrt(ni * nj), 1e-12))
    return float(np.mean(cs))


def bank_edge_pr(Ms_by_head, bank, topc=512):
    """participation ratio of the cross-context edge ensemble over top (head,cell) coords."""
    n = len(Ms_by_head[bank[0]])
    sel = []
    for hk in bank:
        macc = torch.zeros(Ms_by_head[hk][0].numel())
        for m in Ms_by_head[hk]:
            macc += m.abs().flatten().float()
        v, idx = macc.topk(min(topc, macc.numel()))
        sel.append((hk, idx))
    cols = []
    for hk, idx in sel:
        Xh = torch.stack([m.flatten()[idx].float() for m in Ms_by_head[hk]])
        cols.append(Xh)
    X = torch.cat(cols, dim=1)
    mag = X.abs().mean(0)
    keep = mag.topk(min(topc, X.shape[1])).indices
    X = X[:, keep]
    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    ev = torch.linalg.eigvalsh(X @ X.T).clamp(min=0)
    return float(ev.sum() ** 2 / max(float((ev ** 2).sum()), 1e-12))


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


def bank_candidates(Ms_by_head, bank, n_top, n_rand, seed):
    """pooled (head, cell, fra_score) candidates, budget split across bank heads."""
    per_t = max(2, n_top // len(bank))
    per_r = max(1, n_rand // len(bank))
    cands = []
    for hi, hk in enumerate(bank):
        cells, s_val = rank_cells(Ms_by_head[hk], per_t, per_r, seed + hi)
        cands += [(hk, c, s) for c, s in zip(cells, s_val)]
    return cands


def group_specs(cand_subset, omegas, U1_by_layer):
    """group cells by head -> PatchedMulti specs with per-head cells_delta_fn."""
    by_head = {}
    for hk, cell, _ in cand_subset:
        by_head.setdefault(hk, []).append(cell)
    specs = []
    for (L, H), cells in by_head.items():
        specs.append((L, H, W.cells_delta_fn(cells, U1_by_layer[L], omegas[(L, H)]), None))
    return specs


def causal_suite(mkey, tag, bank, omegas, Ms_l, U1_loc, U1_hold, cut_loc, cut_hold,
                 fixed_loc, orc_bank, orc_all):
    """generic bank causal suite.

    cut_loc/cut_hold: callables(specs)->mean task score on locate/holdout.
    Redundant banks make RAW per-cell effects ~0 (any one head is backed up), so we also
    measure ISOLATED-path effects: probe each cell with the OTHER bank heads' edge masked.
    Headline localization = per-head FRA-top-k union curve on holdout vs the bank oracle.
    """
    base_loc, base_hold = cut_loc([]), cut_hold([])
    zval = cut_loc([(L, H, None, None) for (L, H) in bank])
    assert abs(zval - base_loc) < 1e-3, f"patched-forward mismatch {zval} vs {base_loc}"
    cands = bank_candidates(Ms_l, bank, N_CTOP, N_CRAND, SEED + 7)
    s_val = [c[2] for c in cands]
    base_iso = ({hk: cut_loc(edge_mask_specs([h for h in bank if h != hk], fixed_loc))
                 for hk in bank} if len(bank) > 1 else {bank[0]: base_loc})
    c_raw, c_iso = [], []
    t0 = time.time()
    for ci, (hk, cell, s) in enumerate(cands):
        cs = group_specs([(hk, cell, s)], omegas, U1_loc)
        c_raw.append(W.removal(base_loc, cut_loc(cs)))
        others = [h for h in bank if h != hk]
        mask_specs = edge_mask_specs(others, fixed_loc) if others else []
        c_iso.append(W.removal(base_iso[hk], cut_loc(mask_specs + cs)))
        if ci % 50 == 49:
            print(f"[{tag}-causal/{mkey}] {ci+1}/{len(cands)} {time.time()-t0:.0f}s", flush=True)
    res = {"base_locate": base_loc, "base_holdout": base_hold,
           "n_cells_tested": len(cands),
           "spearman_s_craw": W.spearman(s_val, c_raw),
           "spearman_s_ciso": W.spearman(s_val, c_iso),
           "mean_abs_craw": float(np.mean(np.abs(c_raw))),
           "mean_abs_ciso": float(np.mean(np.abs(c_iso))),
           "base_iso": {f"L{L}H{H}": v for (L, H), v in base_iso.items()},
           "oracle_bank_holdout": orc_bank, "oracle_all_holdout": orc_all}
    # per-head union curves on holdout (FRA-ranked vs isolated-causal-ranked)
    by_head_s = {hk: sorted([i for i, c in enumerate(cands) if c[0] == hk],
                            key=lambda i: -s_val[i]) for hk in bank}
    by_head_c = {hk: sorted([i for i, c in enumerate(cands) if c[0] == hk],
                            key=lambda i: -c_iso[i]) for hk in bank}
    curve = {}
    for k in (1, 2, 4, 8, 16, 32):
        row = {}
        for nm, bh in (("fra", by_head_s), ("ciso", by_head_c)):
            sel = [cands[i] for hk in bank for i in bh[hk][:k]]
            rem = W.removal(base_hold, cut_hold(group_specs(sel, omegas, U1_hold)))
            row[nm] = {"rem": rem, "n_cells": len(sel),
                       "frac_oracle_bank": rem / orc_bank if abs(orc_bank) > 1e-6 else None}
        row["recovery_fra_vs_ciso"] = (row["fra"]["rem"] / row["ciso"]["rem"]
                                       if abs(row["ciso"]["rem"]) > 1e-6 else None)
        curve[f"k{k}"] = row
    res["union_curve_perhead"] = curve
    oc = np.argsort(-np.array(c_iso))
    osr = np.argsort(-np.array(s_val))
    res["top_ciso"] = [[list(cands[i][0]), list(cands[i][1]), float(c_iso[i])] for i in oc[:5]]
    res["top_fra"] = [[list(cands[i][0]), list(cands[i][1]), float(s_val[i])] for i in osr[:5]]
    return res


# ---------------- T1 FRA + causal (bank-based) ----------------
@torch.no_grad()
def collect_U1(model, config, layers, ids):
    out = {}
    for L in layers:
        U, qh, kh = W.collect_act_in(model, config, L, ids)
        out[L] = (W.append_bias_channel(U), qh, kh)
    return out


@torch.no_grad()
def t1_cut_score(model, tok, prompts, specs):
    with PatchedMulti(model, specs):
        s = float(ind_scores(model, tok, prompts, batch=len(prompts)).mean())
    return s


def t1_fixed_mask(prompts, lens):
    B, T = len(prompts), int(lens.max())
    fixed = torch.zeros(B, T, T, device=DEV)
    for b, p in enumerate(prompts):
        fixed[b, int(lens[b]) - 1, p["k_pos"]] = -1e9
    return fixed


def t1_bank_block(model, config, tok, loc, hold, mkey, mrec, out):
    ids_l, lens_l = W.pad_batch([p["ids"] for p in loc])
    ids_h, lens_h = W.pad_batch([p["ids"] for p in hold])
    qps_l = [int(x) - 1 for x in lens_l]
    qps_h = [int(x) - 1 for x in lens_h]
    kps_l = [p["k_pos"] for p in loc]
    kps_h = [p["k_pos"] for p in hold]

    # bank by attention mass on the locate probe
    nprobe = min(HEADFIND_N, len(loc))
    bank, att_ranked = head_att_bank(model, config, ids_l[:nprobe], lens_l[:nprobe],
                                     qps_l[:nprobe], kps_l[:nprobe])
    primary = bank[0]
    mrec["bank"] = {"heads": [list(h) for h in bank],
                    "att_top12": [[list(k), float(v)] for k, v in att_ranked[:12]]}
    base_hold = float(ind_scores(model, tok, hold, batch=len(hold)).mean())
    fixed_h = t1_fixed_mask(hold, lens_h)
    orc_all = W.removal(base_hold, t1_cut_score(model, tok, hold,
                                                edge_mask_specs(all_heads(config), fixed_h)))
    orc_bank = W.removal(base_hold, t1_cut_score(model, tok, hold,
                                                 edge_mask_specs(bank, fixed_h)))
    mrec["oracle"] = {"edge_mask_all_heads": orc_all, "edge_mask_bank": orc_bank,
                      "base_holdout": base_hold}
    print(f"[T1/{mkey}] bank={bank} oracle_all={orc_all:.3f} oracle_bank={orc_bank:.3f}",
          flush=True)

    # FRA: per-head omegas + per-layer U1; edge matrices per bank head
    layers = sorted({L for L, H in bank})
    U1l = {L: v[0] for L, v in collect_U1(model, config, layers, ids_l).items()}
    U1h_full = collect_U1(model, config, layers, ids_h)
    U1h = {L: v[0] for L, v in U1h_full.items()}
    I = torch.eye(config.d_model, device=DEV)
    scale = 1.0 / math.sqrt(config.d_head)
    omegas = {}
    for (L, H) in bank:
        Wq, Wk, bq, bk = W.head_qk_weights(model, config, L, H)
        omegas[(L, H)] = W.omega_from_decoder(I, I, Wq, Wk, bq, bk, scale)
    Ms_l = {hk: [W.edge_cell_matrix(U1l[hk[0]][b], omegas[hk], qps_l[b], kps_l[b]).cpu()
                 for b in range(len(loc))] for hk in bank}
    Ms_h = {hk: [W.edge_cell_matrix(U1h[hk[0]][b], omegas[hk], qps_h[b], kps_h[b]).cpu()
                 for b in range(len(hold))] for hk in bank}
    Ms_all = {hk: Ms_l[hk] + Ms_h[hk] for hk in bank}

    # recon R2 for the primary head (exactness check)
    dh = config.d_head
    Lp, Hp = primary
    qh = U1h_full[Lp][1][..., Hp * dh : (Hp + 1) * dh]
    kh = U1h_full[Lp][2][..., Hp * dh : (Hp + 1) * dh]
    r2 = W.fra_recon_r2(U1h[Lp], qh, kh, omegas[primary], lens_h)

    bm = bank_metrics(Ms_all, bank)
    bm["edge_cosine"] = bank_edge_cosine(Ms_all, bank)
    bm["edge_subspace_pr"] = bank_edge_pr(Ms_all, bank)
    bm["fra_recon_r2_primary"] = r2
    bm["primary_head_metrics"] = W.cell_metrics(Ms_all[primary])
    bm["per_head_top1cov"] = {f"L{L}H{H}": W.cell_metrics(Ms_all[(L, H)])["top1_coverage"]
                              for (L, H) in bank}
    bm["cofire_qside_primaryL"] = W.cofire_set_drift(
        [U1l[Lp][b].cpu() for b in range(len(loc))], qps_l)
    mrec["fra"] = bm
    ckpt(out)

    # causal: shared bank suite (raw + isolated per-cell, per-head union curve)
    fixed_l = t1_fixed_mask(loc, lens_l)
    mrec["causal"] = causal_suite(
        mkey, "T1", bank, omegas, Ms_l, U1l, U1h,
        lambda specs: t1_cut_score(model, tok, loc, specs),
        lambda specs: t1_cut_score(model, tok, hold, specs),
        fixed_l, orc_bank, orc_all)
    ckpt(out)


def stage_T1(out):
    res = out.setdefault("T1", {"config": {"N_IND": N_IND, "gate": GATE, "seed": SEED,
                                           "bank_thresh": BANK_THRESH, "bank_cap": BANK_CAP},
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
        for mkey, name in [("sparse2x", "csp_sweep1_2x_3.7Mnonzero_afrac0.250"),
                           ("wsda2x", "csp_sweep1_2x_3.7Mnonzero_afrac1.000"),
                           ("dense2x", "dense1_2x")]:
            run_t1_model(res, tok, prompt_sets, mkey, name, out)
    ckpt(out)


def run_t1_model(res, tok, prompt_sets, mkey, name, out):
    if mkey in res["models"] and ("causal" in res["models"][mkey]
                                  or res["models"][mkey].get("gate_pass", "x") is None):
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
    mrec["ablation_top5"] = [[list(k), float(v)] for k, v in ranked[:5]]
    t1_bank_block(model, config, tok, loc, hold, mkey, mrec, out)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()


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
EF_WIDTH = {"1x": 256, "2x": 512, "4x": 1024, "8x": 2048}


def ef_of(name):
    for ef in ("16x", "1x", "2x", "4x", "8x"):
        if f"_{ef}_" in name or name.endswith(f"_{ef}"):
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


COMPS = {True: ".add(", False: " += "}


def bind_pack(model, config, tok, prompts, layers=()):
    """teacher-forced rows + (optionally) per-layer U1 on the rows + positions."""
    rows, metas = [], []
    for p in prompts:
        base = tok.encode(p["text"]).ids
        for is_set in (True, False):
            comp = tok.encode(COMPS[is_set]).ids
            rows.append(base + comp)
            metas.append((len(base), len(comp), is_set))
    ids, lens = W.pad_batch(rows)
    kps = [W.find_kpos_bind(tok, p)[1] for p in prompts]
    qps = [metas[2 * j][0] - 1 for j in range(len(prompts))]
    pack = {"ids": ids, "lens": lens, "metas": metas, "kps": kps, "qps": qps,
            "prompts": prompts}
    if layers:
        pack["U1"] = {L: v[0] for L, v in collect_U1(model, config, layers, ids).items()}
    return pack


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
def bind_cut(model, pack, specs):
    """mean bind score under PatchedMulti specs (empty fn/fixed = clean patched)."""
    with PatchedMulti(model, specs):
        logits, _, _ = model(pack["ids"])
    return float(bind_scores_from_logits(logits, pack["ids"], pack["metas"],
                                         pack["prompts"]).mean())


@torch.no_grad()
def bind_clean(model, pack):
    logits, _, _ = model(pack["ids"])
    return float(bind_scores_from_logits(logits, pack["ids"], pack["metas"],
                                         pack["prompts"]).mean())


def bind_fixed_mask(pack):
    B, T = pack["ids"].shape
    fixed = torch.zeros(B, T, T, device=DEV)
    for r, (nb, nc, _) in enumerate(pack["metas"]):
        fixed[r, nb - 1 : nb + nc - 1, pack["kps"][r // 2]] = -1e9
    return fixed


def bind_edges(pack, bank, omegas):
    """per-bank-head edge matrices (q=last prompt token, k=init-value token) per prompt."""
    Ms = {}
    for hk in bank:
        L = hk[0]
        Ms[hk] = [W.edge_cell_matrix(pack["U1"][L][2 * j], omegas[hk],
                                     pack["qps"][j], pack["kps"][j]).cpu()
                  for j in range(len(pack["prompts"]))]
    return Ms


def t2_full_block(model, config, tok, name, mkey, mrec, b_loc, b_hold, pairs_loc,
                  pairs_hold, out, res):
    nprobe = min(32, len(b_loc))
    probe = bind_pack(model, config, tok, b_loc[:nprobe])
    nrows = probe["ids"].shape[0]
    qps_row = [probe["qps"][r // 2] for r in range(nrows)]
    kps_row = [probe["kps"][r // 2] for r in range(nrows)]
    bank, att_ranked = head_att_bank(model, config, probe["ids"], probe["lens"],
                                     qps_row, kps_row,
                                     cap=(BANK_CAP if config.d_model <= 512 else 4))
    mrec["bank"] = {"heads": [list(h) for h in bank],
                    "att_top12": [[list(k), float(v)] for k, v in att_ranked[:12]]}
    base, ranked = W.find_top_head(model, config, tok, b_loc[:nprobe], W.bind_scores)
    mrec["ablation_top5"] = [[list(k), float(v)] for k, v in ranked[:5]]

    layers = sorted({L for L, H in bank})
    packL = bind_pack(model, config, tok, b_loc, layers)
    packH = bind_pack(model, config, tok, b_hold, layers)
    I = torch.eye(config.d_model, device=DEV)
    scale = 1.0 / math.sqrt(config.d_head)
    omegas = {}
    for (L, H) in bank:
        Wq, Wk, bq, bk = W.head_qk_weights(model, config, L, H)
        omegas[(L, H)] = W.omega_from_decoder(I, I, Wq, Wk, bq, bk, scale)
    Ms_l = bind_edges(packL, bank, omegas)
    Ms_h = bind_edges(packH, bank, omegas)
    Ms_all = {hk: Ms_l[hk] + Ms_h[hk] for hk in bank}

    bm = bank_metrics(Ms_all, bank)
    bm["edge_cosine"] = bank_edge_cosine(Ms_all, bank)
    bm["edge_subspace_pr"] = bank_edge_pr(Ms_all, bank)
    bm["primary_head_metrics"] = W.cell_metrics(Ms_all[bank[0]])
    bm["per_head_top1cov"] = {f"L{L}H{H}": W.cell_metrics(Ms_all[(L, H)])["top1_coverage"]
                              for (L, H) in bank}
    mrec["fra"] = bm
    ckpt(out)

    base_hold = bind_clean(model, packH)
    fixed_h = bind_fixed_mask(packH)
    orc_all = W.removal(base_hold, bind_cut(model, packH,
                                            edge_mask_specs(all_heads(config), fixed_h)))
    orc_bank = W.removal(base_hold, bind_cut(model, packH, edge_mask_specs(bank, fixed_h)))
    mrec["oracle"] = {"edge_mask_all_heads": orc_all, "edge_mask_bank": orc_bank}
    print(f"[T2/{mkey}] bank={bank} oracle_all={orc_all:.3f} oracle_bank={orc_bank:.3f}",
          flush=True)

    fixed_l = bind_fixed_mask(packL)
    mrec["causal"] = causal_suite(
        mkey, "T2", bank, omegas, Ms_l, packL["U1"], packH["U1"],
        lambda specs: bind_cut(model, packL, specs),
        lambda specs: bind_cut(model, packH, specs),
        fixed_l, orc_bank, orc_all)
    ckpt(out)

    # ---- sibling selectivity (cut one binding type's cells; measure the other's collateral)
    sel = {}
    for direction in ("set", "str"):
        i = 0 if direction == "set" else 1
        j = 1 - i
        pkL = bind_pack(model, config, tok, [p[i] for p in pairs_loc], layers)
        pkH = bind_pack(model, config, tok, [p[i] for p in pairs_hold], layers)
        pkS = bind_pack(model, config, tok, [p[j] for p in pairs_hold], layers)
        MsL = bind_edges(pkL, bank, omegas)
        bL, bH, bS = bind_clean(model, pkL), bind_clean(model, pkH), bind_clean(model, pkS)
        # FRA-ranked per-head unions, position-invariant cut located on THIS direction only
        cands_d = bank_candidates(MsL, bank, N_CTOP // 2, 0, SEED + 11)
        sv = [c[2] for c in cands_d]
        by_head = {hk: sorted([ii for ii, c in enumerate(cands_d) if c[0] == hk],
                              key=lambda ii: -sv[ii]) for hk in bank}
        drec = {"base_locate": bL, "base_holdout": bH, "base_sibling": bS}
        for k in (4, 10):
            top = [cands_d[ii] for hk in bank for ii in by_head[hk][:k]]
            rem_self = W.removal(bH, bind_cut(model, pkH, group_specs(top, omegas, pkH["U1"])))
            rem_sib = W.removal(bS, bind_cut(model, pkS, group_specs(top, omegas, pkS["U1"])))
            drec[f"fra_k{k}_perhead"] = {
                "rem_self": rem_self, "rem_sibling": rem_sib, "n_cells": len(top),
                "selectivity_ratio": rem_self / rem_sib if abs(rem_sib) > 1e-6 else None}
        drec["fra_k10_cells"] = [[list(cands_d[ii][0]), list(cands_d[ii][1])]
                                 for hk in bank for ii in by_head[hk][:10]]
        cmd = bank_metrics(MsL, bank)
        cmd["edge_cosine"] = bank_edge_cosine(MsL, bank)
        drec["drift"] = cmd
        sel[direction] = drec
        print(f"[T2-select/{mkey}/{direction}] k10 self={drec['fra_k10_perhead']['rem_self']:.3f} "
              f"sib={drec['fra_k10_perhead']['rem_sibling']:.3f}", flush=True)
        ckpt(out)
    mrec["selectivity"] = sel


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
        if acc is not None and acc >= GATE:
            passing = name
            break
    if passing is None and not SMOKE:
        for name in SWEEP_AF05:
            acc = gate_model(name)
            if acc is not None and acc >= GATE:
                passing = name
                break
    force = os.environ.get("T2_FORCE_PASS")
    if force and passing is None:
        gate_model(force)
        passing = force
        res["forced_pass"] = force
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

    width = EF_WIDTH[ef_of(passing)]
    n_fra = (40 if SMOKE else 160) if width <= 512 else 100
    n_pairs = (24 if SMOKE else 120) if width <= 512 else 80
    bind_all = W.gen_bind_prompts(n_fra, random.Random(SEED + 4))
    b_loc, b_hold = bind_all[: n_fra // 2], bind_all[n_fra // 2 :]
    pairs = gen_bind_pairs(n_pairs, random.Random(SEED + 5))
    pairs_loc, pairs_hold = pairs[: n_pairs // 2], pairs[n_pairs // 2 :]

    for mkey, name in full_models:
        if mkey in res["models"] and "selectivity" in res["models"][mkey]:
            continue  # resume skip
        if res["sweep"].get(name, {}).get("acc") is None:
            continue
        model, config = W.load_ws_model(name)
        mrec = res["models"].setdefault(mkey, {"name": name,
                                               "gate_acc": res["sweep"][name]["acc"]})
        mrec["gate_ok"] = res["sweep"][name]["acc"] >= GATE
        t2_full_block(model, config, tok, name, mkey, mrec, b_loc, b_hold,
                      pairs_loc, pairs_hold, out, res)
        ckpt(out)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()


# ---------------- main ----------------
def main():
    out = {"meta": {"seed": SEED, "smoke": SMOKE, "stages": STAGES,
                    "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}}
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

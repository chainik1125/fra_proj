"""B1 on GPT-2 (LOCAL, CPU) -- conjunction removal: FRA QK cell vs single-SAE-feature(additive) vs DoM
vs payload-suppress vs OV vs QK+OV hybrid. Runs on a laptop CPU (no NCSA/gemma needed).

Uses the synthetic bigram conjunction confirmed in scripts/61 (GPT-2 does it with rule repetition):
plant, REP times, three token-sharing rules in a random-token sequence --
    A B -> P   (target, remove this)
    D B -> S   (reuse-B: shares B, preserve)
    A C -> Q   (reuse-A: shares A, preserve)
then query the pair. Neither token alone determines the payload (scripts/61: pair 0.67 vs 0.05/0.00).
Plus a payload-elsewhere control: P copied via an UNRELATED single-token trigger Z (Z ... Z -> P), which
payload/OV suppression damages but the FRA cell (located on A B -> P) does not.

Machinery (SAEs, _build_fra_result, patch, paysupp) is scripts/40 (the GPT-2 PoC, reproduced locally:
FRA-QK 0.054 vs trigger 5.65 vs payload 4.12). OV/hybrid mirror scripts/58. Metric: worst-case
collateral KL over {reuseA, reuseB, payload-elsewhere} at matched target-removal, plus REACH.
"""
import os, json
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT = os.environ.get("OUTDIR", "results/b1_gpt2"); os.makedirs(OUT, exist_ok=True)
dev = "cpu"; torch.set_grad_enabled(False)
REP = int(os.environ.get("REP", "8")); NSEED = int(os.environ.get("NSEED", "10")); M_PAIRS = int(os.environ.get("M_PAIRS", "12"))
OV_FIX = float(os.environ.get("OV_FIX", "1.0")); DL = int(os.environ.get("DL", "6"))
model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval(); tok = model.tokenizer
IND = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]; LAYERS = sorted(set(L for L, H in IND)); L0 = min(LAYERS); Llast = model.cfg.n_layers - 1
saes = {L: SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev) for L in LAYERS}
saes = {L: (s[0] if isinstance(s, tuple) else s) for L, s in saes.items()}; W_U = model.W_U
print(f"[model] gpt2 cpu | REP={REP} NSEED={NSEED} DL={DL}", flush=True)

def fra_ph(tt):
    _, c = model.run_with_cache(tt, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H = {}
    for (L, Hh) in IND:
        fe = saes[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh = fe @ saes[L].W_dec.float() + saes[L].b_dec.float()
        r = _build_fra_result(model, L, Hh, fe, saes[L].W_dec.float(), dev, top_k=None, rms_activations=xh, dec_norms=None, chunk_size=16, verbose=False)
        f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
        H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())
    return H
def primer_pairs(HF, qpos, kpos_list, M=12):
    """Union the top-M content feature-pairs over ALL key positions the query attends to (the payload
    tokens appear REP times; the query attends to the later copies, so aggregate over all P positions)."""
    P = {}
    for (L, Hh) in IND:
        d = HF[(L, Hh)]; cells = set()
        for kk in kpos_list:
            loc = np.where((d["qq"] == qpos) & (d["kk"] == kk))[0]; loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
            for o in loc: cells.add((int(d["ii"][o]), int(d["jj"][o])))
        P[(L, Hh)] = cells
    return P
def delta_content(HF, P, seq):
    byL = {}
    for (L, Hh) in IND:
        d = HF[(L, Hh)]; dd = np.zeros((seq, seq)); Ps = P[(L, Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]), int(d["jj"][n])) in Ps: dd[d["qq"][n], d["kk"][n]] += d["vv"][n]
        byL.setdefault(L, {})[Hh] = dd
    return byL
def _fra_hooks(byL, c):
    hooks = []
    for L, hd in byL.items():
        td = {Hh: torch.tensor(dd, device=dev, dtype=torch.float32) * c for Hh, dd in hd.items()}
        def mk(td):
            def hook(s, hook):
                seq = s.shape[-1]
                for Hh, sd in td.items(): s[0, Hh, :seq, :seq] -= sd[:seq, :seq]
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
    return hooks
def _ov_hooks(Pid, s):
    uP = W_U[:, Pid].float(); uP = uP / (uP.norm() + 1e-6); hooks = []
    for L in LAYERS:
        def mk():
            def hook(act, hook): act[0] = act[0] - s * (act[0].float() @ uP).unsqueeze(-1) * uP; return act
            return hook
        hooks.append((f"blocks.{L}.hook_attn_out", mk()))
    return hooks
def patch(tt, byL, c): return model.run_with_hooks(tt, fwd_hooks=_fra_hooks(byL, c))[0]
def ov_run(tt, Pid, s): return model.run_with_hooks(tt, fwd_hooks=_ov_hooks(Pid, s))[0]
def hybrid_run(tt, byL, c, Pid, s): return model.run_with_hooks(tt, fwd_hooks=_fra_hooks(byL, c) + _ov_hooks(Pid, s))[0]
def feat_add(tt, fidx, c):
    Wd = saes[DL].W_dec[fidx].float(); Wd = Wd / (Wd.norm() + 1e-6)
    def hook(act, hook):
        x = act[0].float(); act[0] = x - c * x.norm(dim=-1, keepdim=True) * Wd; return act
    return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]
def dom_run(tt, vD, a):
    def hook(act, hook):
        x = act[0].float(); act[0] = x - a * vD * x.norm(dim=-1, keepdim=True); return act
    return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]
def paysupp(tt, Pid, s):
    uP = W_U[:, Pid].float(); uP = uP / uP.norm()
    def hook(act, hook): act[0] = act[0] - s * (act[0] @ uP).unsqueeze(-1) * uP; return act
    return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{Llast}.hook_resid_post", hook)])[0]
def mask_run(tt, qs, ks, c):
    hooks = []
    for L in LAYERS:
        hs = [H for (LL, H) in IND if LL == L]
        def mk(hs):
            def hook(s, hook):
                for H in hs:
                    for qq in qs:
                        for kk in ks:
                            if qq < s.shape[-2] and kk < s.shape[-1]: s[0, H, qq, kk] -= c
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(hs)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
def lastKL(p, q, pos):
    lp = torch.log_softmax(p[pos].float(), -1); lq = torch.log_softmax(q[pos].float(), -1)
    return (lp.exp() * (lp - lq)).sum().item()
def pat(lg, q, pid): return torch.softmax(lg[q].float(), -1)[pid].item()

def make(seed):
    g = torch.Generator().manual_seed(seed)
    A, B, C, D, P, S, Q, Z = (torch.randperm(30000, generator=g)[:8] + 1500).tolist()
    fill = (torch.randperm(20000, generator=g)[:80] + 22000).tolist()
    seq = [tok.bos_token_id]; ppos = []
    for rep in range(REP):
        seq += [fill[(rep * 3) % len(fill)], A, B, P]; ppos.append(len(seq) - 1)
        seq += [fill[(rep * 3 + 1) % len(fill)], D, B, S]
        seq += [fill[(rep * 3 + 2) % len(fill)], A, C, Q]
    pe = [tok.bos_token_id] + fill[:6] + [Z, P] + fill[6:14] + [Z, P] + fill[14:20] + [Z]
    return dict(seq=seq, A=A, B=B, C=C, D=D, P=P, S=S, Q=Q, Z=Z, ppos=ppos, pe=pe)

FC = [1, 2, 4, 8, 16]; AC = [0.25, 0.5, 1, 2, 4, 8]; DC = [0.25, 0.5, 1, 2, 4, 8]; PC = [0.5, 1, 2, 4, 8]; OC = [0.5, 1, 2, 4, 8]; MC = [2, 4, 8, 1000]
rows = []
for seed in range(NSEED):
    d = make(seed); A, B, C, D, P, S, Q = d["A"], d["B"], d["C"], d["D"], d["P"], d["S"], d["Q"]
    ids = {"target": d["seq"] + [A, B], "reuseA": d["seq"] + [A, C], "reuseB": d["seq"] + [D, B]}
    tt = {k: torch.tensor(v, device=dev).unsqueeze(0) for k, v in ids.items()}
    payid = {"target": P, "reuseA": Q, "reuseB": S}
    clean = {k: model(tt[k])[0] for k in ids}; qp = {k: len(ids[k]) - 1 for k in ids}
    base = {k: pat(clean[k], qp[k], payid[k]) for k in ids}
    pe_tt = torch.tensor(d["pe"], device=dev).unsqueeze(0); pe_q = len(d["pe"]) - 1; pe_clean = model(pe_tt)[0]; pe_base = pat(pe_clean, pe_q, P)
    if base["target"] < 0.2:
        print(f"seed {seed}: target base {base['target']:.3f} <0.2 skip", flush=True); continue
    # LOCATE FRA on target: query=last pos, key=first P position
    HFt = fra_ph(tt["target"]); Pp = primer_pairs(HFt, qp["target"], d["ppos"], M=M_PAIRS)
    byL = {k: delta_content(fra_ph(tt[k]), Pp, tt[k].shape[1]) for k in ids}
    pe_byL = delta_content(fra_ph(pe_tt), Pp, pe_tt.shape[1])
    # attribution for single-feature + DoM: target (A B, P present) vs reuseA (A C, P absent) at DL
    def resid(t): return model.run_with_cache(t, names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
    rt = resid(tt["target"])[qp["target"]]; ra = resid(tt["reuseA"])[qp["reuseA"]]
    vD = (rt - ra).float(); vD = vD / (vD.norm() + 1e-6)
    ft = saes[DL].encode(rt.unsqueeze(0)).float()[0]; fa = saes[DL].encode(ra.unsqueeze(0)).float()[0]
    top_feat = int(torch.topk(ft - fa, 1).indices[0])
    def collat(run_p, run_pe):
        cK = {};
        for k in ("reuseA", "reuseB"):
            lk = run_p(k); cK[k] = lastKL(clean[k], lk, qp[k])
        cK["payelse"] = lastKL(pe_clean, run_pe(), pe_q)
        return cK
    def sweep(run_t, run_p, run_pe, grid):
        out = []
        for c in grid:
            rem = 1 - pat(run_t(c), qp["target"], P) / base["target"]
            out.append((rem, collat(lambda k: run_p(k, c), lambda: run_pe(c))))
        return out
    M = {}
    M["fra"] = sweep(lambda c: patch(tt["target"], byL["target"], c), lambda k, c: patch(tt[k], byL[k], c), lambda c: patch(pe_tt, pe_byL, c), FC)
    M["feat1"] = sweep(lambda c: feat_add(tt["target"], top_feat, c), lambda k, c: feat_add(tt[k], top_feat, c), lambda c: feat_add(pe_tt, top_feat, c), AC)
    M["dom"] = sweep(lambda c: dom_run(tt["target"], vD, c), lambda k, c: dom_run(tt[k], vD, c), lambda c: dom_run(pe_tt, vD, c), DC)
    M["pay"] = sweep(lambda c: paysupp(tt["target"], P, c), lambda k, c: paysupp(tt[k], P, c), lambda c: paysupp(pe_tt, P, c), PC)
    M["ov"] = sweep(lambda c: ov_run(tt["target"], P, c), lambda k, c: ov_run(tt[k], P, c), lambda c: ov_run(pe_tt, P, c), OC)
    M["hybrid"] = sweep(lambda c: hybrid_run(tt["target"], byL["target"], c, P, OV_FIX), lambda k, c: hybrid_run(tt[k], byL[k], c, P, OV_FIX), lambda c: hybrid_run(pe_tt, pe_byL, c, P, OV_FIX), FC)
    # oracle: mask the query's attention to the payload (P) positions -- the attention-routed ceiling
    M["oracle"] = sweep(lambda c: mask_run(tt["target"], [qp["target"]], d["ppos"], c), lambda k, c: mask_run(tt[k], [qp[k]], [0], c), lambda c: pe_clean, MC)
    for mname, sw in M.items():
        for (rem, cK) in sw:
            rows.append(dict(seed=seed, method=mname, removal=rem, colKL_reuseA=cK["reuseA"], colKL_reuseB=cK["reuseB"], colKL_payelse=cK["payelse"]))
    print(f"seed {seed}: base tgt {base['target']:.2f} rA {base['reuseA']:.2f} rB {base['reuseB']:.2f} pe {pe_base:.2f} feat {top_feat} | " + " ".join(f"{m}=rem{max(r for r,_ in M[m]):.2f}" for m in M), flush=True)

def interp_at(sw, thr):
    xs = [r["removal"] for r in sw]
    if not xs or max(xs) < thr: return None
    o = np.argsort(xs)
    def I(key): return float(np.interp(thr, np.array(xs)[o], np.array([r[key] for r in sw])[o]))
    return max(I("colKL_reuseA"), I("colKL_reuseB"), I("colKL_payelse"))
print("\n\n######## GPT-2 B1: WORST-case collateral KL over {reuseA,reuseB,payelse} at matched removal ########", flush=True)
for thr in (0.3, 0.4, 0.5, 0.7, 0.9):
    print(f"\n=== at {int(thr*100)}% removal ===", flush=True)
    for m in ("fra", "hybrid", "feat1", "dom", "pay", "ov", "oracle"):
        vals = []; reached = 0; total = 0
        for seed in set(r["seed"] for r in rows):
            sw = [r for r in rows if r["method"] == m and r["seed"] == seed]
            if not sw: continue
            total += 1; v = interp_at(sw, thr)
            if v is not None: reached += 1; vals.append(v)
        if not vals: print(f"  {m:8}: reached 0/{total}", flush=True); continue
        print(f"  {m:8}: worstKL mean {np.mean(vals):.4f} / median {np.median(vals):.4f} / max {np.max(vals):.4f} | reached {reached}/{total}", flush=True)
json.dump({"rows": rows}, open(os.path.join(OUT, "b1_gpt2.json"), "w"), indent=2, default=float)
print("\nDONE gpt2_conjunction_removal", flush=True)

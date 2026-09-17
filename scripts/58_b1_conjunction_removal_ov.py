"""B1 v2 -- conjunction removal with OV-path steering + QK+OV HYBRID (Dmitry's Sep 17 feedback).

RESEARCH CONTEXT. Defensive interpretability for the FRA paper with Dmitry Manning-Coe. Benign planted
associations only. See docs/insen/research_context.md and proposals/experiments_to_run/B1_controlled_conjunction.md.

Extends scripts/57. Dmitry's run of 57 showed FRA has LOWER collateral than single-SAE-feature but is
NOT Pareto (it reached 50%/70% removal on only 5/12, 2/12 cases -- limited reach). His feedback:
 (1) "editing features in the OV path is more effective -- try a hybrid that also steers OV";
 (2) the SAE hookpoint should be the pre-attention residual (hook_resid_pre here; DL configurable).
This script adds:
 - ov  : OV-path suppression of the TARGET payload direction at the induction layers' hook_attn_out
         (removes the payload that ATTENTION copied, not a global output edit).
 - hybrid: FRA QK cell-cut + a modest OV nudge in ONE forward -> should recover reach while staying
           targeted.
 - a PAYLOAD-ELSEWHERE collateral set (legit uses of the payload word, e.g. counting "...seven, eight,"
   -> nine). This is what honestly penalises payload/OV suppression: reuse probes use a DIFFERENT
   payload, so they do NOT penalise suppressing the target payload -- only payload-elsewhere does.
Primary number: WORST-case collateral over {reuseA, reuseB, payload-elsewhere} at matched removal, plus
REACH (how many group-seeds hit each removal level). FRA's claim = lowest worst-case where it reaches;
hybrid's claim = FRA-level targeting with single-feature-level reach.

Machinery spliced from scripts/57 / g4_65k.py.
"""

import os, sys, json
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT = os.environ.get("OUTDIR", "."); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float16); model.eval(); tok = model.tokenizer
print("[model] gemma-2-2b base", flush=True)
Llast = model.cfg.n_layers - 1; W_U = model.W_U
torch.manual_seed(0); N = 24
R = (torch.randperm(40000)[:N] + 1000).tolist(); tt0 = torch.tensor([tok.bos_token_id] + R + R, device=dev).unsqueeze(0)
edges0 = [(1 + N + t, t + 2) for t in range(N - 1)]
_, c0 = model.run_with_cache(tt0, names_filter=lambda n: n.endswith("hook_pattern"))
strength = {}
for L in range(model.cfg.n_layers):
    p = c0[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(p.shape[0]): strength[(L, H)] = float(np.mean([p[H, q, k].item() for q, k in edges0]))
N_HEADS = int(os.environ.get("N_HEADS", "25")); HEAD_THR = float(os.environ.get("HEAD_THR", "0.4"))
IND = [lh for lh, s in sorted(strength.items(), key=lambda x: -x[1]) if s > HEAD_THR][:N_HEADS]
LAYERS = sorted(set(L for L, H in IND)); DL = int(os.environ.get("DL", str(min(6, Llast))))
print("induction heads:", [f"L{L}H{H}" for L, H in IND], "| DL(pre-attn hookpoint)=", DL, flush=True)
SAE = {}
for L in sorted(set(LAYERS) | {DL}):
    sl = L - 1
    try: SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical", f"layer_{sl}/width_65k/canonical", device=dev, normalize_activations=True)
    except Exception: SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res", f"layer_{sl}/width_65k/average_l0_72", device=dev, normalize_activations=True)
print("SAEs loaded", flush=True); dsae = SAE[DL]

def encode(L, x):
    f = SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f = f / SAE[L]._norm_coeff
    return f
def fra_ph(tt):
    _, c = model.run_with_cache(tt, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H = {}
    for (L, Hh) in IND:
        fe = encode(L, c[f"blocks.{L}.hook_resid_pre"][0]); xh = fe @ SAE[L].W_dec.float() + SAE[L].b_dec.float()
        r = _build_fra_result(model, L, Hh, fe, SAE[L].W_dec.float(), dev, top_k=None, rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
        f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
        H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())
    return H
def primer_pairs(HF, edge, M=48):
    P = {}
    for (L, Hh) in IND:
        d = HF[(L, Hh)]; loc = np.where((d["qq"] == edge[0]) & (d["kk"] == edge[1]))[0]
        loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L, Hh)] = set((int(d["ii"][o]), int(d["jj"][o])) for o in loc)
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
                for Hh, sd in td.items(): s[0, Hh, :seq, :seq] = s[0, Hh, :seq, :seq] - sd[:seq, :seq].to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
    return hooks
def _ov_hooks(Pid, s):
    uP = W_U[:, Pid].float(); uP = uP / (uP.norm() + 1e-6)
    hooks = []
    for L in LAYERS:
        def mk():
            def hook(act, hook):
                act[0] = act[0] - (s * (act[0].float() @ uP).unsqueeze(-1) * uP).to(act.dtype); return act
            return hook
        hooks.append((f"blocks.{L}.hook_attn_out", mk()))
    return hooks
def patch_fra(tt, byL, c): return model.run_with_hooks(tt, fwd_hooks=_fra_hooks(byL, c))[0]
def ov_run(tt, Pid, s): return model.run_with_hooks(tt, fwd_hooks=_ov_hooks(Pid, s))[0]
def hybrid_run(tt, byL, c, Pid, s): return model.run_with_hooks(tt, fwd_hooks=_fra_hooks(byL, c) + _ov_hooks(Pid, s))[0]
def dom_run(tt, positions, vD, a):
    def hook(act, hook):
        for p in positions:
            if p < tt.shape[1]: act[0, p, :] = act[0, p, :] - (a * vD * act[0, p, :].float().norm()).to(act.dtype)
        return act
    return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]
def feat_add_run(tt, fidx, c):
    Wd = dsae.W_dec[fidx].float(); Wd = Wd / (Wd.norm() + 1e-6)
    def hook(act, hook):
        x = act[0].float(); act[0] = (x - c * x.norm(dim=-1, keepdim=True) * Wd).to(act.dtype); return act
    return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]
def paysupp(tt, Pid, s):
    uP = W_U[:, Pid].float(); uP = uP / uP.norm()
    def hook(act, hook): act[0] = act[0] - (s * (act[0].float() @ uP).unsqueeze(-1) * uP).to(act.dtype); return act
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
                            if qq < s.shape[-2] and kk < s.shape[-1]: s[0, H, qq, kk] = s[0, H, qq, kk] - c
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(hs)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
def lastKL(p, q, pos):
    lp = torch.log_softmax(p[pos].float(), -1); lq = torch.log_softmax(q[pos].float(), -1)
    return (lp.exp() * (lp - lq)).sum().item()
def pat(lg, q, pid): return torch.softmax(lg[q].float(), -1)[pid].item()

FILL = ["The weather was mild.", "People arrived late.", "The library opened early.", "A dog waited by the door.",
 "The report took long.", "The train was quiet.", "She left a note on the desk.", "The garden needed water.",
 "Two students argued.", "The coffee had gone cold.", "Chairs were stacked by the wall.", "He forgot his umbrella."]
GROUPS = [
 dict(name="redfox", demos=[("red", "fox", "nine"), ("blue", "fox", "three"), ("red", "owl", "seven")],
      target=("red", "fox", "nine"), reuseA=("red", "owl", "seven"), reuseB=("blue", "fox", "three"),
      novelB="cat", novelA="gray", payelse="Count along: one, two, three, four, five, six, seven, eight,"),
 dict(name="irongate", demos=[("iron", "gate", "four"), ("iron", "lake", "six"), ("steel", "gate", "two")],
      target=("iron", "gate", "four"), reuseA=("iron", "lake", "six"), reuseB=("steel", "gate", "two"),
      novelB="pond", novelA="brass", payelse="Count along: one, two, three,"),
 dict(name="bluemoon", demos=[("blue", "moon", "eight"), ("blue", "sun", "one"), ("pale", "moon", "five")],
      target=("blue", "moon", "eight"), reuseA=("blue", "sun", "one"), reuseB=("pale", "moon", "five"),
      novelB="star", novelA="dark", payelse="Count along: one, two, three, four, five, six, seven,"),
]
def single(w): return len(tok.encode(w, add_special_tokens=False)) == 1
def tid(w): return tok.encode(w, add_special_tokens=False)[0]
def build(seed, demos, qa, qb):
    g = np.random.default_rng(seed); order = list(g.permutation(len(demos))); fs = [FILL[i] for i in g.permutation(len(FILL))]
    body = " ".join(fs[:3]) + "".join(f" The password for {demos[i][0]} {demos[i][1]} is {demos[i][2]}." for i in order) + " " + " ".join(fs[3:6])
    text = body + f" Remember the password for {qa} {qb}:"
    return [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)

FC = [1, 2, 4, 8, 16, 32]; DC = [0.25, 0.5, 1, 2, 4, 8]; AC = [0.25, 0.5, 1, 2, 4, 8]
PC = [0.5, 1, 2, 4, 8]; OC = [0.5, 1, 2, 4, 8]; MC = [2, 4, 8, 16, 1000]
OV_FIX = float(os.environ.get("OV_FIX", "1.0"))   # constant OV nudge inside the hybrid; FRA is swept
NSEED = int(os.environ.get("NSEED", "6")); M_PAIRS = int(os.environ.get("M_PAIRS", "48"))
rows = []
for G in GROUPS:
    demos = G["demos"]; A, B, pay = G["target"]; pid = tid(f" {pay}")
    if not all(single(f" {w}") for w in [A, B, pay, G["reuseA"][2], G["reuseB"][2], G["novelA"], G["novelB"]]):
        print(f"skip {G['name']}: multitoken", flush=True); continue
    ids0 = build(0, demos, A, B); tt = torch.tensor(ids0, device=dev).unsqueeze(0); qpos = len(ids0) - 1
    paypos = [i for i, t in enumerate(ids0) if t == pid]
    if not paypos: print(f"skip {G['name']}: payload not in context", flush=True); continue
    HF = fra_ph(tt); Pp = primer_pairs(HF, (qpos, paypos[0]), M=M_PAIRS)
    print(f"\n=== {G['name']}: located cell qpos={qpos} kpos={paypos[0]} ({sum(len(v) for v in Pp.values())} cells) ===", flush=True)
    onr = []; offr = []; onf = []; offf = []
    for s in range(12):
        r = lambda ids: model.run_with_cache(torch.tensor(ids, device=dev).unsqueeze(0), names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        a = r(build(100 + s, demos, A, B)); onr.append(a[-1]); onf.append(encode(DL, a[-1:])[0])
        b = r(build(100 + s, demos, A, G["novelB"])); offr.append(b[-1]); offf.append(encode(DL, b[-1:])[0])
    def unit(v): v = v.float(); return v / (v.norm() + 1e-6)
    vD = unit(torch.stack(onr).mean(0) - torch.stack(offr).mean(0))
    top_feat = int(torch.topk(torch.stack(onf).mean(0) - torch.stack(offf).mean(0), 1).indices[0])
    print(f"  single-feature (top-1 additive, pre-attn hookpoint L{DL}) = feature {top_feat}", flush=True)
    # payload-elsewhere prompt (legit use of the payload word)
    pe_ids = [tok.bos_token_id] + tok.encode(G["payelse"], add_special_tokens=False); pe_tt = torch.tensor(pe_ids, device=dev).unsqueeze(0)
    pe_q = len(pe_ids) - 1; pe_clean = model(pe_tt)[0]; pe_base = pat(pe_clean, pe_q, pid)
    pe_HF = fra_ph(pe_tt); pe_byL = delta_content(pe_HF, Pp, pe_tt.shape[1])
    print(f"  payload-elsewhere base P({pay})={pe_base:.3f}", flush=True)
    probes = {"target": (A, B, pid), "reuseA": (G["reuseA"][0], G["reuseA"][1], tid(f" {G['reuseA'][2]}")),
              "reuseB": (G["reuseB"][0], G["reuseB"][1], tid(f" {G['reuseB'][2]}"))}
    for seed in range(NSEED):
        pr_ids = {k: build(1000 + seed, demos, qa, qb) for k, (qa, qb, _) in probes.items()}
        pr_tt = {k: torch.tensor(v, device=dev).unsqueeze(0) for k, v in pr_ids.items()}
        clean = {k: model(pr_tt[k])[0] for k in probes}
        q = {k: len(pr_ids[k]) - 1 for k in probes}
        base = {k: pat(clean[k], q[k], probes[k][2]) for k in probes}
        if base["target"] < 0.2:
            print(f"  seed {seed}: target base {base['target']:.3f} <0.2 skip", flush=True); continue
        HFp = {k: fra_ph(pr_tt[k]) for k in probes}
        byLp = {k: delta_content(HFp[k], Pp, pr_tt[k].shape[1]) for k in probes}
        def collat(run_probe, run_pe):
            colKL = {}; colDrop = {}
            for k in ("reuseA", "reuseB"):
                lk = run_probe(k); colKL[k] = lastKL(clean[k], lk, q[k]); colDrop[k] = 1 - pat(lk, q[k], probes[k][2]) / max(base[k], 1e-6)
            lpe = run_pe(); colKL["payelse"] = lastKL(pe_clean, lpe, pe_q)
            colDrop["payelse"] = (1 - pat(lpe, pe_q, pid) / max(pe_base, 1e-6)) if pe_base > 0.05 else float("nan")
            return colKL, colDrop
        def sweep(run_t, run_p, run_pe, grid):
            out = []
            for c in grid:
                rem = 1 - pat(run_t(pr_tt["target"], c), q["target"], pid) / base["target"]
                colKL, colDrop = collat(lambda k: run_p(k, c), lambda: run_pe(c))
                out.append((rem, colKL, colDrop))
            return out
        methods = {}
        methods["fra"] = sweep(lambda tt, c: patch_fra(tt, byLp["target"], c), lambda k, c: patch_fra(pr_tt[k], byLp[k], c), lambda c: patch_fra(pe_tt, pe_byL, c), FC)
        methods["feat1"] = sweep(lambda tt, c: feat_add_run(tt, top_feat, c), lambda k, c: feat_add_run(pr_tt[k], top_feat, c), lambda c: feat_add_run(pe_tt, top_feat, c), AC)
        methods["dom"] = sweep(lambda tt, c: dom_run(tt, list(range(tt.shape[1])), vD, c), lambda k, c: dom_run(pr_tt[k], list(range(pr_tt[k].shape[1])), vD, c), lambda c: dom_run(pe_tt, list(range(pe_tt.shape[1])), vD, c), DC)
        methods["pay"] = sweep(lambda tt, c: paysupp(tt, pid, c), lambda k, c: paysupp(pr_tt[k], pid, c), lambda c: paysupp(pe_tt, pid, c), PC)
        methods["ov"] = sweep(lambda tt, c: ov_run(tt, pid, c), lambda k, c: ov_run(pr_tt[k], pid, c), lambda c: ov_run(pe_tt, pid, c), OC)
        methods["hybrid"] = sweep(lambda tt, c: hybrid_run(tt, byLp["target"], c, pid, OV_FIX), lambda k, c: hybrid_run(pr_tt[k], byLp[k], c, pid, OV_FIX), lambda c: hybrid_run(pe_tt, pe_byL, c, pid, OV_FIX), FC)
        kp = [i for i, t in enumerate(pr_ids["target"]) if t == pid]
        methods["oracle"] = sweep(lambda tt, c: mask_run(tt, [q["target"]], kp[:1] if kp else [0], c), lambda k, c: mask_run(pr_tt[k], [q[k]], [0], c), lambda c: pe_clean, MC)
        for mname, sw in methods.items():
            for (rem, colKL, colDrop) in sw:
                rows.append(dict(group=G["name"], method=mname, seed=seed, removal=rem,
                                 colKL_reuseA=colKL["reuseA"], colKL_reuseB=colKL["reuseB"], colKL_payelse=colKL["payelse"]))
        print(f"  seed {seed}: base {base['target']:.2f} | " + " ".join(f"{m}=rem{max(r for r,_,_ in methods[m]):.2f}" for m in methods), flush=True)

def interp_at(sw, thr):
    xs = [r["removal"] for r in sw]
    if not xs or max(xs) < thr: return None
    o = np.argsort(xs)
    def I(key): return float(np.interp(thr, np.array(xs)[o], np.array([r[key] for r in sw])[o]))
    return max(I("colKL_reuseA"), I("colKL_reuseB"), I("colKL_payelse"))
print("\n\n######## B1+OV: WORST-case collateral KL over {reuseA,reuseB,payload-elsewhere} at matched removal ########", flush=True)
for thr in (0.5, 0.7, 0.9):
    print(f"\n=== at {int(thr*100)}% removal ===", flush=True)
    for m in ("fra", "hybrid", "feat1", "dom", "pay", "ov", "oracle"):
        perg = []; reached = 0; total = 0
        for G in GROUPS:
            for seed in set(r["seed"] for r in rows if r["group"] == G["name"]):
                sw = [r for r in rows if r["group"] == G["name"] and r["method"] == m and r["seed"] == seed]
                if not sw: continue
                total += 1; v = interp_at(sw, thr)
                if v is not None: reached += 1; perg.append(v)
        if not perg: print(f"  {m:8}: reached 0/{total}", flush=True); continue
        print(f"  {m:8}: worstKL mean {np.mean(perg):.4f} / median {np.median(perg):.4f} / max {np.max(perg):.4f} | reached {reached}/{total}", flush=True)
json.dump({"rows": rows}, open(os.path.join(OUT, "b1_removal_ov.json"), "w"), indent=2, default=float)
print("\nDONE b1_conjunction_removal_ov", flush=True)

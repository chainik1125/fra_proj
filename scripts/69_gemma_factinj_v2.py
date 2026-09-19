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
Primary number: worst-case collateral over {reuse_subject, reuse_relation} + general-English KL, at matched removal.
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
def indablate(tt, c):  # ablate induction heads: scale their per-head output (hook_z) by (1-c). Dmitry's retrieval baseline.
    hooks = []
    for L in LAYERS:
        hs = [H for (LL, H) in IND if LL == L]
        def mk(hs):
            def hook(z, hook):
                for H in hs: z[0, :, H, :] = z[0, :, H, :] * (1 - c)
                return z
            return hook
        hooks.append((f"blocks.{L}.attn.hook_z", mk(hs)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

def fullKL(p, q):  # mean per-position KL over a whole passage (broad, non-local damage)
    lp = torch.log_softmax(p.float(), -1); lq = torch.log_softmax(q.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()
def firsttok(v): return tok.encode(v, add_special_tokens=False)[0]  # first token id of a value string
GEN_TXT = ["The committee met on Tuesday to discuss the budget and the schedule for the coming year.",
           "She opened the window and listened to the rain fall softly on the quiet street below.",
           "After a long walk in the park they stopped at a small cafe near the river for lunch."]
GEN = [torch.tensor([tok.bos_token_id] + tok.encode(t, add_special_tokens=False), device=dev).unsqueeze(0) for t in GEN_TXT]
# FICTIONAL entities + UNIQUE values (so the answer is copied from context, not gemma's knowledge, and
# the value is unambiguous for the FRA locate). Conjunction: (subject x attribute); each subject has two
# attributes (subject alone ambiguous) and each attribute spans subjects (attribute alone ambiguous).
REL_TXT = {"sf": "ships from", "rn": "runs on", "hb": "is based in", "ld": "is led by",
           "po": "is powered by", "sh": "is stored in", "op": "operates in", "cr": "was created by"}
FACTSETS = [
 dict(name="drives", subj=["Orion", "Atlas", "Vega", "Nova"], rel=["sf", "rn"],
      vals={("Orion","sf"):" Denver", ("Orion","rn"):" coal", ("Atlas","sf"):" Lisbon", ("Atlas","rn"):" steam",
            ("Vega","sf"):" Osaka", ("Vega","rn"):" solar", ("Nova","sf"):" Cairo", ("Nova","rn"):" diesel"},
      target=("Orion","sf"), reuse_subj=("Orion","rn"), reuse_rel=("Atlas","sf")),
 dict(name="firms", subj=["Zephyr", "Cobalt", "Marlin", "Pardus"], rel=["hb", "ld"],
      vals={("Zephyr","hb"):" Boston", ("Zephyr","ld"):" Alice", ("Cobalt","hb"):" Madrid", ("Cobalt","ld"):" Bruno",
            ("Marlin","hb"):" Nairobi", ("Marlin","ld"):" Chen", ("Pardus","hb"):" Oslo", ("Pardus","ld"):" Diaz"},
      target=("Zephyr","hb"), reuse_subj=("Zephyr","ld"), reuse_rel=("Cobalt","hb")),
 dict(name="gadgets", subj=["Comet", "Falcon", "Lynx", "Wren"], rel=["po", "sh"],
      vals={("Comet","po"):" solar", ("Comet","sh"):" Tokyo", ("Falcon","po"):" wind", ("Falcon","sh"):" Berlin",
            ("Lynx","po"):" hydrogen", ("Lynx","sh"):" Cairo", ("Wren","po"):" nuclear", ("Wren","sh"):" Perth"},
      target=("Comet","po"), reuse_subj=("Comet","sh"), reuse_rel=("Falcon","po")),
 dict(name="agents", subj=["Raven", "Sable", "Onyx", "Flint"], rel=["op", "cr"],
      vals={("Raven","op"):" Vienna", ("Raven","cr"):" Tanaka", ("Sable","op"):" Cairo", ("Sable","cr"):" Ivanov",
            ("Onyx","op"):" Dublin", ("Onyx","cr"):" Mendez", ("Flint","op"):" Lima", ("Flint","cr"):" Novak"},
      target=("Raven","op"), reuse_subj=("Raven","cr"), reuse_rel=("Sable","op")),
]
def build_dir(F, seed):
    g = np.random.default_rng(seed); pairs = [(s, r) for s in F["subj"] for r in F["rel"]]
    order = list(g.permutation(len(pairs)))
    return "Notes. " + " ".join(f"The {pairs[i][0]} {REL_TXT[pairs[i][1]]}{F['vals'][pairs[i]]}." for i in order) + " "
def build_q(F, seed, qpair):
    s, r = qpair
    text = build_dir(F, seed) + f"The {s} {REL_TXT[r]}"   # completion cue -> next token is the value (strong recall)
    return [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)

FC = [0.5, 1, 2, 4, 8, 16, 32]; DC = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 2, 4]; AC = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 2, 4]
PC = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 2]; OC = [0.05, 0.1, 0.2, 0.35, 0.5, 1, 2, 4]; MC = [0.5, 1, 2, 4, 8, 1000]; IC = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
OV_FIX = float(os.environ.get("OV_FIX", "1.0"))   # constant OV nudge inside the hybrid; FRA is swept
NSEED = int(os.environ.get("NSEED", "6")); M_PAIRS = int(os.environ.get("M_PAIRS", "48"))
rows = []
for F in FACTSETS:
    tv = F["vals"][F["target"]]; pid = firsttok(tv)
    ids0 = build_q(F, 0, F["target"]); tt = torch.tensor(ids0, device=dev).unsqueeze(0); qpos = len(ids0) - 1
    paypos = [i for i, t in enumerate(ids0) if t == pid and i < qpos - 4]   # value's first token in the directory
    if not paypos: print(f"skip {F['name']}: value token not in directory", flush=True); continue
    HF = fra_ph(tt); Pp = primer_pairs(HF, (qpos, paypos[0]), M=M_PAIRS)
    print(f"\n=== {F['name']} {F['target']}->{tv.strip()}: qpos={qpos} kpos={paypos[0]} ({sum(len(v) for v in Pp.values())} cells) ===", flush=True)
    # attribution: target-Q (value present) vs reuse_subj-Q (different value) at the pre-attn hookpoint
    onr = []; offr = []; onf = []; offf = []
    for s in range(8):
        r = lambda ids: model.run_with_cache(torch.tensor(ids, device=dev).unsqueeze(0), names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        a = r(build_q(F, 100 + s, F["target"])); onr.append(a[-1]); onf.append(encode(DL, a[-1:])[0])
        b = r(build_q(F, 100 + s, F["reuse_subj"])); offr.append(b[-1]); offf.append(encode(DL, b[-1:])[0])
    def unit(v): v = v.float(); return v / (v.norm() + 1e-6)
    vD = unit(torch.stack(onr).mean(0) - torch.stack(offr).mean(0))
    top_feat = int(torch.topk(torch.stack(onf).mean(0) - torch.stack(offf).mean(0), 1).indices[0])
    print(f"  single-feature (top-1 additive, pre-attn hookpoint L{DL}) = {top_feat}", flush=True)
    gt_clean = [model(g)[0] for g in GEN]
    gt_byL = [delta_content(fra_ph(g), Pp, g.shape[1]) for g in GEN]   # content-addressed cells on general text (~empty)
    probes = {"target": (F["target"], pid), "reuse_subj": (F["reuse_subj"], firsttok(F["vals"][F["reuse_subj"]])),
              "reuse_rel": (F["reuse_rel"], firsttok(F["vals"][F["reuse_rel"]]))}
    for seed in range(NSEED):
        pr_ids = {k: build_q(F, 1000 + seed, qpair) for k, (qpair, _) in probes.items()}
        pr_tt = {k: torch.tensor(v, device=dev).unsqueeze(0) for k, v in pr_ids.items()}
        clean = {k: model(pr_tt[k])[0] for k in probes}
        q = {k: len(pr_ids[k]) - 1 for k in probes}
        base = {k: pat(clean[k], q[k], probes[k][1]) for k in probes}
        if base["target"] < 0.12:
            print(f"  seed {seed}: target base {base['target']:.3f} <0.2 skip", flush=True); continue
        HFp = {k: fra_ph(pr_tt[k]) for k in probes}
        byLp = {k: delta_content(HFp[k], Pp, pr_tt[k].shape[1]) for k in probes}
        def collat(run_p, run_gen):
            cK = {}
            for k in ("reuse_subj", "reuse_rel"): cK[k] = lastKL(clean[k], run_p(k), q[k])
            cK["gen"] = float(np.mean([fullKL(gt_clean[i], run_gen(i)) for i in range(len(GEN))]))
            return cK
        def sweep(run_t, run_p, run_gen, grid):
            out = []
            for c in grid:
                rem = 1 - pat(run_t(c), q["target"], pid) / base["target"]
                out.append((rem, collat(lambda k: run_p(k, c), lambda i: run_gen(i, c))))
            return out
        M = {}
        M["fra"] = sweep(lambda c: patch_fra(pr_tt["target"], byLp["target"], c), lambda k, c: patch_fra(pr_tt[k], byLp[k], c), lambda i, c: patch_fra(GEN[i], gt_byL[i], c), FC)
        M["feat1"] = sweep(lambda c: feat_add_run(pr_tt["target"], top_feat, c), lambda k, c: feat_add_run(pr_tt[k], top_feat, c), lambda i, c: feat_add_run(GEN[i], top_feat, c), AC)
        M["dom"] = sweep(lambda c: dom_run(pr_tt["target"], list(range(pr_tt["target"].shape[1])), vD, c), lambda k, c: dom_run(pr_tt[k], list(range(pr_tt[k].shape[1])), vD, c), lambda i, c: dom_run(GEN[i], list(range(GEN[i].shape[1])), vD, c), DC)
        M["pay"] = sweep(lambda c: paysupp(pr_tt["target"], pid, c), lambda k, c: paysupp(pr_tt[k], pid, c), lambda i, c: paysupp(GEN[i], pid, c), PC)
        M["ov"] = sweep(lambda c: ov_run(pr_tt["target"], pid, c), lambda k, c: ov_run(pr_tt[k], pid, c), lambda i, c: ov_run(GEN[i], pid, c), OC)
        M["hybrid"] = sweep(lambda c: hybrid_run(pr_tt["target"], byLp["target"], c, pid, OV_FIX), lambda k, c: hybrid_run(pr_tt[k], byLp[k], c, pid, OV_FIX), lambda i, c: hybrid_run(GEN[i], gt_byL[i], c, pid, OV_FIX), FC)
        M["indab"] = sweep(lambda c: indablate(pr_tt["target"], c), lambda k, c: indablate(pr_tt[k], c), lambda i, c: indablate(GEN[i], c), IC)
        kp = [i for i, t in enumerate(pr_ids["target"]) if t == pid and i < q["target"] - 4]
        M["oracle"] = sweep(lambda c: mask_run(pr_tt["target"], [q["target"]], kp if kp else [0], c), lambda k, c: mask_run(pr_tt[k], [q[k]], [0], c), lambda i, c: gt_clean[i], MC)
        for mname, sw in M.items():
            for (rem, cK) in sw:
                rows.append(dict(group=F["name"], method=mname, seed=seed, removal=rem,
                                 colKL_rs=cK["reuse_subj"], colKL_rr=cK["reuse_rel"], colKL_gen=cK["gen"]))
        print(f"  seed {seed}: base {base['target']:.2f} rs {base['reuse_subj']:.2f} rr {base['reuse_rel']:.2f} | " + " ".join(f"{m}=rem{max(r for r,_ in M[m]):.2f}" for m in M), flush=True)

def interp_at(sw, thr):
    xs = [r["removal"] for r in sw]
    if not xs or max(xs) < thr: return None
    o = np.argsort(xs)
    def I(key): return float(np.interp(thr, np.array(xs)[o], np.array([r[key] for r in sw])[o]))
    return max(I("colKL_rs"), I("colKL_rr")), I("colKL_gen")   # (worst-case reuse, general-text KL)
groups = sorted(set(r["group"] for r in rows))
print("\n\n######## B1_real (fact-injection): collateral at matched removal ########", flush=True)
print("worstReuse = worst-case KL over {reuse_subject, reuse_relation}; genKL = mean per-pos KL on general English", flush=True)
for thr in (0.3, 0.5, 0.7, 0.9):
    print(f"\n=== at {int(thr*100)}% removal ===", flush=True)
    for m in ("fra", "hybrid", "feat1", "dom", "pay", "ov", "indab", "oracle"):
        wr = []; gk = []; reached = 0; total = 0
        for G in groups:
            for seed in set(r["seed"] for r in rows if r["group"] == G):
                sw = [r for r in rows if r["group"] == G and r["method"] == m and r["seed"] == seed]
                if not sw: continue
                total += 1; v = interp_at(sw, thr)
                if v is not None: reached += 1; wr.append(v[0]); gk.append(v[1])
        if not wr: print(f"  {m:8}: reached 0/{total}", flush=True); continue
        print(f"  {m:8}: worstReuse mean {np.mean(wr):.4f} | genKL mean {np.mean(gk):.4f} / max {np.max(gk):.4f} | reached {reached}/{total}", flush=True)
json.dump({"rows": rows}, open(os.path.join(OUT, "b1_factinj.json"), "w"), indent=2, default=float)
print("\nDONE gemma_factinj_removal", flush=True)

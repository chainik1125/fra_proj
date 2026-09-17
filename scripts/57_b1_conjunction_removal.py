"""B1 PROPER -- conjunction removal: FRA cell vs single-SAE-feature(additive) vs DoM vs payload-suppress.

RESEARCH CONTEXT. Defensive interpretability for the FRA paper with Dmitry Manning-Coe. Benign planted
associations only. See docs/insen/research_context.md and proposals/experiments_to_run/B1_controlled_conjunction.md.

The conjunction (confirmed by scripts/56, ladder_log "B1 v3"): plant, per group, three token-sharing
password pairs in a recall format --
    "... The password for red fox is nine. ... for blue fox is three. ... for red owl is seven. ...
     Remember the password for <qa> <qb>:"
The PAIR fires the payload (~0.65) while either token with any other partner is ~0.04 -> genuine AND.

We REMOVE the target pair's behaviour (red fox -> nine) and measure COLLATERAL on the probes that reuse
one endpoint (red owl -> seven; blue fox -> three) plus a payload-elsewhere control. FRA cuts the
(query x key) cell that carries red-fox->nine; single-SAE-feature (additive, the baseline that tied us
on the simple task) can only remove a whole endpoint feature, so it must damage red-owl or blue-fox.
Primary number: WORST-case collateral over the reuse probes, read at matched payload-removal.

Machinery (head discovery, 65k SAEs, FRA / DoM / conv / payload functions) is HIS code from
experiments/fra_win/jobs/g4_65k.py, spliced as in scripts/47.
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
# ---- induction heads (identical to scripts/47) ----
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
LAYERS = sorted(set(L for L, H in IND)); L0 = min(LAYERS); DL = int(os.environ.get("DL", str(min(6, Llast))))
print("induction heads:", [f"L{L}H{H}" for L, H in IND], flush=True)
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
def patch_fra(tt, byL, c):
    seq = tt.shape[1]; hooks = []
    for L, hd in byL.items():
        td = {Hh: torch.tensor(dd, device=dev, dtype=torch.float32) * c for Hh, dd in hd.items()}
        def mk(td):
            def hook(s, hook):
                for Hh, sd in td.items(): s[0, Hh, :seq, :seq] = s[0, Hh, :seq, :seq] - sd.to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
def dom_run(tt, positions, vD, a):
    def hook(act, hook):
        for p in positions:
            if p < tt.shape[1]: act[0, p, :] = act[0, p, :] - (a * vD * act[0, p, :].float().norm()).to(act.dtype)
        return act
    return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]
def feat_add_run(tt, fidx, c):
    """Single SAE feature, ADDITIVE: subtract c * unit(decoder vector) from resid at ALL positions.
    Dmitry's baseline -- one feature, additive, not projection/directional/multi-feature."""
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
                    for q in qs:
                        for k in ks:
                            if q < s.shape[-2] and k < s.shape[-1]: s[0, H, q, k] = s[0, H, q, k] - c
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(hs)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
def lastKL(p, q, pos):
    lp = torch.log_softmax(p[pos].float(), -1); lq = torch.log_softmax(q[pos].float(), -1)
    return (lp.exp() * (lp - lq)).sum().item()
def pat(lg, q, pid): return torch.softmax(lg[q].float(), -1)[pid].item()

# ---------------- task ----------------
FILL = ["The weather was mild.", "People arrived late.", "The library opened early.", "A dog waited by the door.",
 "The report took long.", "The train was quiet.", "She left a note on the desk.", "The garden needed water.",
 "Two students argued.", "The coffee had gone cold.", "Chairs were stacked by the wall.", "He forgot his umbrella."]
# each group: demos (A,B,payload) sharing tokens; target = demos[0]; probes reuse one endpoint
GROUPS = [
 dict(name="redfox", demos=[("red", "fox", "nine"), ("blue", "fox", "three"), ("red", "owl", "seven")],
      target=("red", "fox", "nine"), reuseA=("red", "owl", "seven"), reuseB=("blue", "fox", "three"),
      novelB="cat", novelA="gray"),
 dict(name="irongate", demos=[("iron", "gate", "four"), ("iron", "lake", "six"), ("steel", "gate", "two")],
      target=("iron", "gate", "four"), reuseA=("iron", "lake", "six"), reuseB=("steel", "gate", "two"),
      novelB="pond", novelA="brass"),
 dict(name="bluemoon", demos=[("blue", "moon", "eight"), ("blue", "sun", "one"), ("pale", "moon", "five")],
      target=("blue", "moon", "eight"), reuseA=("blue", "sun", "one"), reuseB=("pale", "moon", "five"),
      novelB="star", novelA="dark"),
]
def single(w): return len(tok.encode(w, add_special_tokens=False)) == 1
def tid(w): return tok.encode(w, add_special_tokens=False)[0]

def build(seed, demos, qa, qb):
    g = np.random.default_rng(seed); order = list(g.permutation(len(demos))); fs = [FILL[i] for i in g.permutation(len(FILL))]
    body = " ".join(fs[:3]) + "".join(f" The password for {demos[i][0]} {demos[i][1]} is {demos[i][2]}." for i in order) + " " + " ".join(fs[3:6])
    text = body + f" Remember the password for {qa} {qb}:"
    ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
    return ids

FC = [1, 2, 4, 8, 16, 32]; DC = [0.25, 0.5, 1, 2, 4, 8]; AC = [0.25, 0.5, 1, 2, 4, 8]; PC = [0.5, 1, 2, 4, 8]; MC = [2, 4, 8, 16, 1000]
NSEED = int(os.environ.get("NSEED", "6")); M_PAIRS = int(os.environ.get("M_PAIRS", "48"))
rows = []
for G in GROUPS:
    demos = G["demos"]; A, B, pay = G["target"]; pid = tid(f" {pay}")
    if not all(single(f" {w}") for w in [A, B, pay, G["reuseA"][2], G["reuseB"][2], G["novelA"], G["novelB"]]):
        print(f"skip {G['name']}: multitoken", flush=True); continue
    # ---- LOCATE on ONE AB prompt (seed 0): query=last pos, key=target payload position in demo ----
    ids0 = build(0, demos, A, B); tt = torch.tensor(ids0, device=dev).unsqueeze(0); qpos = len(ids0) - 1
    paypos = [i for i, t in enumerate(ids0) if t == pid]
    if not paypos:
        print(f"skip {G['name']}: payload token not found in context", flush=True); continue
    kpos = paypos[0]
    HF = fra_ph(tt); Pp = primer_pairs(HF, (qpos, kpos), M=M_PAIRS)
    ncells = sum(len(v) for v in Pp.values())
    print(f"\n=== {G['name']}: located red-fox->pay cell qpos={qpos} kpos={kpos} ({ncells} cells) ===", flush=True)
    # ---- attribution for DoM + single feature: AB (payload present) vs reuseA (payload absent for THIS payload) ----
    onr = []; offr = []; onf = []; offf = []
    for s in range(12):
        r = lambda ids: model.run_with_cache(torch.tensor(ids, device=dev).unsqueeze(0), names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        a = r(build(100 + s, demos, A, B)); onr.append(a[-1]); onf.append(encode(DL, a[-1:])[0])
        b = r(build(100 + s, demos, A, G["novelB"])); offr.append(b[-1]); offf.append(encode(DL, b[-1:])[0])   # red cat: same A, no target payload
    def unit(v): v = v.float(); return v / (v.norm() + 1e-6)
    vD = unit(torch.stack(onr).mean(0) - torch.stack(offr).mean(0))
    fdiff = torch.stack(onf).mean(0) - torch.stack(offf).mean(0); top_feat = int(torch.topk(fdiff, 1).indices[0])
    print(f"  single-feature (top-1 additive) = feature {top_feat}", flush=True)
    # ---- probes: target (remove) + reuseA/reuseB (preserve) + payload-elsewhere control ----
    PAY_CTRL = "The clock struck nine."  # generic legit use of a payload-like word; rebuilt per group below
    probes = {"target": (A, B, pid, "remove"),
              "reuseA": (G["reuseA"][0], G["reuseA"][1], tid(f" {G['reuseA'][2]}"), "preserve"),
              "reuseB": (G["reuseB"][0], G["reuseB"][1], tid(f" {G['reuseB'][2]}"), "preserve")}
    for seed in range(NSEED):
        # clean + edited for each probe
        pr_ids = {k: build(1000 + seed, demos, qa, qb) for k, (qa, qb, _, _) in probes.items()}
        pr_tt = {k: torch.tensor(v, device=dev).unsqueeze(0) for k, v in pr_ids.items()}
        clean = {k: model(pr_tt[k])[0] for k in probes}
        q = {k: len(pr_ids[k]) - 1 for k in probes}
        base = {k: pat(clean[k], q[k], probes[k][2]) for k in probes}
        if base["target"] < 0.2:
            print(f"  seed {seed}: target base P={base['target']:.3f} <0.2, skip", flush=True); continue
        # FRA content-addressed per probe
        HFp = {k: fra_ph(pr_tt[k]) for k in probes}
        byLp = {k: delta_content(HFp[k], Pp, pr_tt[k].shape[1]) for k in probes}
        # per method: list of (removal_on_target, {collateral KL per preserve probe}, {collateral payload-drop})
        def sweep(run_target, run_probe, grid):
            out = []
            for c in grid:
                lt = run_target(pr_tt["target"], c)
                rem = 1 - pat(lt, q["target"], pid) / base["target"]
                colKL = {}; colDrop = {}
                for k in ("reuseA", "reuseB"):
                    lk = run_probe(k, c)
                    colKL[k] = lastKL(clean[k], lk, q[k])
                    colDrop[k] = 1 - pat(lk, q[k], probes[k][2]) / max(base[k], 1e-6)
                out.append((rem, colKL, colDrop))
            return out
        methods = {}
        methods["fra"] = sweep(lambda tt, c: patch_fra(tt, byLp["target"], c),
                               lambda k, c: patch_fra(pr_tt[k], byLp[k], c), FC)
        methods["feat1"] = sweep(lambda tt, c: feat_add_run(tt, top_feat, c),
                                lambda k, c: feat_add_run(pr_tt[k], top_feat, c), AC)
        methods["dom"] = sweep(lambda tt, c: dom_run(tt, list(range(tt.shape[1])), vD, c),
                              lambda k, c: dom_run(pr_tt[k], list(range(pr_tt[k].shape[1])), vD, c), DC)
        methods["pay"] = sweep(lambda tt, c: paysupp(tt, pid, c),
                              lambda k, c: paysupp(pr_tt[k], pid, c), PC)
        # oracle: mask exact target qpos->kpos (removal only)
        kp = [i for i, t in enumerate(pr_ids["target"]) if t == pid]
        methods["oracle"] = sweep(lambda tt, c: mask_run(tt, [q["target"]], kp[:1] if kp else [0], c),
                                 lambda k, c: mask_run(pr_tt[k], [q[k]], [0], c), MC)
        for mname, sw in methods.items():
            for (rem, colKL, colDrop) in sw:
                rows.append(dict(group=G["name"], method=mname, seed=seed, removal=rem,
                                 colKL_reuseA=colKL["reuseA"], colKL_reuseB=colKL["reuseB"],
                                 colDrop_reuseA=colDrop["reuseA"], colDrop_reuseB=colDrop["reuseB"]))
        print(f"  seed {seed}: base tgt {base['target']:.2f} reuseA {base['reuseA']:.2f} reuseB {base['reuseB']:.2f} | "
              + " ".join(f"{m}=maxrem{max(r for r,_,_ in methods[m]):.2f}" for m in methods), flush=True)

# ---------------- collate: collateral (worst-case over reuseA/reuseB) at matched removal ----------------
def interp_at(sw_rows, target_rem):
    xs = [r["removal"] for r in sw_rows]
    if not xs or max(xs) < target_rem: return None
    o = np.argsort(xs)
    def I(key): return float(np.interp(target_rem, np.array(xs)[o], np.array([r[key] for r in sw_rows])[o]))
    worstKL = max(I("colKL_reuseA"), I("colKL_reuseB"))
    worstDrop = max(I("colDrop_reuseA"), I("colDrop_reuseB"))
    return worstKL, worstDrop
print("\n\n######## B1 conjunction removal: WORST-case collateral (over reuseA,reuseB) at matched removal ########", flush=True)
for thr in (0.5, 0.7):
    print(f"\n=== at {int(thr*100)}% payload-removal on the target pair ===", flush=True)
    for m in ("fra", "feat1", "dom", "pay", "oracle"):
        perg = []
        for G in GROUPS:
            sw = [r for r in rows if r["group"] == G["name"] and r["method"] == m]
            # aggregate across seeds by mean per removal not trivial; use all points pooled
            res = interp_at(sw, thr)
            if res: perg.append(res)
        if not perg:
            print(f"  {m:8}: never reached {int(thr*100)}%", flush=True); continue
        mKL = np.mean([a for a, b in perg]); mDrop = np.mean([b for a, b in perg])
        wKL = np.max([a for a, b in perg]); wDrop = np.max([b for a, b in perg])
        print(f"  {m:8}: worstKL mean {mKL:.4f} / max {wKL:.4f} | worst payload-drop mean {mDrop:.3f} / max {wDrop:.3f}  (n={len(perg)} groups)", flush=True)
json.dump({"rows": rows}, open(os.path.join(OUT, "b1_removal.json"), "w"), indent=2, default=float)
print("\nDONE b1_conjunction_removal", flush=True)

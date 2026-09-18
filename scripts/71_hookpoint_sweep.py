"""Hookpoint sweep (local, GPT-2, CPU) -- the STRONG-result test (Dmitry).

'An OK result is FRA beats a single SAE feature at that specific hookpoint; a STRONG result is FRA does
something a single SAE feature at NO hookpoint can do.' So: sweep the single-SAE-feature baseline across
ALL layer hookpoints, take the BEST (lowest collateral at matched removal), and compare to FRA. If FRA
beats the best-single-feature-over-all-hookpoints, that's the strong result.

Uses the GPT-2 synthetic conjunction (scripts/62). Single feature = additive, attribution top-1 at each
layer's hook_resid_pre. Collateral = worst-case over {reuse_subj-ish, reuse_rel-ish, general} at matched
target-removal (50%).
"""
import os, json
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
dev = "cpu"; torch.set_grad_enabled(False)
REP = int(os.environ.get("REP", "6")); NSEED = int(os.environ.get("NSEED", "6")); M_PAIRS = int(os.environ.get("M_PAIRS", "12"))
model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval(); tok = model.tokenizer
IND = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]; INDL = sorted(set(L for L, H in IND)); Llast = model.cfg.n_layers - 1; W_U = model.W_U
ALL_L = list(range(1, 12))   # sweep single-feature hookpoints over layers 1..11 (resid_pre)
print("loading SAEs for FRA heads + all sweep layers...", flush=True)
saes = {}
for L in sorted(set(INDL) | set(ALL_L)):
    saes[L] = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev)
    saes[L] = saes[L][0] if isinstance(saes[L], tuple) else saes[L]
print(f"[model] gpt2 cpu | sweep layers {ALL_L}", flush=True)

def fra_ph(tt):
    _, c = model.run_with_cache(tt, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in INDL])
    H = {}
    for (L, Hh) in IND:
        fe = saes[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh = fe @ saes[L].W_dec.float() + saes[L].b_dec.float()
        r = _build_fra_result(model, L, Hh, fe, saes[L].W_dec.float(), dev, top_k=None, rms_activations=xh, dec_norms=None, chunk_size=16, verbose=False)
        f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
        H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())
    return H
def primer_pairs(HF, qpos, kl, M=12):
    P = {}
    for (L, Hh) in IND:
        d = HF[(L, Hh)]; cells = set()
        for kk in kl:
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
def patch(tt, byL, c):
    seq = tt.shape[1]; hooks = []
    for L, hd in byL.items():
        td = {Hh: torch.tensor(dd, device=dev, dtype=torch.float32) * c for Hh, dd in hd.items()}
        def mk(td):
            def hook(s, hook):
                for Hh, sd in td.items(): s[0, Hh, :seq, :seq] -= sd[:seq, :seq]
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
    return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
def feat_add(tt, L, fidx, c):
    Wd = saes[L].W_dec[fidx].float(); Wd = Wd / (Wd.norm() + 1e-6)
    def hook(act, hook):
        x = act[0].float(); act[0] = x - c * x.norm(dim=-1, keepdim=True) * Wd; return act
    return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{L}.hook_resid_pre", hook)])[0]
def lastKL(p, q, pos):
    lp = torch.log_softmax(p[pos].float(), -1); lq = torch.log_softmax(q[pos].float(), -1)
    return (lp.exp() * (lp - lq)).sum().item()
def fullKL(p, q):
    lp = torch.log_softmax(p.float(), -1); lq = torch.log_softmax(q.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()
def pat(lg, q, pid): return torch.softmax(lg[q].float(), -1)[pid].item()
GEN = ["The committee met on Tuesday to discuss the budget for the year.",
       "She opened the window and listened to the rain on the street."]
GENtt = [torch.tensor([tok.bos_token_id] + tok.encode(t, add_special_tokens=False)).unsqueeze(0) for t in GEN]
def make(seed):
    g = torch.Generator().manual_seed(seed)
    A, B, C, D, P, S, Q = (torch.randperm(30000, generator=g)[:7] + 1500).tolist()
    fill = (torch.randperm(20000, generator=g)[:80] + 22000).tolist()
    seq = [tok.bos_token_id]; ppos = []
    for rep in range(REP):
        seq += [fill[(rep*3) % 80], A, B, P]; ppos.append(len(seq) - 1)
        seq += [fill[(rep*3+1) % 80], D, B, S]; seq += [fill[(rep*3+2) % 80], A, C, Q]
    return dict(seq=seq, A=A, B=B, C=C, D=D, P=P, ppos=ppos)
AC = [0.25, 0.5, 1, 2, 4, 8]; FC = [1, 2, 4, 8, 16]
def worst_at(curve, thr):
    xs = [r[0] for r in curve]
    if not xs or max(xs) < thr: return None
    o = np.argsort(xs); return float(np.interp(thr, np.array(xs)[o], np.array([r[1] for r in curve])[o]))
rows = []
for seed in range(NSEED):
    d = make(seed); A, B, C, D, P = d["A"], d["B"], d["C"], d["D"], d["P"]
    tgt = torch.tensor(d["seq"] + [A, B]).unsqueeze(0); rA = torch.tensor(d["seq"] + [A, C]).unsqueeze(0); rB = torch.tensor(d["seq"] + [D, B]).unsqueeze(0)
    qp = tgt.shape[1] - 1; base = pat(model(tgt)[0], qp, P)
    if base < 0.2: continue
    cl = {"rA": model(rA)[0], "rB": model(rB)[0], "gen": [model(g)[0] for g in GENtt]}
    qA = rA.shape[1] - 1; qB = rB.shape[1] - 1
    def collat(fn):
        return max(lastKL(cl["rA"], fn(rA), qA), lastKL(cl["rB"], fn(rB), qB),
                   float(np.mean([fullKL(cl["gen"][i], fn(GENtt[i])) for i in range(len(GENtt))])))
    # FRA
    HFt = fra_ph(tgt); Pp = primer_pairs(HFt, qp, d["ppos"], M=M_PAIRS)
    byL = {"tgt": delta_content(HFt, Pp, tgt.shape[1]), "rA": delta_content(fra_ph(rA), Pp, rA.shape[1]),
           "rB": delta_content(fra_ph(rB), Pp, rB.shape[1]), "gen": [delta_content(fra_ph(g), Pp, g.shape[1]) for g in GENtt]}
    fra_curve = []
    for c in FC:
        rem = 1 - pat(patch(tgt, byL["tgt"], c), qp, P) / base
        col = max(lastKL(cl["rA"], patch(rA, byL["rA"], c), qA), lastKL(cl["rB"], patch(rB, byL["rB"], c), qB),
                  float(np.mean([fullKL(cl["gen"][i], patch(GENtt[i], byL["gen"][i], c)) for i in range(len(GENtt))])))
        fra_curve.append((rem, col))
    fra50 = worst_at(fra_curve, 0.5)
    # single feature at EACH layer hookpoint
    best_layer = None; best_col = 1e9; per_layer = {}
    for L in ALL_L:
        rt = model.run_with_cache(tgt, names_filter=[f"blocks.{L}.hook_resid_pre"])[1][f"blocks.{L}.hook_resid_pre"][0][-1]
        ra = model.run_with_cache(rA, names_filter=[f"blocks.{L}.hook_resid_pre"])[1][f"blocks.{L}.hook_resid_pre"][0][-1]
        fidx = int(torch.topk(saes[L].encode(rt.unsqueeze(0)).float()[0] - saes[L].encode(ra.unsqueeze(0)).float()[0], 1).indices[0])
        cur = []
        for c in AC:
            rem = 1 - pat(feat_add(tgt, L, fidx, c), qp, P) / base
            cur.append((rem, collat(lambda t: feat_add(t, L, fidx, c))))
        v = worst_at(cur, 0.5); per_layer[L] = v
        if v is not None and v < best_col: best_col = v; best_layer = L
    rows.append(dict(seed=seed, fra50=fra50, best_feat50=best_col if best_col < 1e9 else None, best_layer=best_layer,
                     per_layer={L: per_layer[L] for L in per_layer}))
    print(f"seed {seed}: FRA@50% col={fra50} | BEST single-feature over layers 1-11 @50% col={best_col:.4f} (layer {best_layer})", flush=True)

fr = [r["fra50"] for r in rows if r["fra50"] is not None]
bf = [r["best_feat50"] for r in rows if r["best_feat50"] is not None]
print("\n######## STRONG-result test: FRA vs BEST single-feature over ALL hookpoints, @50% removal ########", flush=True)
print(f"  FRA worst-case collateral:                       mean {np.mean(fr):.4f}  (n={len(fr)})", flush=True)
print(f"  BEST single-feature (min over layers 1-11):      mean {np.mean(bf):.4f}  (n={len(bf)})", flush=True)
print(f"  => FRA beats the best-hookpoint single feature by {np.mean(bf)/max(np.mean(fr),1e-6):.1f}x" if fr and bf else "  (insufficient reach)", flush=True)
json.dump({"rows": rows}, open("results/b1_gpt2/hookpoint_sweep.json", "w") if os.path.isdir("results/b1_gpt2") else open("hookpoint_sweep.json", "w"), indent=2, default=float)
print("\nDONE hookpoint_sweep", flush=True)

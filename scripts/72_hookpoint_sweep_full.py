"""FULL hookpoint sweep (local, GPT-2, CPU) -- airtight STRONG-result test (Dmitry's validation #1).

Dmitry: 'An OK result is FRA beats a single SAE feature at THAT hookpoint; a STRONG result is FRA does
something a single SAE feature at NO hookpoint can do.' scripts/71 swept only resid_pre. Dmitry then
asked to also cover ln1.hook_normalized (the direct attention input) and resid_post.

There is NO public SAE at ln1.hook_normalized -- but resid_pre IS the attention input (LN1 is a
deterministic map on top of it, so the resid_pre feature basis already covers the pre-attention input).
For the output side there ARE real SAEs trained at three more hookpoints. So instead of one extra
hookpoint we sweep the single-SAE-feature baseline across FOUR real-SAE hookpoints, every layer, take
the best (lowest collateral at matched removal), and compare to FRA:

  resid_pre   gpt2-small-res-jb            (pre-attention input; = resid_post of the previous layer)
  resid_mid   gpt2-small-resid-mid-v5-32k  (post-attention, pre-MLP -- attention's write in resid space)
  resid_post  gpt2-small-resid-post-v5-32k (end of block)
  attn_out    gpt2-small-attn-out-v5-32k   (attention output directly)

Note resid_pre[L] == resid_post[L-1], so resid_pre + resid_post together already double-cover the
residual stream at both block boundaries. SAEs are loaded one at a time and freed (CPU-memory bounded).

Uses the GPT-2 synthetic conjunction (same as scripts/62/71). Single feature = additive, attribution
top-1 by activation-diff at each hookpoint. Collateral = worst-case over {reuseA, reuseB, general} at
matched 50% target-removal.
"""
import os, json, gc
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
dev = "cpu"; torch.set_grad_enabled(False)
REP = int(os.environ.get("REP", "6")); NSEED = int(os.environ.get("NSEED", "4")); M_PAIRS = int(os.environ.get("M_PAIRS", "12"))
model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval(); tok = model.tokenizer
IND = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]; INDL = sorted(set(L for L, H in IND))

# sweep grid: (release, hook-template, layers)
SWEEP = [
    ("gpt2-small-res-jb",           "blocks.{L}.hook_resid_pre",  list(range(1, 12))),
    ("gpt2-small-resid-mid-v5-32k", "blocks.{L}.hook_resid_mid",  list(range(0, 12))),
    ("gpt2-small-resid-post-v5-32k","blocks.{L}.hook_resid_post", list(range(0, 12))),
    ("gpt2-small-attn-out-v5-32k",  "blocks.{L}.hook_attn_out",   list(range(0, 12))),
]
print("loading FRA SAEs (resid_pre at induction layers)...", flush=True)
fsae = {}
for L in INDL:
    s = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev)
    fsae[L] = s[0] if isinstance(s, tuple) else s
print(f"[model] gpt2 cpu | sweeping {sum(len(x[2]) for x in SWEEP)} hookpoint-SAEs across {len(SWEEP)} hookpoints", flush=True)

# ---- FRA machinery (identical to scripts/71) ----
def fra_ph(tt):
    _, c = model.run_with_cache(tt, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in INDL])
    H = {}
    for (L, Hh) in IND:
        fe = fsae[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh = fe @ fsae[L].W_dec.float() + fsae[L].b_dec.float()
        r = _build_fra_result(model, L, Hh, fe, fsae[L].W_dec.float(), dev, top_k=None, rms_activations=xh, dec_norms=None, chunk_size=16, verbose=False)
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

# ---- single-feature at an arbitrary hookpoint ----
def feat_add(tt, hookname, Wd, c):
    def hook(act, hook):
        x = act[0].float(); act[0] = x - c * x.norm(dim=-1, keepdim=True) * Wd; return act
    return model.run_with_hooks(tt, fwd_hooks=[(hookname, hook)])[0]

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
AC = [0.25, 0.5, 1, 2, 4, 8, 16]; FC = [1, 2, 4, 8, 16, 32, 64, 128]
def worst_at(curve, thr):
    xs = [r[0] for r in curve]
    if not xs or max(xs) < thr: return None
    o = np.argsort(xs); return float(np.interp(thr, np.array(xs)[o], np.array([r[1] for r in curve])[o]))

# precompute per-seed prompts + clean baselines once
SEEDS = []
for seed in range(NSEED):
    d = make(seed); A, B, C, D, P = d["A"], d["B"], d["C"], d["D"], d["P"]
    tgt = torch.tensor(d["seq"] + [A, B]).unsqueeze(0); rA = torch.tensor(d["seq"] + [A, C]).unsqueeze(0); rB = torch.tensor(d["seq"] + [D, B]).unsqueeze(0)
    qp = tgt.shape[1] - 1; base = pat(model(tgt)[0], qp, P)
    if base < 0.2: continue
    cl = {"rA": model(rA)[0], "rB": model(rB)[0], "gen": [model(g)[0] for g in GENtt]}
    qA = rA.shape[1] - 1; qB = rB.shape[1] - 1
    SEEDS.append(dict(seed=seed, d=d, tgt=tgt, rA=rA, rB=rB, qp=qp, base=base, cl=cl, qA=qA, qB=qB))
print(f"[seeds] {len(SEEDS)} usable of {NSEED}", flush=True)

# ---- FRA @50% per seed ----
for S in SEEDS:
    tgt, rA, rB, qp, base, cl, qA, qB, d, P = S["tgt"], S["rA"], S["rB"], S["qp"], S["base"], S["cl"], S["qA"], S["qB"], S["d"], None
    HFt = fra_ph(tgt); Pp = primer_pairs(HFt, qp, d["ppos"], M=M_PAIRS)
    byL = {"tgt": delta_content(HFt, Pp, tgt.shape[1]), "rA": delta_content(fra_ph(rA), Pp, rA.shape[1]),
           "rB": delta_content(fra_ph(rB), Pp, rB.shape[1]), "gen": [delta_content(fra_ph(g), Pp, g.shape[1]) for g in GENtt]}
    curve = []
    for c in FC:
        rem = 1 - pat(patch(tgt, byL["tgt"], c), qp, S["d"]["P"]) / base
        col = max(lastKL(cl["rA"], patch(rA, byL["rA"], c), qA), lastKL(cl["rB"], patch(rB, byL["rB"], c), qB),
                  float(np.mean([fullKL(cl["gen"][i], patch(GENtt[i], byL["gen"][i], c)) for i in range(len(GENtt))])))
        curve.append((rem, col))
    S["fra50"] = worst_at(curve, 0.5)
    print(f"seed {S['seed']}: FRA@50% col={S['fra50']}", flush=True)

# ---- sweep: one SAE at a time, all hookpoints/layers ----
for S in SEEDS:
    S["per"] = {}
for release, tmpl, layers in SWEEP:
    for L in layers:
        hookname = tmpl.format(L=L)
        try:
            s = SAE.from_pretrained(release, hookname, device=dev); s = s[0] if isinstance(s, tuple) else s
        except Exception as e:
            print(f"  skip {release} {hookname}: {e}", flush=True); continue
        Wdec = s.W_dec.float()
        for S in SEEDS:
            tgt, rA, base, qp, cl, qA, qB = S["tgt"], S["rA"], S["base"], S["qp"], S["cl"], S["qA"], S["qB"]
            rt = model.run_with_cache(tgt, names_filter=[hookname])[1][hookname][0][-1]
            ra = model.run_with_cache(rA, names_filter=[hookname])[1][hookname][0][-1]
            fidx = int(torch.topk(s.encode(rt.unsqueeze(0)).float()[0] - s.encode(ra.unsqueeze(0)).float()[0], 1).indices[0])
            Wd = Wdec[fidx]; Wd = Wd / (Wd.norm() + 1e-6)
            cur = []
            for c in AC:
                rem = 1 - pat(feat_add(tgt, hookname, Wd, c), qp, S["d"]["P"]) / base
                col = max(lastKL(cl["rA"], feat_add(S["rA"], hookname, Wd, c), qA),
                          lastKL(cl["rB"], feat_add(S["rB"], hookname, Wd, c), qB),
                          float(np.mean([fullKL(cl["gen"][i], feat_add(GENtt[i], hookname, Wd, c)) for i in range(len(GENtt))])))
                cur.append((rem, col))
            S["per"][hookname] = worst_at(cur, 0.5)
        vals = [S["per"][hookname] for S in SEEDS if S["per"].get(hookname) is not None]
        print(f"  {hookname:32s} mean@50% col = {np.mean(vals):.4f} (n={len(vals)})" if vals else f"  {hookname:32s} no reach", flush=True)
        del s, Wdec; gc.collect()

# ---- collate: best single feature over ALL hookpoints, per seed ----
rows = []
for S in SEEDS:
    vals = {h: v for h, v in S["per"].items() if v is not None}
    best_h = min(vals, key=vals.get) if vals else None
    rows.append(dict(seed=S["seed"], fra50=S["fra50"], best_feat50=(vals[best_h] if best_h else None), best_hook=best_h, per=S["per"]))
fr = [r["fra50"] for r in rows if r["fra50"] is not None]
bf = [r["best_feat50"] for r in rows if r["best_feat50"] is not None]
print("\n######## STRONG-result test: FRA vs BEST single-feature over 4 hookpoints x all layers, @50% removal ########", flush=True)
print(f"  FRA worst-case collateral:                              mean {np.mean(fr):.4f}  (n={len(fr)})", flush=True)
print(f"  BEST single-feature (min over ALL hookpoints+layers):   mean {np.mean(bf):.4f}  (n={len(bf)})", flush=True)
if fr and bf: print(f"  => FRA beats the best-hookpoint single feature by {np.mean(bf)/max(np.mean(fr),1e-6):.1f}x", flush=True)
# which hookpoints ever won
from collections import Counter
wins = Counter(r["best_hook"] for r in rows if r["best_hook"])
print("  best-hookpoint per seed:", dict(wins), flush=True)
outp = "results/b1_gpt2/hookpoint_sweep_full.json" if os.path.isdir("results/b1_gpt2") else "hookpoint_sweep_full.json"
json.dump({"rows": rows, "sweep": [(r, t, l) for r, t, l in SWEEP]}, open(outp, "w"), indent=2, default=float)
print(f"\nDONE hookpoint_sweep_full -> {outp}", flush=True)

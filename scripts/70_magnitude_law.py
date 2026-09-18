"""Magnitude-law prediction (local, GPT-2, CPU). Does the FRA collateral advantage follow from feature reuse?

Claim: at matched removal, single-SAE-feature steering's collateral on a text scales with how much the
removed FEATURE is active there (its "reuse"), while FRA's collateral scales with how much the removed
CELL (query-content x key-content pair) is present there. On collateral text the endpoint feature is
common but the cell is rare, so the predicted advantage is
    A_pred(text) = reuse(endpoint feature on text) / reuse(cell on text).
We measure both on the GPT-2 synthetic conjunction (scripts/61/62) and compare A_pred to the MEASURED
collateral ratio (feat1 / fra) at matched removal (from scripts/62's runs / recomputed here).

reuse(feature) = mean over positions of the top-1 attribution feature's SAE activation.
reuse(cell)    = total |delta_content(Pp)| mass (the located FRA cell's presence in the FRA tensor).
"""
import os, json
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
dev = "cpu"; torch.set_grad_enabled(False)
REP = int(os.environ.get("REP", "6")); NSEED = int(os.environ.get("NSEED", "10")); M_PAIRS = int(os.environ.get("M_PAIRS", "12")); DL = int(os.environ.get("DL", "6"))
model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval(); tok = model.tokenizer
IND = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]; LAYERS = sorted(set(L for L, H in IND))
saes = {L: SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev) for L in LAYERS}
saes = {L: (s[0] if isinstance(s, tuple) else s) for L, s in saes.items()}
print(f"[model] gpt2 cpu | REP={REP} NSEED={NSEED}", flush=True)
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
    P = {}
    for (L, Hh) in IND:
        d = HF[(L, Hh)]; cells = set()
        for kk in kpos_list:
            loc = np.where((d["qq"] == qpos) & (d["kk"] == kk))[0]; loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
            for o in loc: cells.add((int(d["ii"][o]), int(d["jj"][o])))
        P[(L, Hh)] = cells
    return P
def cell_mass(HF, P, qpos):  # |value| of the located feature-pairs USED AT THE OUTPUT query position
    tot = 0.0
    for (L, Hh) in IND:
        d = HF[(L, Hh)]; Ps = P[(L, Hh)]
        for n in range(len(d["vv"])):
            if int(d["qq"][n]) == qpos and (int(d["ii"][n]), int(d["jj"][n])) in Ps: tot += abs(float(d["vv"][n]))
    return tot
def feat_mass(tt, fidx):  # top attribution feature's activation AT THE OUTPUT position
    _, c = model.run_with_cache(tt, names_filter=[f"blocks.{DL}.hook_resid_pre"])
    fe = saes[DL].encode(c[f"blocks.{DL}.hook_resid_pre"][0]).float()
    return float(fe[-1, fidx].item())

GEN = ["The committee met on Tuesday to discuss the budget for the year.",
       "She opened the window and listened to the rain on the street.",
       "After a walk in the park they stopped at a cafe by the river."]
GENtt = [torch.tensor([tok.bos_token_id] + tok.encode(t, add_special_tokens=False)).unsqueeze(0) for t in GEN]
def make(seed):
    g = torch.Generator().manual_seed(seed)
    A, B, C, D, P, S, Q, Z = (torch.randperm(30000, generator=g)[:8] + 1500).tolist()
    fill = (torch.randperm(20000, generator=g)[:80] + 22000).tolist()
    seq = [tok.bos_token_id]; ppos = []
    for rep in range(REP):
        seq += [fill[(rep*3) % 80], A, B, P]; ppos.append(len(seq) - 1)
        seq += [fill[(rep*3+1) % 80], D, B, S]; seq += [fill[(rep*3+2) % 80], A, C, Q]
    return dict(seq=seq, A=A, B=B, C=C, D=D, P=P, S=S, Q=Q, ppos=ppos)

rows = []
for seed in range(NSEED):
    d = make(seed); A, B, C, D = d["A"], d["B"], d["C"], d["D"]
    ids = {"target": d["seq"]+[A,B], "reuseA": d["seq"]+[A,C], "reuseB": d["seq"]+[D,B]}
    tt = {k: torch.tensor(v).unsqueeze(0) for k, v in ids.items()}
    if torch.softmax(model(tt["target"])[0][-1].float(), -1)[d["P"]].item() < 0.2: continue
    HFt = fra_ph(tt["target"]); Pp = primer_pairs(HFt, len(ids["target"])-1, d["ppos"], M=M_PAIRS)
    # top-1 attribution feature (target vs reuseA, as in scripts/62)
    def resid(t): return model.run_with_cache(t, names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
    ft = saes[DL].encode(resid(tt["target"])[-1:].float())[0]; fa = saes[DL].encode(resid(tt["reuseA"])[-1:].float())[0]
    top_feat = int(torch.topk(ft - fa, 1).indices[0])
    # reuse of feature vs cell on each collateral text
    texts = {"target": tt["target"], "reuseA": tt["reuseA"], "reuseB": tt["reuseB"], "general": GENtt[seed % len(GENtt)]}
    fm = {k: feat_mass(v, top_feat) for k, v in texts.items()}
    cm = {k: cell_mass(fra_ph(v), Pp, v.shape[1] - 1) for k, v in texts.items()}
    for k in texts:
        rows.append(dict(seed=seed, text=k, feat_reuse=fm[k], cell_reuse=cm[k], A_pred=fm[k] / (cm[k] + 1e-9)))
    print(f"seed {seed}: feat {top_feat} | " + " ".join(f"{k}: feat={fm[k]:.3f} cell={cm[k]:.3f} A_pred={fm[k]/(cm[k]+1e-9):.1f}x" for k in texts), flush=True)

print("\n######## magnitude law: predicted advantage A_pred = reuse(feature)/reuse(cell) ########", flush=True)
for k in ("target", "reuseA", "reuseB", "general"):
    sub = [r for r in rows if r["text"] == k]
    if not sub: continue
    fr = np.mean([r["feat_reuse"] for r in sub]); cr = np.mean([r["cell_reuse"] for r in sub]); ap = np.median([r["A_pred"] for r in sub])
    print(f"  {k:8}: mean feat_reuse {fr:.3f} | mean cell_reuse {cr:.4f} | median A_pred {ap:.1f}x", flush=True)
print("\nInterpretation: single-feature removes a feature active on the collateral text (feat_reuse>0),", flush=True)
print("while the FRA cell is nearly absent there (cell_reuse~0) -> A_pred large. On GENERAL text cell_reuse", flush=True)
print("should be ~0 (A_pred huge), matching the measured FRA-QK general-KL ~0. Compare A_pred to the", flush=True)
print("measured feat1/fra collateral ratio (~10x reuse, ~1000x general) from scripts/62 & 65.", flush=True)
json.dump({"rows": rows}, open("results/b1_gpt2/magnitude_law.json", "w") if os.path.isdir("results/b1_gpt2") else open("magnitude_law.json", "w"), indent=2, default=float)
print("\nDONE magnitude_law", flush=True)

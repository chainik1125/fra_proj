"""REDTEAM_MECHANISM — adversarial probe of the acronym FRA win.
Tests three attacks:
  A) Are heads [8.11,9.9,10.10,11.4] actually the load-bearing ones for THIS probe?
     -> auto-discover top heads by (i) raw attn on Officer edge, (ii) causal edge-cut dP(O).
  B) Does FRA on-target removal discriminate content from position?
     -> FRA vs position-patch on the ORIGINAL probe (expect tie => on-target non-discriminating).
  C) Content-addressed vs position-generic: TRANSFER to probe-B with Officer at a NEW position,
     and a HARD generic control: does FRA suppress a NON-target key at the same position?
"""
import os, sys, json
sys.path.insert(0, "/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT = os.environ.get("OUTDIR", "/tmp"); dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval(); tok = model.tokenizer; W_U = model.W_U
PAPER = [(8, 11), (9, 9), (10, 10), (11, 4)]; LY = sorted(set(L for L, H in PAPER))
sae = {L: (lambda s: (s[0] if isinstance(s, tuple) else s))(SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev)) for L in range(12)}
def enc(s): return torch.tensor([tok.bos_token_id] + tok.encode(s), device=dev).unsqueeze(0)
def kpos(ids, sub, before):
    w = tok.encode(sub)[0]; c = [i for i, x in enumerate(ids) if x == w and i < before]; return c[-1] if c else None
def Pof(tt, ans, hooks=None):
    lg = (model.run_with_hooks(tt, fwd_hooks=hooks) if hooks else model(tt))[0]; return torch.softmax(lg[-1].float(), -1)[tok.encode(ans)[0]].item()
def fra_edge(tt, L, H):
    HK = f"blocks.{L}.hook_resid_pre"; fe = sae[L].encode(model.run_with_cache(tt, names_filter=lambda n: n == HK)[1][HK][0]).float()
    xh = fe @ sae[L].W_dec.float() + sae[L].b_dec.float()
    r = _build_fra_result(model, L, H, fe, sae[L].W_dec.float(), dev, top_k=None, rms_activations=xh, dec_norms=None, chunk_size=16, verbose=False)
    f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy(); return dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())

RES = {}
# ---------- ATTACK A: which heads are load-bearing for THIS probe? ----------
pA = "The Chief Executive Officer (CE"; tA = enc(pA); idsA = [tok.bos_token_id] + tok.encode(pA); QA = tA.shape[1] - 1; KA = kpos(idsA, " Officer", QA)
_, cache = model.run_with_cache(tA, names_filter=lambda n: n.endswith("hook_pattern"))
attn = {}
for L in range(12):
    pt = cache[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(12): attn[(L, H)] = float(pt[H, QA, KA].item())
top_attn = sorted(attn, key=lambda x: -attn[x])[:8]
# causal: cut each head's Officer edge alone, measure dP(O)
def cut_edge(tt, heads, Q, K):
    byL = {}
    for L, H in heads: byL.setdefault(L, []).append(H)
    hk = []
    for L, Hs in byL.items():
        def mk(Hs):
            def hook(s, hook):
                for H in Hs: s[0, H, Q, K] = -1e4
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
    return torch.softmax(model.run_with_hooks(tt, fwd_hooks=hk)[0][Q].float(), -1)[tok.encode("O")[0]].item()
B0 = Pof(tA, "O")
causal = {}
for L in range(12):
    for H in range(12): causal[(L, H)] = B0 - cut_edge(tA, [(L, H)], QA, KA)
top_causal = sorted(causal, key=lambda x: -causal[x])[:8]
RES["A_heads"] = {
    "paper_heads": [list(h) for h in PAPER],
    "top8_by_attn": [[list(h), round(attn[h], 3)] for h in top_attn],
    "top8_by_causal_dP": [[list(h), round(causal[h], 4)] for h in top_causal],
    "paper_in_top8_attn": [list(h) for h in PAPER if h in top_attn],
    "paper_in_top8_causal": [list(h) for h in PAPER if h in top_causal],
}
print("== ATTACK A: head load-bearingness ==", flush=True)
print(f"  paper heads: {PAPER}", flush=True)
print(f"  top8 by raw attn on Officer edge: {[(h, round(attn[h],2)) for h in top_attn]}", flush=True)
print(f"  top8 by causal dP(O) single-edge-cut: {[(h, round(causal[h],3)) for h in top_causal]}", flush=True)
print(f"  paper heads in causal top8: {RES['A_heads']['paper_in_top8_causal']}", flush=True)

# Build the FRA edit on PAPER heads (as the campaign did)
HF = {(L, H): fra_edge(tA, L, H) for L, H in PAPER}; P = {}
for (L, H) in PAPER:
    d = HF[(L, H)]; on = (d["qq"] == QA) & (d["kk"] == KA); oi = np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
    P[(L, H)] = set((int(d["ii"][o]), int(d["jj"][o])) for o in oi)
def fra_delta(d, Ps, sq):
    dd = np.zeros((sq, sq))
    for n in range(len(d["vv"])):
        if (int(d["ii"][n]), int(d["jj"][n])) in Ps: dd[d["qq"][n], d["kk"][n]] += d["vv"][n]
    return dd
def fra_hooks(tt, c=8):
    sq = tt.shape[1]; byL = {}
    for (L, H) in PAPER: byL.setdefault(L, {})[H] = torch.tensor(fra_delta(fra_edge(tt, L, H), P[(L, H)], sq), device=dev, dtype=torch.float32) * c
    hk = []
    for L, hd in byL.items():
        def mk(hd):
            def hook(s, hook):
                for H, dd in hd.items(): s[0, H, :dd.shape[0], :dd.shape[1]] -= dd.to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores", mk(hd)))
    return hk
def patch_hooks(tt, keypos):
    byL = {}
    for L, H in PAPER: byL.setdefault(L, []).append(H)
    hk = []
    for L, Hs in byL.items():
        def mk(Hs):
            def hook(s, hook):
                Q = s.shape[2] - 1
                if keypos is not None and keypos < s.shape[3]:
                    for H in Hs: s[0, H, Q, keypos] = -1e4
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
    return hk

# ---------- ATTACK B: on-target, does FRA discriminate content from position? ----------
on_fra = Pof(tA, "O", fra_hooks(tA, 8)); on_patch = Pof(tA, "O", patch_hooks(tA, KA))
RES["B_ontarget"] = {"base": B0, "fra": on_fra, "position_patch": on_patch,
                     "discriminates": abs(on_fra - on_patch) > 0.05}
print("\n== ATTACK B: on-target FRA vs position-patch (tie => on-target non-discriminating) ==", flush=True)
print(f"  base {B0:.3f} | FRA {on_fra:.3f} | position-patch@KA {on_patch:.3f}", flush=True)

# ---------- ATTACK C: content-addressed vs position-generic ----------
# C1: TRANSFER — Officer at a genuinely different position
pB = "the board has recently appointed a brand new Chief Executive Officer (CE"
tB = enc(pB); idsB = [tok.bos_token_id] + tok.encode(pB); QB = tB.shape[1] - 1; KB = kpos(idsB, " Officer", QB)
bB = Pof(tB, "O")
fB = Pof(tB, "O", fra_hooks(tB, 8)); p_naive = Pof(tB, "O", patch_hooks(tB, KA)); p_oracle = Pof(tB, "O", patch_hooks(tB, KB))
# C2: GENERIC-CONTROL — does the FRA edit (calibrated on Officer) suppress a DIFFERENT word's
#     initial when that word sits at the SAME position Officer occupied? If FRA is truly content-
#     addressed it should NOT; a position-generic edit would.
pC = "The Royal Air Force (R"  # 'Force'->F ; different content word at acronym-final slot
# Use a probe where a non-Officer capitalized word is the target, check FRA(Officer-pairs) leaves it alone:
pD = "The Federal Bureau of Investigation (FB"; tD = enc(pD); bD = Pof(tD, "I")
fD = Pof(tD, "I", fra_hooks(tD, 8))  # FRA-Officer-pairs should not touch 'Investigation'->I
RES["C_transfer"] = {"KA": KA, "KB": KB, "base_B": bB, "fra_B": fB,
                     "patch_naive_B": p_naive, "patch_oracle_B": p_oracle,
                     "fra_transfers": fB < 0.5 * bB, "naive_patch_fails": p_naive > 0.5 * bB}
RES["C_generic_control"] = {"FBI_base_P(I)": bD, "FBI_under_Officer_edit": fD,
                            "leaves_nontarget_alone": abs(fD - bD) < 0.05}
print("\n== ATTACK C: content-addressed vs position-generic ==", flush=True)
print(f"  Officer pos: probeA={KA} probeB={KB} (different: {KA != KB})", flush=True)
print(f"  TRANSFER probe-B base P(O) {bB:.3f}: FRA {fB:.3f} | patch@KA(naive) {p_naive:.3f} | patch@KB(oracle) {p_oracle:.3f}", flush=True)
print(f"  GENERIC CONTROL (FBI->I under Officer-pairs edit): base {bD:.3f} -> {fD:.3f} (should be ~unchanged)", flush=True)

# ---------- ATTACK D (novelty/single-SAE): re-run on-target with a different SAE layer set ----------
# Does the win hinge on resid_pre@{8,9,10,11}? Try restricting the edit to ONLY the strongest causal head.
strongest = top_causal[0]
RES["meta"] = {"strongest_causal_head": list(strongest), "strongest_causal_dP": round(causal[strongest], 4)}
json.dump(RES, open(os.path.join(OUT, "redteam_mechanism.json"), "w"), indent=2, default=float)
print("\nDONE redteam_mechanism -> " + os.path.join(OUT, "redteam_mechanism.json"), flush=True)

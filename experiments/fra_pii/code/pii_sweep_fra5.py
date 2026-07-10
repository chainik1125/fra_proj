"""pii_sweep_fra5.py — B3 M^B-SVD rank-r bilinear edit (condensate diagnostic).

Round 4 found: cutting the top digit-content feature-pairs at ALL (q,k) positions they occur
(content-addressed, not edge-restricted) suppresses emit by ~63% while PRESERVING lookup and
sibling and costing near-zero general-text KL -- the best selectivity signature in the campaign,
but the cut used discrete top-M CELL selection (mass-ranked (i,j) pairs), which is a magnitude
proxy, not causally validated per-pair.

This round replaces cell selection with a low-rank bilinear edit (the B3 M^B-SVD idea): build the
per-head coupling submatrix M[i,j] (SIGNED sum of FRA score contributions over active query-role
feature i / key-role feature j, restricted to cells whose key position is a target digit
position, capped to the top ~500 most-active i's and j's for a tractable dense SVD), decompose
M = U S V^T, and define the rank-r bilinear score reconstruction at ANY (q,k):
    score_r(q,k) = sum_{m<r} S[m] * (fe(q) . U[:,m]) * (fe(k) . V[:,m])
computed directly from each context's own encoded features (fe), so it applies (content-
addressed, exactly as round 4) to te/tl/tsib without re-deriving U/S/V.

Reports:
  - r_eff: the rank needed to reach ~most of round 4's M=ALL effect (condensate diagnostic --
    low r_eff means the redundant/distributed conjunction collapses onto few collective modes).
  - variance-explained per r (cheap, no forward passes -- straight from S).
  - single-mode peel-back on the first few individual modes: which ONE mode, causally tested
    alone, actually suppresses emit vs is collateral-only (the SVD is magnitude-based, so this
    validates rather than assumes the top modes are the causal ones).
"""
import os, sys, json
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
OUTDIR = os.environ.get("OUTDIR", os.environ.get("OUT_DIR", "."))
os.makedirs(OUTDIR, exist_ok=True)
TAG = os.environ.get("SWEEP_TAG", "fra5")

model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float16)
model.eval(); tok = model.tokenizer
NL, NH = model.cfg.n_layers, model.cfg.n_heads

BEST_HEADS = [(18, 6), (17, 3), (15, 0), (22, 3), (12, 2), (14, 5), (10, 5), (20, 7)]
NEED_LAYERS = sorted(set(L for L, _ in BEST_HEADS))
TOP_N = 500  # active features kept per side before SVD (dense TOP_NxTOP_N SVD, instant)


def db_fixed():
    return [
        {"name": "Marcus Webb",  "ssn": "531-42-8817", "city": "Portland"},
        {"name": "Elena Ortiz",  "ssn": "624-19-5530", "city": "Denver"},
        {"name": "Priya Nair",   "ssn": "708-63-2291", "city": "Bristol"},
    ]

def db_text(recs): return json.dumps(recs, indent=2)
def emit_prompt(recs, name): return db_text(recs) + f"\n\nThe SSN of {name} is "
def lookup_prompt(recs, ssn): return db_text(recs) + f"\n\nThe person with SSN {ssn} is named "

GENERAL = ["The weather today is", "She opened the door and", "In physics, energy is",
           "The recipe calls for two", "After the meeting they", "My favorite season is",
           "The train arrived at", "He picked up the phone and"]

def toks(s): return tok(s, return_tensors="pt").to(dev).input_ids
def _dig(s): return "".join(c for c in s if c.isdigit())
def emit_hit(cont, ssn): return _dig(ssn) in _dig(cont)

def find_subseq(hay, needle):
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle: return i
    return -1

def p_tok(t, tid, hooks=()):
    lg = model.run_with_hooks(t, fwd_hooks=list(hooks)) if hooks else model(t)
    return torch.softmax(lg[0, -1].float(), -1)[tid].item()

def gen_cont(t, hooks=(), n=14):
    cur = t
    for _ in range(n):
        lg = model.run_with_hooks(cur, fwd_hooks=list(hooks)) if hooks else model(cur)
        cur = torch.cat([cur, torch.tensor([[int(lg[0, -1].argmax())]], device=dev)], 1)
    return tok.decode(cur[0, t.shape[1]:], skip_special_tokens=True)

def gen_kl(hooks=()):
    tot = 0.0
    for s in GENERAL:
        t = toks(s)
        p = torch.log_softmax(model(t)[0, -1].float(), -1)
        q = torch.log_softmax(model.run_with_hooks(t, fwd_hooks=list(hooks))[0, -1].float(), -1)
        tot += (p.exp() * (p - q)).sum().item()
    return tot / len(GENERAL)

def fra_hooks(byL, c, sqx):
    hooks = []
    for L, hd in byL.items():
        td = {Hh: torch.tensor(dd, device=dev, dtype=torch.float32) * c for Hh, dd in hd.items()}
        def mk(td):
            def hook(s, hook):
                ql, kl = s.shape[2], s.shape[3]
                for Hh, sd in td.items():
                    qn, kn = min(ql, sd.shape[0]), min(kl, sd.shape[1])
                    s[0, Hh, :qn, :kn] -= sd[:qn, :kn].to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
    return hooks

# ---- setup ----
recs = db_fixed(); tgt, sib = recs[0], recs[1]
te = toks(emit_prompt(recs, tgt["name"])); ide = te[0].tolist(); sq = te.shape[1]
ssn_ids = tok(tgt["ssn"], add_special_tokens=False).input_ids
kpos = find_subseq(ide, ssn_ids); digit_kpositions = list(range(kpos, kpos + len(ssn_ids)))
first_id = ssn_ids[0]
base_emit = p_tok(te, first_id)
base_cont = gen_cont(te)
tl = toks(lookup_prompt(recs, tgt["ssn"])); sq_tl = tl.shape[1]
tsib = toks(emit_prompt(recs, sib["name"])); sq_tsib = tsib.shape[1]
base_look = gen_cont(tl); base_sib = gen_cont(tsib)
print(f"[base] emit P(first)={base_emit:.3f} look='{base_look[:24]}' sibemit='{base_sib[:24]}'", flush=True)

SAE = {L: GemmaScopeSAE("gemma-scope-2b-pt-res-canonical", f"layer_{L-1}/width_16k/canonical",
                        device=dev, normalize_activations=True) for L in NEED_LAYERS}

def enc_f(L, x):
    f = SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f = f / SAE[L]._norm_coeff
    return f

def fra_build(t, heads):
    _, c = model.run_with_cache(t, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in NEED_LAYERS])
    H = {}
    for (L, Hh) in heads:
        fe = enc_f(L, c[f"blocks.{L}.hook_resid_pre"][0]); xh = fe @ SAE[L].W_dec.float() + SAE[L].b_dec.float()
        r = _build_fra_result(model, L, Hh, fe, SAE[L].W_dec.float(), dev, top_k=None,
                              rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
        f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
        H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())
    return H

def get_fe(t, layers):
    _, c = model.run_with_cache(t, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in layers])
    return {L: enc_f(L, c[f"blocks.{L}.hook_resid_pre"][0]) for L in layers}

HF_te = fra_build(te, BEST_HEADS)
FE_te = get_fe(te, NEED_LAYERS); FE_tl = get_fe(tl, NEED_LAYERS); FE_tsib = get_fe(tsib, NEED_LAYERS)

# ---- coupling submatrix + SVD, per head ----
def build_coupling_and_svd(HF, kpositions, top_n):
    out = {}
    for (L, Hh), d in HF.items():
        loc = np.where(np.isin(d["kk"], kpositions))[0]
        if len(loc) == 0: out[(L, Hh)] = None; continue
        ii, jj, vv = d["ii"][loc], d["jj"][loc], d["vv"][loc]
        i_mass, j_mass = {}, {}
        for a, b, v in zip(ii, jj, np.abs(vv)):
            i_mass[a] = i_mass.get(a, 0.0) + v; j_mass[b] = j_mass.get(b, 0.0) + v
        top_i = np.array([k for k, _ in sorted(i_mass.items(), key=lambda x: -x[1])[:top_n]])
        top_j = np.array([k for k, _ in sorted(j_mass.items(), key=lambda x: -x[1])[:top_n]])
        i_idx = {v: n for n, v in enumerate(top_i)}; j_idx = {v: n for n, v in enumerate(top_j)}
        M = np.zeros((len(top_i), len(top_j)))
        for a, b, v in zip(ii, jj, vv):
            if a in i_idx and b in j_idx: M[i_idx[a], j_idx[b]] += v
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        out[(L, Hh)] = dict(active_i=top_i, active_j=top_j, U=U, S=S, Vt=Vt, M=M)
    return out

SVDH = build_coupling_and_svd(HF_te, digit_kpositions, TOP_N)
for (L, Hh), s in SVDH.items():
    if s is not None:
        cum = np.cumsum(s["S"] ** 2) / max((s["S"] ** 2).sum(), 1e-12)
        print(f"[svd] L{L}H{Hh}: coupling shape={s['M'].shape} top singular values={s['S'][:6].round(3).tolist()} "
              f"cumvar@r=1,2,4,8={[round(cum[min(r,len(cum))-1],3) for r in [1,2,4,8]]}", flush=True)

def rankr_delta(svdres, fe_layer, modes, sqx):
    """modes: list/array of singular-mode indices to include (order doesn't need to be 0..r-1)."""
    ai, aj = svdres["active_i"], svdres["active_j"]
    U = svdres["U"][:, modes]; S = svdres["S"][modes]; Vt = svdres["Vt"][modes, :]
    feq = fe_layer[:, ai].cpu().numpy(); fek = fe_layer[:, aj].cpu().numpy()
    QR = feq @ U; KR = fek @ Vt.T
    dd = (QR * S[None, :]) @ KR.T
    return dd[:sqx, :sqx]

def hooks_for(FE_ctx, modes, sqx):
    byL = {}
    for (L, Hh), svdres in SVDH.items():
        if svdres is None: continue
        dd = rankr_delta(svdres, FE_ctx[L], modes, sqx)
        byL.setdefault(L, {})[Hh] = dd
    return byL

def metrics(modes, c):
    byL_te = hooks_for(FE_te, modes, sq); byL_tl = hooks_for(FE_tl, modes, sq_tl); byL_tsib = hooks_for(FE_tsib, modes, sq_tsib)
    h_te = fra_hooks(byL_te, c, sq); h_tl = fra_hooks(byL_tl, c, sq_tl); h_tsib = fra_hooks(byL_tsib, c, sq_tsib)
    pe = p_tok(te, first_id, h_te)
    return dict(emit_p=pe, emit_supp=1 - pe / max(base_emit, 1e-6),
                emit_ok=emit_hit(gen_cont(te, h_te), tgt["ssn"]),
                look_ok=(tgt["name"].split()[1] in gen_cont(tl, h_tl)),
                sib_ok=emit_hit(gen_cont(tsib, h_tsib), sib["ssn"]),
                gen_kl=gen_kl(h_te))

def dump(payload):
    with open(os.path.join(OUTDIR, f"pii_sweep_{TAG}.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

results = dict(base_emit=base_emit,
               varexpl={f"L{L}H{Hh}": (np.cumsum(s["S"]**2)/max((s["S"]**2).sum(),1e-12))[:16].tolist()
                        for (L,Hh), s in SVDH.items() if s is not None})
rows = []
dump(dict(method="fra5", **results, rows=rows))

# ---- (1) main r-sweep (cumulative rank truncation), fixed c=20 ----
C_MAIN = 20
for r in [1, 2, 4, 8, 16, 32]:
    modes = list(range(r))
    m = metrics(modes, C_MAIN)
    print(f"[fra5] r={r} c={C_MAIN}: supp={m['emit_supp']:.3f} emit_ok={m['emit_ok']} "
          f"look_ok={m['look_ok']} sib_ok={m['sib_ok']} kl={m['gen_kl']:.4f}", flush=True)
    rows.append(dict(kind="fra5_rankr", r=r, c=C_MAIN, **m))
    dump(dict(method="fra5", **results, rows=rows))

# a couple of c values at the rank that looks most promising by supp so far
best_r = max([row["r"] for row in rows if row["emit_supp"] > 0], default=8,
             key=lambda rr: next(row["emit_supp"] for row in rows if row["r"] == rr))
for c in [45, 90]:
    m = metrics(list(range(best_r)), c)
    print(f"[fra5] r={best_r}(best) c={c}: supp={m['emit_supp']:.3f} emit_ok={m['emit_ok']} "
          f"look_ok={m['look_ok']} sib_ok={m['sib_ok']} kl={m['gen_kl']:.4f}", flush=True)
    rows.append(dict(kind="fra5_rankr_cscan", r=best_r, c=c, **m))
    dump(dict(method="fra5", **results, rows=rows))

# ---- (2) single-mode peel-back: is mode m ALONE causal, or collateral-only? ----
for m_idx in range(6):
    mm = metrics([m_idx], C_MAIN)
    print(f"[fra5] SINGLE mode={m_idx} c={C_MAIN}: supp={mm['emit_supp']:.3f} emit_ok={mm['emit_ok']} "
          f"look_ok={mm['look_ok']} sib_ok={mm['sib_ok']} kl={mm['gen_kl']:.4f}", flush=True)
    rows.append(dict(kind="fra5_singlemode", mode=m_idx, c=C_MAIN, **mm))
    dump(dict(method="fra5", **results, rows=rows))

print(f"[pii_sweep_fra5] DONE -> pii_sweep_{TAG}.json", flush=True)

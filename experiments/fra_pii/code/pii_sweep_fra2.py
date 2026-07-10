"""pii_sweep_fra2.py — round-2 FRA agent: locate the real causal edge, THEN sweep FRA on it.

Round-1 finding (pii_sweep_diag.json / pii_sweep_fra.json): on the 14 hand-picked candidate
heads (inherited from pii_cut.py), the ORACLE (literal answer->ssn-digit-position attn zeroed
on the causal-top-6 of those 14) suppresses emit P by only 1.8%, and FRA-cut-ALL (all cells,
c up to 60) on the SAME heads reproduces that number EXACTLY (0.01835 both) -> FRA is
mechanically correct (== oracle at saturation) but the *edge* (these heads, digit-token keys)
carries almost none of the emission signal. Sweeping M/c further there is a dead end by
construction: the oracle IS FRA's ceiling on that edge.

This script:
  (A) causal-scans ALL heads x ALL layers (no SAE, cheap) on THREE key-position hypotheses:
      'digit' (the literal ssn-digit tokens, same as round 1), 'name' (the target name-value
      tokens — maybe retrieval is name-keyed, not digit-keyed), 'brace' (the target record's
      closing '}' — maybe a record-summary/boundary token carries the compressed record, per
      the ROME-style "late MLP lookup, early attention gather" pattern).
  (B) takes the keydef with the highest 6-head oracle ceiling, loads SAE only for the layers
      that ceiling needs, builds FRA, and sweeps M x c on it — the same sweep round 1 ran, but
      now pointed at an edge with real causal weight instead of ~0.
"""
import os, sys, json, traceback
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
OUTDIR = os.environ.get("OUTDIR", os.environ.get("OUT_DIR", "."))
os.makedirs(OUTDIR, exist_ok=True)
TAG = os.environ.get("SWEEP_TAG", "fra2")

model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float16)
model.eval(); tok = model.tokenizer
NL, NH = model.cfg.n_layers, model.cfg.n_heads


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

def oracle_hooks(heads, qpos, kpositions):
    byL = {}
    for L, H in heads: byL.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in byL.items():
        def mk(Hs):
            def hook(s, hook):
                for H in Hs:
                    for kp in kpositions:
                        if kp < s.shape[3] and qpos < s.shape[2]: s[0, H, qpos, kp] = -1e4
                        if s.shape[2] == 1:
                            for kp in kpositions:
                                if kp < s.shape[3]: s[0, H, 0, kp] = -1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
    return hooks

# ---- setup: base prompts, target ssn/name key positions ----
recs = db_fixed(); tgt, sib = recs[0], recs[1]
dbtxt = db_text(recs)
te = toks(emit_prompt(recs, tgt["name"])); ide = te[0].tolist(); sq = te.shape[1]
ssn_ids = tok(tgt["ssn"], add_special_tokens=False).input_ids
kpos = find_subseq(ide, ssn_ids); digit_kpositions = list(range(kpos, kpos + len(ssn_ids)))
first_id = ssn_ids[0]; qpos = sq - 1
base_emit = p_tok(te, first_id)
base_cont = gen_cont(te)
tl = toks(lookup_prompt(recs, tgt["ssn"]))
tsib = toks(emit_prompt(recs, sib["name"]))
base_look = gen_cont(tl); base_sib = gen_cont(tsib)
print(f"[base] emit P(first)={base_emit:.3f} emit='{base_cont[:24]}' look='{base_look[:24]}' "
      f"sibemit='{base_sib[:24]}' kpos={kpos} sq={sq}", flush=True)

# ---- alternate key-position defs ----
name_ids = tok(tgt["name"], add_special_tokens=False).input_ids
npos = find_subseq(ide, name_ids)
name_kpositions = list(range(npos, npos + len(name_ids))) if npos >= 0 else []

c0 = dbtxt.index("{"); c1 = dbtxt.index("}", c0)  # record-0 span (first record in the array)
brace_kpositions = []
try:
    off = tok(dbtxt, return_offsets_mapping=True, add_special_tokens=True)["offset_mapping"]
    brace_kpositions = [i for i, (a, b) in enumerate(off) if a < c1 <= b or a == b == c1]
except Exception:
    print("[keydefs] offset_mapping unsupported, brace keydef skipped", flush=True)
print(f"[keydefs] digit={digit_kpositions} name={name_kpositions} brace={brace_kpositions}", flush=True)

KEYDEFS = {"digit": digit_kpositions, "name": name_kpositions, "brace": brace_kpositions}
KEYDEFS = {k: v for k, v in KEYDEFS.items() if v}

# ---- (A) full head x layer x keydef causal scan, NO SAE ----
scan = {}
for kdname, kps in KEYDEFS.items():
    drops = []
    for L in range(NL):
        for H in range(NH):
            p = p_tok(te, first_id, oracle_hooks([(L, H)], qpos, kps))
            drops.append(((L, H), base_emit - p))
    drops.sort(key=lambda x: -x[1])
    scan[kdname] = drops
    print(f"[scanA] keydef={kdname} top8 single-head drops: "
          f"{[(f'L{L}H{H}', round(d,4)) for (L,H),d in drops[:8]]}", flush=True)

oracle_ceiling = {}
for kdname, kps in KEYDEFS.items():
    for ntop in (6, 12):
        heads = [h for h, _ in scan[kdname][:ntop]]
        p = p_tok(te, first_id, oracle_hooks(heads, qpos, kps))
        supp = 1 - p / max(base_emit, 1e-6)
        print(f"[scanA] ORACLE ceiling keydef={kdname} top{ntop}: emit_p={p:.3f} supp={supp:.3f}", flush=True)
        if kdname not in oracle_ceiling or supp > oracle_ceiling[kdname]["emit_supp"]:
            oracle_ceiling[kdname] = dict(heads=heads, ntop=ntop, emit_p=p, emit_supp=supp)

best_kd = max(oracle_ceiling, key=lambda k: oracle_ceiling[k]["emit_supp"])
print(f"[scanA] BEST keydef={best_kd} supp={oracle_ceiling[best_kd]['emit_supp']:.3f} "
      f"heads={[f'L{L}H{H}' for L,H in oracle_ceiling[best_kd]['heads']]}", flush=True)

def dump(rows=None):
    out = dict(method="fra2", base_emit=base_emit,
               keydefs={k: list(v) for k, v in KEYDEFS.items()},
               scan_top={k: [(f"L{L}H{H}", d) for (L, H), d in v[:12]] for k, v in scan.items()},
               oracle_ceiling={k: dict(heads=[f"L{L}H{H}" for L, H in v["heads"]], ntop=v["ntop"],
                                        emit_p=v["emit_p"], emit_supp=v["emit_supp"])
                                for k, v in oracle_ceiling.items()},
               best_keydef=best_kd)
    if rows is not None: out["rows"] = rows
    with open(os.path.join(OUTDIR, f"pii_sweep_{TAG}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)

dump()

# ---- (B) FRA build + M x c sweep on the winning (heads, keydef) ----
best_heads = oracle_ceiling[best_kd]["heads"][:8]
best_kps = KEYDEFS[best_kd]
need_layers = sorted(set(L for L, _ in best_heads))
print(f"[scanC] loading SAE for layers {need_layers}", flush=True)
SAE = {L: GemmaScopeSAE("gemma-scope-2b-pt-res-canonical", f"layer_{L-1}/width_16k/canonical",
                        device=dev, normalize_activations=True) for L in need_layers}

def enc_f(L, x):
    f = SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f = f / SAE[L]._norm_coeff
    return f

def fra_build(t, heads):
    _, c = model.run_with_cache(t, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in need_layers])
    H = {}
    for (L, Hh) in heads:
        fe = enc_f(L, c[f"blocks.{L}.hook_resid_pre"][0]); xh = fe @ SAE[L].W_dec.float() + SAE[L].b_dec.float()
        r = _build_fra_result(model, L, Hh, fe, SAE[L].W_dec.float(), dev, top_k=None,
                              rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
        f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
        H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())
    return H

def fra_delta_topM(HF, qposx, kpositions, sqx, M):
    byL = {}
    for (L, Hh), d in HF.items():
        mask = (d["qq"] == qposx) & np.isin(d["kk"], kpositions)
        loc = np.where(mask)[0]
        if len(loc):
            order = loc[np.argsort(-np.abs(d["vv"][loc]))]
            keep = order if (M is None or M >= len(order)) else order[:M]
            dd = np.zeros((sqx, sqx))
            for o in keep: dd[d["qq"][o], d["kk"][o]] += d["vv"][o]
            byL.setdefault(L, {})[Hh] = dd
    return byL

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

# sibling key positions under the SAME keydef, on the sibling emit prompt
idsib = tsib[0].tolist()
if best_kd == "digit":
    sib_ssn_ids = tok(sib["ssn"], add_special_tokens=False).input_ids
    p0 = find_subseq(idsib, sib_ssn_ids); sib_kps = list(range(p0, p0 + len(sib_ssn_ids))) if p0 >= 0 else []
elif best_kd == "name":
    sib_name_ids = tok(sib["name"], add_special_tokens=False).input_ids
    p0 = find_subseq(idsib, sib_name_ids); sib_kps = list(range(p0, p0 + len(sib_name_ids))) if p0 >= 0 else []
else:  # brace: record-0 text is an identical shared prefix across emit_prompt(tgt) / emit_prompt(sib)
    sib_kps = best_kps

HF = fra_build(te, best_heads)
HFsib = fra_build(tsib, best_heads)

def metrics(emit_hooks, sib_hooks):
    pe = p_tok(te, first_id, emit_hooks)
    return dict(
        emit_p=pe, emit_supp=1 - pe / max(base_emit, 1e-6),
        emit_ok=emit_hit(gen_cont(te, emit_hooks), tgt["ssn"]),
        sib_ok=emit_hit(gen_cont(tsib, sib_hooks), sib["ssn"]) if sib_hooks or not sib_kps else emit_hit(gen_cont(tsib), sib["ssn"]),
        gen_kl=gen_kl(emit_hooks),
    )

sweep_rows = []
for M in [50, 200, 1000, None]:
    byL = fra_delta_topM(HF, qpos, best_kps, sq, M)
    byLsib = fra_delta_topM(HFsib, tsib.shape[1] - 1, sib_kps, tsib.shape[1], M) if sib_kps else {}
    for c in [8, 20, 45, 90]:
        m = metrics(fra_hooks(byL, c, sq), fra_hooks(byLsib, c, tsib.shape[1]) if byLsib else ())
        lab = f"kd={best_kd},M={'ALL' if M is None else M},c={c}"
        print(f"[fra2] {lab}: supp={m['emit_supp']:.2f} emit_ok={m['emit_ok']} sib_ok={m['sib_ok']} kl={m['gen_kl']:.3f}", flush=True)
        sweep_rows.append(dict(kind="fra2", keydef=best_kd, heads=[f"L{L}H{H}" for L, H in best_heads],
                                M=(-1 if M is None else M), c=c, **m))
        dump(sweep_rows)

# lookup preservation at the best operating point (highest supp among sib_ok==True rows)
ok_rows = [r for r in sweep_rows if r["sib_ok"]]
best_row = max(ok_rows, key=lambda r: r["emit_supp"]) if ok_rows else max(sweep_rows, key=lambda r: r["emit_supp"])
M = None if best_row["M"] == -1 else best_row["M"]; c = best_row["c"]
idl = tl[0].tolist(); klp = find_subseq(idl, ssn_ids)
look_kpos_digit = list(range(klp, klp + len(ssn_ids))) if klp >= 0 else []
if look_kpos_digit:
    HFlook = fra_build(tl, best_heads)
    byLlook = fra_delta_topM(HFlook, tl.shape[1] - 1, look_kpos_digit, tl.shape[1], M)
    look_cont = gen_cont(tl, fra_hooks(byLlook, c, tl.shape[1]))
else:
    look_cont = gen_cont(tl)
look_ok = tgt["name"].split()[1] in look_cont
best_row["look_ok"] = look_ok
print(f"[fra2] BEST operating point: {best_row}", flush=True)

dump(sweep_rows)
print(f"[pii_sweep_fra2] DONE -> pii_sweep_{TAG}.json", flush=True)

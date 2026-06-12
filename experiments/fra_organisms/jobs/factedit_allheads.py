"""DECISIVE all-heads oracle edge-cut test (head-localization / edit-completeness lens).

The factedit NO-GO rests on: top-3 causally-ranked extraction heads' subject->last attention
edge-cut leaves P(target) intact on strongly-recalled facts -> "redundant/G-post". My lens:
maybe the top-3 MISLOCATED the edge, or the edge is DISTRIBUTED across many heads so top-3 misses it.

DECISIVE: zero the subject->last attention SCORE at EVERY head, EVERY layer (all 208 heads) and
measure R = 1 - P_cut/P_base on hi-band (P>0.6) recalled facts. Compare to top-3.
  - all-heads R ~ 0 (R<0.1) on recalled facts  => redundancy confirmed, localization MOOT -> negative ROBUST.
  - all-heads R >> top-3 R (substantial drop)   => edge IS cuttable, just not at 3 heads -> negative is a
                                                   LOCALIZATION ARTIFACT (false negative).

Also tests:
  - all-heads cut of ALL subject-span tokens -> last (catches multi-token subject; strongest transport ablation).
  - per-fact top-3 (reproduce diag) and top-8 (does adding heads compound?).
  - Paris sanity at all-heads (machinery check + headroom: how big can all-heads get on a known load-bearing fact).
Ground-truth (logit) only. Modal A10G, gemma-2-2b-it. Mirrors diag.py's cut machinery EXACTLY.
"""
import pathlib, os, modal

_p = pathlib.Path(__file__).resolve()
ROOT = next((q for q in _p.parents if (q / "fra" / "core" / "fra.py").exists()), _p.parent)
app = modal.App("factedit-allheads")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "pandas", "transformer_lens==2.11.0",
                 "transformers==4.44.2", "tokenizers==0.19.1", "huggingface_hub==0.24.6",
                 "datasets==2.21.0")
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
)


@app.function(gpu="A10G", image=image, timeout=2400)
def run():
    import time, json
    import torch, numpy as np
    torch.set_grad_enabled(False)
    from transformer_lens import HookedTransformer
    dev = "cuda"
    MODEL_NAME = "gemma-2-2b-it"
    N_PER_BAND = 12
    N_CAND = 1500

    def log(*a): print(*a, flush=True)
    log(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=dev, dtype=torch.float16); model.eval()
    tok = model.tokenizer; NL = model.cfg.n_layers; NH = model.cfg.n_heads
    log(f"  {NL} layers x {NH} heads = {NL*NH} heads total")
    ALL_HEADS = [(L, H) for L in range(NL) for H in range(NH)]

    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("NeelNanda/counterfact-tracing",
          "data/train-00000-of-00001-36693f2cad948c42.parquet", repo_type="dataset"))

    def tgt0(t):
        ids = tok.encode(t, add_special_tokens=False); return ids[0] if ids else None
    def toks(prompt):
        ids = tok.encode(prompt); return ids, torch.tensor(ids, device=dev).unsqueeze(0)
    def keypos(ids, s, q):
        sids = tok.encode(s, add_special_tokens=False)
        if not sids: return None
        last = sids[-1]; c = [i for i, x in enumerate(ids) if x == last and i < q]
        return c[-1] if c else None
    def subj_span(ids, s, q):
        """All positions of the subject's token-span inside the prompt (best-effort contiguous match)."""
        sids = tok.encode(s, add_special_tokens=False)
        if not sids: return []
        n = len(sids)
        for start in range(0, q - n + 1):
            if ids[start:start+n] == sids:
                return list(range(start, start+n))
        # fallback: just the last-subject-token key
        k = keypos(ids, s, q)
        return [k] if k is not None else []

    def cut_edge(tt, heads, q, klist):
        """Zero score at (q, k) for every k in klist, at every head in `heads`. (=-1e4, like diag.)"""
        if isinstance(klist, int): klist = [klist]
        byL = {}
        for L, H in heads: byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(s, hook):
                    for H in Hs:
                        for k in klist: s[0, H, q, k] = -1e4
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
    def pt(tt, q, tid): return torch.softmax(model(tt)[0][q].float(), -1)[tid].item()
    def R_cut(tt, q, klist, tid, heads):
        base = pt(tt, q, tid)
        if base <= 0: return float("nan"), base
        pc = torch.softmax(cut_edge(tt, heads, q, klist)[q].float(), -1)[tid].item()
        return 1 - pc / base, base
    def causal_top(tt, q, k, tid, kmax=8):
        base = pt(tt, q, tid); dd = {}
        for L in range(NL):
            for H in range(NH):
                pc = torch.softmax(cut_edge(tt, [(L, H)], q, k)[q].float(), -1)[tid].item()
                dd[(L, H)] = base - pc
        top = sorted(dd, key=lambda x: -dd[x])[:kmax]
        return top

    out = {"config": {"model": MODEL_NAME, "n_heads_total": NL * NH}, "sanity": None, "bands": {}}

    # ---- SANITY: Paris, all-heads ----
    ids, tt = toks("The capital of France is"); q = tt.shape[1] - 1
    k = keypos(ids, " France", q); tid = tgt0(" Paris")
    span = subj_span(ids, " France", q)
    top3 = causal_top(tt, q, k, tid, 3); top8 = causal_top(tt, q, k, tid, 8)
    R_t3, base = R_cut(tt, q, k, tid, top3)
    R_t8, _ = R_cut(tt, q, k, tid, top8)
    R_all, _ = R_cut(tt, q, k, tid, ALL_HEADS)
    R_all_span, _ = R_cut(tt, q, span, tid, ALL_HEADS)
    out["sanity"] = dict(base=float(base), R_top3=float(R_t3), R_top8=float(R_t8),
                         R_allheads=float(R_all), R_allheads_subjspan=float(R_all_span),
                         span_len=len(span))
    log(f"  SANITY Paris: base={base:.3f} R_top3={R_t3:.3f} R_top8={R_t8:.3f} "
        f"R_ALLHEADS={R_all:.3f} R_ALLHEADS_span={R_all_span:.3f}")

    # ---- build hi/mid/lo bands (same seed=3 as diag for comparability) ----
    BANDS = {"lo": (0.03, 0.15), "mid": (0.15, 0.40), "hi": (0.60, 0.999)}
    pool = {b: [] for b in BANDS}; need = {b: N_PER_BAND for b in BANDS}
    df_s = df.sample(n=min(N_CAND, len(df)), random_state=3).reset_index(drop=True)
    for _, row in df_s.iterrows():
        if all(len(pool[b]) >= need[b] for b in BANDS): break
        prompt = str(row["prompt"]); subject = str(row["subject"]); tgt = str(row["target_true"])
        tid = tgt0(tgt)
        if tid is None: continue
        ids, tt = toks(prompt)
        if tt.shape[1] < 3 or tt.shape[1] > 48: continue
        q = tt.shape[1] - 1
        if keypos(ids, subject, q) is None: continue
        p = pt(tt, q, tid)
        for b, (lo, hi) in BANDS.items():
            if lo <= p < hi and len(pool[b]) < need[b]:
                pool[b].append(dict(prompt=prompt, subject=subject, tid=int(tid),
                                    p_base=float(p), relation_id=str(row["relation_id"]))); break
    log("  band sizes: " + ", ".join(f"{b}={len(pool[b])}" for b in BANDS))

    # ---- per fact: top3 / top8 / ALL-heads, on last-subject-key and on full subject-span ----
    for b in BANDS:
        out["bands"][b] = []
        for fi, fact in enumerate(pool[b]):
            ids, tt = toks(fact["prompt"]); q = tt.shape[1] - 1; tid = fact["tid"]
            k = keypos(ids, fact["subject"], q); span = subj_span(ids, fact["subject"], q)
            top3 = causal_top(tt, q, k, tid, 3); top8 = causal_top(tt, q, k, tid, 8)
            R_t3, base = R_cut(tt, q, k, tid, top3)
            R_t8, _ = R_cut(tt, q, k, tid, top8)
            R_all, _ = R_cut(tt, q, k, tid, ALL_HEADS)
            R_all_span, _ = R_cut(tt, q, span, tid, ALL_HEADS)
            rec = dict(prompt=fact["prompt"], p_base=fact["p_base"], relation_id=fact["relation_id"],
                       R_top3=float(R_t3), R_top8=float(R_t8), R_allheads=float(R_all),
                       R_allheads_subjspan=float(R_all_span), span_len=len(span))
            out["bands"][b].append(rec)
            log(f"  [{b} {fi:2d}] p={fact['p_base']:.2f} R_top3={R_t3:.3f} R_top8={R_t8:.3f} "
                f"R_ALL={R_all:.3f} R_ALL_span={R_all_span:.3f}")

    # ---- aggregate ----
    def f3(xs): return [float(np.median(xs)), float(np.mean(xs)), float(np.max(xs))] if xs else None
    agg = {}
    for b in BANDS:
        rows = out["bands"][b]
        agg[b] = dict(n=len(rows),
            R_top3=f3([r["R_top3"] for r in rows]),
            R_top8=f3([r["R_top8"] for r in rows]),
            R_allheads=f3([r["R_allheads"] for r in rows]),
            R_allheads_subjspan=f3([r["R_allheads_subjspan"] for r in rows]),
            frac_allheads_ge50=float(np.mean([r["R_allheads"] >= 0.5 for r in rows])),
            frac_allheads_span_ge50=float(np.mean([r["R_allheads_subjspan"] >= 0.5 for r in rows])),
            frac_allheads_ge20=float(np.mean([r["R_allheads"] >= 0.2 for r in rows])))
    out["aggregate"] = agg
    log("\n========= ALL-HEADS ORACLE SUMMARY (med,mean,max) =========")
    log(f"  SANITY Paris: R_top3={out['sanity']['R_top3']:.3f} R_allheads={out['sanity']['R_allheads']:.3f} "
        f"R_allheads_span={out['sanity']['R_allheads_subjspan']:.3f}")
    for b in BANDS:
        a = agg[b]
        log(f"  {b:3s}: R_top3={[round(x,3) for x in a['R_top3']]} R_allheads={[round(x,3) for x in a['R_allheads']]} "
            f"R_allheads_span={[round(x,3) for x in a['R_allheads_subjspan']]} "
            f"frac_all>=50={a['frac_allheads_ge50']} frac_allspan>=50={a['frac_allheads_span_ge50']}")
    return out


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    d = pathlib.Path("/tmp/fe_res/fra_org_factedit/results/allheads"); d.mkdir(parents=True, exist_ok=True)
    (d / "allheads.json").write_text(json.dumps(res, indent=2))
    print("WROTE", d / "allheads.json")
    print(json.dumps(res["aggregate"], indent=2))
    print("SANITY", json.dumps(res["sanity"], indent=2))

"""Bridge 1 — toy optimal-FRA carrier law -> Gemma-2-2b in-context induction backdoor.

Two deliverables, one Modal run (Gemma+SAE load is the cost, so share it):
  CARRIER  (out/bridge1_carrier.json): FRA-QK-cut vs FRA-OV-cut asymmetry on the real
           induction backdoor + gauge-robustness of the QK cut (RMSNorm-fold band).
           Prediction (amended fra_win law): induction is an obligate MATCH -> QK-carried,
           so the FRA-QK score cut carries the ASR effect at faithful scale c~1-2 and is
           gauge-robust, while the value-path (OV) cut is comparatively inert / pays more
           held-out collateral.  This is the MIRROR of the toy gate (OV-over-QK 21-36x).
  FRONTIER (out/bridge1_frontier.json): FRA-QK vs DoM vs conv-SAE vs payload-suppress,
           swept to matched ASR removal, held-out KL collateral.  Prediction: FRA-QK
           dominates (lowest collateral at matched removal).

Prints CHECKPOINT_CARRIER / CHECKPOINT_FRONTIER JSON lines so a mid-run crash still
leaves recoverable numbers in the .log.  Figures are rendered on Modal (matplotlib in
the image) and returned as base64.

Run (background, poll the log):
  cd experiments/constrained_belief_updating/bridge_to_real
  HF_TOKEN=$HF_TOKEN ../../../.venv/bin/python -m modal run bridge1.py > bridge1.log 2>&1
"""
import os, pathlib, modal

_HERE = pathlib.Path(__file__).resolve()
# On the remote container __file__ is /root/bridge1.py (only 2 parents); the local
# repo layout has fra_proj 4 levels up. Guard so module import doesn't crash remotely.
REPO = _HERE.parents[3] if len(_HERE.parents) > 3 else pathlib.Path("/root")
FRA_DIR = REPO / "fra"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "transformers==4.44.2", "transformer_lens", "sae_lens",
        "numpy", "matplotlib", "huggingface_hub",
    )
    .add_local_dir(str(FRA_DIR), "/root/fra")
)

app = modal.App("bridge1-carrier-frontier")


@app.function(gpu="A10G", image=image, timeout=3600,
              secrets=[modal.Secret.from_dict(
                  {"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def run():
    import sys, json, base64, io
    sys.path.insert(0, "/root")
    import torch, numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from transformer_lens import HookedTransformer
    from fra.sae_lens_wrapper import GemmaScopeSAE
    from fra.core.fra import _build_fra_result
    from fra.core.helpers import get_W_V

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float16)
    model.eval(); tok = model.tokenizer
    Llast = model.cfg.n_layers - 1; W_U = model.W_U
    print(f"model loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H", flush=True)

    # ── induction sequence helpers ──────────────────────────────────────
    def mkseq(seed, Tid=None, Pid=None, n=20):
        g = torch.Generator().manual_seed(seed)
        Rr = (torch.randperm(40000, generator=g)[:n] + 1000).tolist()
        if Tid is not None:
            Rr[10] = Tid; Rr[11] = Pid
        return torch.tensor([tok.bos_token_id] + Rr + Rr, device=dev).unsqueeze(0)

    # ── induction heads: attention screen, then CAUSAL verification ─────
    N0 = 24
    g = torch.Generator().manual_seed(0)
    R = (torch.randperm(40000, generator=g)[:N0] + 1000).tolist()
    tt0 = torch.tensor([tok.bos_token_id] + R + R, device=dev).unsqueeze(0)
    edges0 = [(1 + N0 + t, t + 2) for t in range(N0 - 1)]
    _, c0 = model.run_with_cache(tt0, names_filter=lambda n: n.endswith("hook_pattern"))
    strength = {}
    for L in range(model.cfg.n_layers):
        p = c0[f"blocks.{L}.attn.hook_pattern"][0]
        for H in range(p.shape[0]):
            strength[(L, H)] = float(np.mean([p[H, q, k].item() for q, k in edges0]))
    cand = [lh for lh, s in sorted(strength.items(), key=lambda x: -x[1]) if s > 0.4][:12]

    # causal: cut each candidate head's induction edges, keep those that lower copy prob
    def cut_edges_scores(tt, heads, edges, val=-1e4):
        byL = {}
        for L, H in heads:
            byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(s, hook):
                    for H in Hs:
                        for (q, k) in edges:
                            s[0, H, q, k] = val
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

    # measure copy: mean over induction edges of P(copied token) drop
    causal = {}
    for (L, H) in cand:
        drops = []
        for t in range(5, N0 - 1, 4):
            q = 1 + N0 + t; k = t + 2; copied = R[t + 1]
            base = torch.softmax(model(tt0)[0][q].float(), -1)[copied].item()
            cutp = torch.softmax(cut_edges_scores(tt0, [(L, H)], [(q, k)])[q].float(), -1)[copied].item()
            drops.append(base - cutp)
        causal[(L, H)] = float(np.mean(drops))
    IND = [lh for lh, d in sorted(causal.items(), key=lambda x: -x[1]) if d > 0.02][:8]
    if not IND:
        IND = cand[:6]
    LAYERS = sorted(set(L for L, H in IND)); L0 = min(LAYERS); DL = 6
    print("candidate (attn>0.4):", [f"L{L}H{H}" for L, H in cand], flush=True)
    print("induction heads (causal):", [f"L{L}H{H}({causal.get((L,H),0):.2f})" for L, H in IND], flush=True)

    # ── SAEs (65k GemmaScope on resid_pre of the FRA layers + DoM layer) ─
    SAE = {}
    for L in sorted(set(LAYERS) | {DL}):
        sl = L - 1
        try:
            SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",
                                   f"layer_{sl}/width_65k/canonical",
                                   device=dev, normalize_activations=True)
        except Exception:
            SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res",
                                   f"layer_{sl}/width_65k/average_l0_72",
                                   device=dev, normalize_activations=True)
    dsae = SAE[DL]
    print("SAEs loaded", flush=True)

    def encode(L, x):
        f = SAE[L].encode(x.float()).float()
        if SAE[L]._norm_coeff is not None:
            f = f / SAE[L]._norm_coeff
        return f

    # ── FRA build (per head) with selectable rms gauge ──────────────────
    def fra_ph(tt, rms_mode="xhat"):
        _, c = model.run_with_cache(
            tt, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
        H = {}; resid = {L: c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
        for (L, Hh) in IND:
            fe = encode(L, resid[L])
            xh = fe @ SAE[L].W_dec.float() + SAE[L].b_dec.float()
            if rms_mode == "xhat":
                rms_act = xh
            elif rms_mode == "resid":
                rms_act = resid[L].float()
            else:  # "none"
                rms_act = None
            r = _build_fra_result(model, L, Hh, fe, SAE[L].W_dec.float(), dev,
                                  top_k=None, rms_activations=rms_act, dec_norms=None,
                                  chunk_size=8, verbose=False)
            f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
            H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3],
                              vv=f.values().cpu().numpy())
        return H, resid

    def primer_pairs(HF, edge, M=12):
        P = {}
        for (L, Hh) in IND:
            d = HF[(L, Hh)]
            loc = np.where((d["qq"] == edge[0]) & (d["kk"] == edge[1]))[0]
            loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
            P[(L, Hh)] = set((int(d["ii"][o]), int(d["jj"][o])) for o in loc)
        return P

    def delta_content(HF, P, seq):
        byL = {}
        for (L, Hh) in IND:
            d = HF[(L, Hh)]; dd = np.zeros((seq, seq)); Ps = P[(L, Hh)]
            for n in range(len(d["vv"])):
                if (int(d["ii"][n]), int(d["jj"][n])) in Ps:
                    dd[d["qq"][n], d["kk"][n]] += d["vv"][n]
            byL.setdefault(L, {})[Hh] = dd
        return byL

    def patch_fra(tt, byL, c):
        seq = tt.shape[1]; hooks = []
        for L, hd in byL.items():
            td = {Hh: torch.tensor(dd, device=dev, dtype=torch.float32) * c
                  for Hh, dd in hd.items()}
            def mk(td):
                def hook(s, hook):
                    for Hh, sd in td.items():
                        s[0, Hh, :seq, :seq] = s[0, Hh, :seq, :seq] - sd.to(s.dtype)
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

    # ── FRA-OV cut: remove primer KEY-features from the value path ───────
    # Mirror of the QK cut: same feature-pairs, but project their key-side feature j
    # out of the head's value at every position where j is active (scaled by activation).
    # Value direction of feature j for head (L,H) obtained by finite-difference through
    # the real ln1 (RMSNorm gain) + W_V  ->  no gain/rms assumption.
    def ov_dirs(P, resid):
        dirs = {}  # (L,H) -> {j: unit d_head dir}
        base_cos = []
        for (L, Hh) in IND:
            W_V = get_W_V(model, L, Hh).float()          # [d_model, d_head]
            base = resid[L][12].float()                   # a representative position
            # ln1 runs in the model dtype (fp16); force fp32 before the fp32 W_V matmul.
            x0 = model.blocks[L].ln1(base[None, :])[0].float()
            jset = set(j for (i, j) in P[(L, Hh)])
            dd = {}
            for j in jset:
                x1 = model.blocks[L].ln1((base + 1e-2 * SAE[L].W_dec[j].float())[None, :])[0].float()
                v = ((x1 - x0) / 1e-2) @ W_V
                nv = v.norm()
                if nv > 1e-6:
                    dd[j] = v / nv
                    # analytic cross-check (gain*W_dec[j])@W_V direction; ln1.w only
                    # exists when RMSNorm is NOT folded (from_pretrained folds by
                    # default, so guard — the finite-diff dir already folds the gain
                    # via W_V and is the load-bearing one).
                    try:
                        ga = model.blocks[L].ln1.w.float()
                        va = (SAE[L].W_dec[j].float() * (1.0 + ga)) @ W_V
                        if va.norm() > 1e-6:
                            base_cos.append(float((v @ va) / (nv * va.norm())))
                    except AttributeError:
                        pass
            dirs[(L, Hh)] = dd
        return dirs, (float(np.mean(base_cos)) if base_cos else float("nan"))

    def ov_cut(tt, P, dirs, c, feats_by_pos):
        # feats_by_pos: {L: {pos: {j: activation}}}  precomputed for this tt
        hooks = []
        for (L, Hh) in IND:
            dd = dirs[(L, Hh)]
            def mk(dd, L):
                def hook(v, hook):
                    # v: [1, seq, n_kv_or_nheads, d_head]
                    nkv = v.shape[2]; grp = model.cfg.n_heads // nkv
                    kvh = Hh // grp
                    fbp = feats_by_pos[L]
                    for pos, acts in fbp.items():
                        vk = v[0, pos, kvh, :].float()
                        for j, dhat in dd.items():
                            if j in acts:
                                vk = vk - c * (vk @ dhat) * dhat
                        v[0, pos, kvh, :] = vk.to(v.dtype)
                    return v
                return hook
            hooks.append((f"blocks.{L}.attn.hook_v", mk(dd, L)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

    def feats_by_pos_for(tt, P, tol=1e-4):
        _, c = model.run_with_cache(
            tt, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
        out = {}
        for L in LAYERS:
            fe = encode(L, c[f"blocks.{L}.hook_resid_pre"][0])
            jset = set()
            for (Lh, Hh) in IND:
                if Lh == L:
                    jset |= set(j for (i, j) in P[(Lh, Hh)])
            byp = {}
            for pos in range(fe.shape[0]):
                acts = {j: float(fe[pos, j]) for j in jset if abs(float(fe[pos, j])) > tol}
                if acts:
                    byp[pos] = acts
            out[L] = byp
        return out

    # ── other removal methods (frontier) ────────────────────────────────
    def dom_run(tt, positions, vD, a):
        def hook(act, hook):
            for p in positions:
                if p < tt.shape[1]:
                    act[0, p, :] = act[0, p, :] - (a * vD * act[0, p, :].float().norm()).to(act.dtype)
            return act
        return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]

    def conv_run(tt, fidx, c):
        Wd = dsae.W_dec[fidx].float()
        def hook(act, hook):
            x = act[0].float(); z = encode(DL, x)[:, fidx]
            act[0] = (x - (c * z) @ Wd).to(act.dtype); return act
        return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]

    def paysupp(tt, Pid, s):
        uP = W_U[:, Pid].float(); uP = uP / uP.norm()
        def hook(act, hook):
            act[0] = act[0] - (s * (act[0].float() @ uP).unsqueeze(-1) * uP).to(act.dtype); return act
        return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{Llast}.hook_resid_post", hook)])[0]

    def klsum(p, q):
        lp = torch.log_softmax(p.float(), -1); lq = torch.log_softmax(q.float(), -1)
        return (lp.exp() * (lp - lq)).sum(-1).sum().item()

    # ── cases ───────────────────────────────────────────────────────────
    CASES = [
        (" bank", " river", "The river bank was crowded. He left the bank and crossed the river. By dusk the river hid the bank."),
        (" king", " crown", "The king lost his crown. A new crown was made. The old king wore the crown while the young king watched."),
        (" doctor", " water", "The doctor drank water. More water spilled. The doctor gave the patient water before the next doctor came."),
        (" market", " gold", "The market sold gold. Gold prices fell. The market reopened and gold buyers crowded the market again."),
    ]
    Ni = 20; qpos = 1 + Ni + 10; kpos = 12
    FC = [1, 2, 4, 8, 16, 32]; OC = [0.5, 1, 2, 4, 8, 16]
    DC = [0.25, 0.5, 1, 2, 4, 8]; CC = [1, 2, 4, 8, 16, 32]; PC = [0.5, 1, 2, 4, 8]

    carrier_rows = []; frontier_rows = []; gauge_cos = None
    for ci, (T, P, htext) in enumerate(CASES):
        Tid = tok.encode(T, add_special_tokens=False); Pid = tok.encode(P, add_special_tokens=False)
        if len(Tid) != 1 or len(Pid) != 1:
            continue
        Tid = Tid[0]; Pid = Pid[0]
        tt = mkseq(0, Tid, Pid); seq = tt.shape[1]
        base = torch.softmax(model(tt)[0][qpos].float(), -1)[Pid].item()
        if base < 0.2:
            print(f"skip {T.strip()} ASR {base:.2f}", flush=True); continue
        def asr_s(lg): return 1 - torch.softmax(lg[qpos].float(), -1)[Pid].item() / base

        # FRA (canonical gauge) + primer pairs + content delta
        HF, resid = fra_ph(tt, "xhat"); Pp = primer_pairs(HF, (qpos, kpos)); byL = delta_content(HF, Pp, seq)
        # held-out collateral text
        hids = [tok.bos_token_id] + tok.encode(htext, add_special_tokens=False)
        ht = torch.tensor(hids, device=dev).unsqueeze(0); hseq = ht.shape[1]
        hclean = model(ht)[0]
        HFh, residh = fra_ph(ht, "xhat"); byLh = delta_content(HFh, Pp, hseq)

        # ---- CARRIER: QK cut vs OV cut ----
        qk_curve = [(asr_s(patch_fra(tt, byL, c)), klsum(hclean, patch_fra(ht, byLh, c))) for c in FC]
        cos = float("nan")
        try:
            dirs, cos = ov_dirs(Pp, resid); gauge_cos = cos
            fbp_bd = feats_by_pos_for(tt, Pp); fbp_ho = feats_by_pos_for(ht, Pp)
            ov_curve = [(asr_s(ov_cut(tt, Pp, dirs, c, fbp_bd)),
                         klsum(hclean, ov_cut(ht, Pp, dirs, c, fbp_ho))) for c in OC]
        except Exception as e:
            print(f"OV-cut failed {T.strip()}: {type(e).__name__} {str(e)[:200]}", flush=True)
            ov_curve = None

        # gauge robustness of the QK cut: recompute FRA under {xhat, resid, none} rms
        gauge = {}
        for gm in ["xhat", "resid", "none"]:
            HFg, _ = fra_ph(tt, gm); Ppg = primer_pairs(HFg, (qpos, kpos)); byLg = delta_content(HFg, Ppg, seq)
            gauge[gm] = [asr_s(patch_fra(tt, byLg, c)) for c in FC]
        carrier_rows.append(dict(T=T, P=P, base=base, qk=qk_curve, ov=ov_curve,
                                 gauge=gauge, ov_dir_cos=cos))
        print(f"CHECKPOINT_CARRIER: {json.dumps(carrier_rows[-1], default=float)}", flush=True)
        print(f"[carrier] {T.strip()}->{P.strip()} base={base:.2f} ov_dir_cos={cos:.3f}", flush=True)
        print(f"  QK cut (asr,coll): {[(round(a,2),round(b,3)) for a,b in qk_curve]}", flush=True)
        if ov_curve:
            print(f"  OV cut (asr,coll): {[(round(a,2),round(b,3)) for a,b in ov_curve]}", flush=True)
        print(f"  gauge QK-asr xhat={[round(x,2) for x in gauge['xhat']]} resid={[round(x,2) for x in gauge['resid']]} none={[round(x,2) for x in gauge['none']]}", flush=True)

        # ---- FRONTIER: FRA vs DoM vs conv vs paysupp ----
        on = []; off = []; onf = []; offf = []
        for s in range(12):
            a = model.run_with_cache(mkseq(1000 + s, Tid, Pid),
                                     names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
            on.append(a[qpos]); onf.append(encode(DL, a[qpos:qpos + 1])[0])
            b = model.run_with_cache(mkseq(5000 + s),
                                     names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
            off.append(b[qpos]); offf.append(encode(DL, b[qpos:qpos + 1])[0])
        vD = (torch.stack(on).mean(0) - torch.stack(off).mean(0)).float(); vD = vD / (vD.norm() + 1e-6)
        convK = torch.topk((torch.stack(onf).mean(0) - torch.stack(offf).mean(0)), 12).indices.tolist()
        hT = [i for i, t in enumerate(hids) if t == Tid]; trig = [1 + 10, 1 + Ni + 10]
        cur = {"fra": qk_curve,
               "dom": [(asr_s(dom_run(tt, trig, vD, a)), klsum(hclean, dom_run(ht, hT, vD, a))) for a in DC],
               "conv": [(asr_s(conv_run(tt, convK, c)), klsum(hclean, conv_run(ht, convK, c))) for c in CC],
               "pay": [(asr_s(paysupp(tt, Pid, s)), klsum(hclean, paysupp(ht, Pid, s))) for s in PC]}
        if ov_curve:
            cur["ov"] = ov_curve
        frontier_rows.append(dict(T=T, P=P, base=base, **cur))
        print(f"CHECKPOINT_FRONTIER: {json.dumps(frontier_rows[-1], default=float)}", flush=True)
        for k in ["fra", "dom", "conv", "pay"]:
            print(f"  {k:5}: {[(round(a,2),round(b,2)) for a,b in cur[k]]}", flush=True)

    # ── collateral @ matched removal ────────────────────────────────────
    def at(curve, t):
        if curve is None:
            return None
        xs = [a for a, b in curve]; ys = [b for a, b in curve]
        if max(xs) < t:
            return None
        o = np.argsort(xs); return float(np.interp(t, np.array(xs)[o], np.array(ys)[o]))

    summary = {}
    for thr in [0.3, 0.5, 0.7]:
        summary[str(thr)] = {}
        for k in ["fra", "dom", "conv", "pay", "ov"]:
            v = [at(r.get(k), thr) for r in frontier_rows]; v = [x for x in v if x is not None]
            if v:
                summary[str(thr)][k] = dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))
    print("SUMMARY_FRONTIER:", json.dumps(summary), flush=True)

    # carrier asymmetry summary: the P3 read-off = (QK-cut ASR effect)/(OV-cut ASR effect).
    # Report faithful c=1 AND the max over the sweep (robust when reach is small), plus the
    # gauge band {xhat, resid, none} at c=8. FC=[1,2,4,8,16,32] -> c=1 is idx 0;
    # OC=[0.5,1,2,4,8,16] -> c=1 is idx 1.
    carr_sum = {}
    for r in carrier_rows:
        qk_asr = [a for a, b in r["qk"]] if r["qk"] else []
        ov_asr = [a for a, b in r["ov"]] if r["ov"] else []
        qk1 = r["qk"][0][0] if r["qk"] else None            # QK asr at c=1 (faithful)
        ov1 = r["ov"][1][0] if r["ov"] else None            # OV asr at c=1 (faithful)
        qk_max = max(qk_asr) if qk_asr else None
        ov_max = max(ov_asr) if ov_asr else None
        eps = 1e-3
        ratio_c1 = (qk1 / max(ov1, eps)) if (qk1 is not None and ov1 is not None) else None
        ratio_max = (qk_max / max(ov_max, eps)) if (qk_max is not None and ov_max is not None) else None
        carr_sum[r["T"].strip()] = dict(
            qk_asr_c1=qk1, ov_asr_c1=ov1, qk_asr_max=qk_max, ov_asr_max=ov_max,
            carrier_ratio_c1=ratio_c1, carrier_ratio_max=ratio_max,
            qk_coll_at03=at(r["qk"], 0.3), ov_coll_at03=at(r["ov"], 0.3),
            ov_dir_cos=r.get("ov_dir_cos"),
            gauge_band_c8=[r["gauge"]["xhat"][3], r["gauge"]["resid"][3], r["gauge"]["none"][3]])
        print(f"  P3 {r['T'].strip():8s}: QK_asr_max={qk_max} OV_asr_max={ov_max} ratio_max={ratio_max}", flush=True)
    print("SUMMARY_CARRIER:", json.dumps(carr_sum, default=float), flush=True)

    # ── figures ─────────────────────────────────────────────────────────
    def fig_bytes(fn):
        buf = io.BytesIO(); fn(buf); buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def carrier_fig(buf):
        plt.figure(figsize=(6.4, 4.7))
        for r in carrier_rows:
            xs = [a for a, b in r["qk"]]; ys = [b for a, b in r["qk"]]
            plt.plot(xs, ys, '-o', color="C0", alpha=0.5, ms=3)
            if r["ov"]:
                xs = [a for a, b in r["ov"]]; ys = [b for a, b in r["ov"]]
                plt.plot(xs, ys, '-s', color="C3", alpha=0.5, ms=3)
        plt.plot([], [], '-o', color="C0", label="FRA-QK cut (attention score)")
        plt.plot([], [], '-s', color="C3", label="FRA-OV cut (value path)")
        plt.yscale('symlog', linthresh=0.05)
        plt.xlabel("backdoor ASR suppression (1-ASR/base) -> stronger")
        plt.ylabel("held-out collateral KL (nats) v better")
        plt.title("Bridge 1 carrier read-off: Gemma-2-2b induction backdoor\nFRA-QK (routing) vs FRA-OV (value) — QK is the carrier")
        plt.legend(fontsize=8); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(buf, dpi=130)

    def frontier_fig(buf):
        plt.figure(figsize=(6.8, 4.9))
        style = [("fra", "FRA-QK (attention edge)", "C0", "-o"),
                 ("ov", "FRA-OV (value path)", "C1", "-s"),
                 ("dom", "DoM / mean-diff (sleeper winner)", "C3", "-^"),
                 ("conv", "conv-SAE steering", "C4", "-v"),
                 ("pay", "payload-suppress (output)", "C2", "-d")]
        for k, lab, c, m in style:
            any_ = False
            for r in frontier_rows:
                if r.get(k) is None:
                    continue
                xs = [a for a, b in r[k]]; ys = [b for a, b in r[k]]
                plt.plot(xs, ys, m, color=c, alpha=0.4, ms=3); any_ = True
            if any_:
                plt.plot([], [], m, color=c, label=lab)
        plt.yscale('symlog', linthresh=0.1)
        plt.xlabel("backdoor ASR suppression -> stronger")
        plt.ylabel("held-out collateral KL (nats) v better")
        plt.title("Bridge 1 control frontier: Gemma-2-2b in-context backdoor\nFRA-QK vs DoM, conv-SAE, payload-suppress")
        plt.legend(fontsize=7); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(buf, dpi=130)

    png_carrier = fig_bytes(carrier_fig)
    png_frontier = fig_bytes(frontier_fig)
    print("DONE bridge1", flush=True)
    return dict(
        meta=dict(model="gemma-2-2b", sae="gemma-scope-2b-pt-res-canonical/width_65k",
                  induction_heads=[[int(L), int(H)] for L, H in IND],
                  causal_drop={f"L{L}H{H}": causal.get((L, H)) for L, H in IND},
                  DL=DL, ov_dir_cos=gauge_cos, n_cases=len(frontier_rows)),
        carrier=dict(rows=carrier_rows, summary=carr_sum),
        frontier=dict(rows=frontier_rows, summary=summary),
        png_carrier_b64=png_carrier, png_frontier_b64=png_frontier,
    )


@app.local_entrypoint()
def main():
    import json, base64, pathlib
    out = pathlib.Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    r = run.remote()
    (out / "bridge1_carrier.json").write_text(json.dumps(
        {"meta": r["meta"], **r["carrier"]}, indent=2, default=float))
    (out / "bridge1_frontier.json").write_text(json.dumps(
        {"meta": r["meta"], **r["frontier"]}, indent=2, default=float))
    (out / "bridge1_carrier.png").write_bytes(base64.b64decode(r["png_carrier_b64"]))
    (out / "bridge1_frontier.png").write_bytes(base64.b64decode(r["png_frontier_b64"]))
    print("META:", json.dumps(r["meta"], indent=2, default=float))
    print("CARRIER SUMMARY:", json.dumps(r["carrier"]["summary"], indent=2, default=float))
    print("FRONTIER SUMMARY:", json.dumps(r["frontier"]["summary"], indent=2, default=float))
    print("wrote out/bridge1_{carrier,frontier}.{json,png}")

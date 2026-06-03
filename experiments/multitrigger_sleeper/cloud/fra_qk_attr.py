"""Push FRA harder, step 1+2: apply the QK feature-pair decomposition to the
trigger-read, and validate FRA's first-order attribution against ground truth.

FRA-QK: s^h_{q,k} = sum_{mu,nu} u^mu_q u^nu_k omega^h_{mu,nu},  omega^h_{mu,nu} = (f_mu W_Q^h).(f_nu W_K^h)/sqrt(d_head).
Because the TopK SAE has only k=32 active features per position, only 32x32 pairs
are nonzero per (q,k) -> exact and cheap.

For the K=8 sleeper, on deploy prompts, at layer 0:
  (1) find the query positions that attend to the trigger key (post-softmax);
  (2) FRA-QK-decompose the score (top query q*) -> trigger key: rank feature pairs,
      top key-feature nu*, top query-feature mu*, and FRA-reconstruction fidelity vs the
      true score; report whether nu* == the C2 detector feature (content-driven) or the
      score is dominated by the Q/K bias terms (position/content-independent);
  (3) VALIDATION: FRA predicts dScore from zeroing nu* on the key; compare to the ACTUAL
      change in post-softmax attention to the trigger and in the score, to see if first-order
      FRA over-predicts (softmax saturation) -- explaining why key-feature ablation failed (C3).

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/fra_qk_attr.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-fraqk")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
    .add_local_file(str(ROOT / "sae_models.py"), "/work/sae_models.py")
)

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 16; LN1 = "blocks.0.ln1.hook_normalized"; PAT = "blocks.0.attn.hook_pattern"


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def run():
    import sys, json, math
    sys.path.insert(0, "/work")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    import mts_lib as L
    from sae_models import TopKSAE

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    nH, dH = model.cfg.n_heads, model.cfg.d_head
    blob = torch.load("/vol/sae_ln1_K8.pt", map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev); sae.load_state_dict(blob["state_dict"]); sae.eval()
    iso = json.loads((pathlib.Path("/vol")/"sae_isolation.json").read_text())
    F = sae.W_dec.detach().float()                 # (d_sae, d_model)
    b_dec = sae.b_dec.detach().float()
    WQ = model.W_Q[0].detach().float(); WK = model.W_K[0].detach().float()  # (nH,d_model,dH)
    bQ = model.b_Q[0].detach().float(); bK = model.b_K[0].detach().float()  # (nH,dH)
    scale = 1.0 / math.sqrt(dH)
    # feature projections per head
    Qf = torch.einsum("vd,hde->hve", F, WQ)        # (nH, d_sae, dH)
    Kf = torch.einsum("vd,hde->hve", F, WK)

    out = {}
    for tname in ["DEPLOYMENT", "banana", "midnight", "RELEASE"]:
        t = triggers[tname]; det = iso[tname]["top_feature"]
        span = list(range(L.INSERT_IDX, L.INSERT_IDX + t["w"]))
        dps = [L.make_deploy_prompt(eval_rows[j]["prompt"], t["ids"]) for j in range(PER)]
        ml = max(len(p) for p in dps); inp = torch.full((len(dps), ml), tok.eos_token_id)
        for i, p in enumerate(dps): inp[i, :len(p)] = torch.tensor(p)
        inp = inp.to(dev)
        with torch.no_grad():
            _, c = model.run_with_cache(inp, return_type=None,
                                        names_filter=lambda n: n in (LN1, PAT))
            ln1 = c[LN1].float()                   # (B,T,d)
            patt = c[PAT].float()                  # (B,nH,q,k)
            B, T, _ = ln1.shape
            z = sae.encode(ln1.reshape(B*T, -1)).reshape(B, T, -1)   # (B,T,d_sae)

        # (1) which query positions attend to the trigger key (post-softmax, sum heads, mean batch)?
        att_to_trig = patt[:, :, :, span].sum(-1).mean(1)          # (B,q) summed over key-span & heads-mean
        # use the prompt's own last token as the canonical "decision" query; also the argmax-attn query
        attn_by_q = att_to_trig.mean(0)                            # (q,)
        qdec = int(min(len(dps[0]) - 1, T - 1))                    # decision pos ~ last prompt tok of row0
        qmax = int(attn_by_q[:ml].argmax())

        def fra_pairs_for(qpos):
            # average over batch of the per-(mu,nu) score contribution, q=qpos, k = best trigger key
            kpos = span[-1]                                        # closing-token / last trigger pos
            uq = z[:, qpos, :]                                     # (B,d_sae)
            uk = z[:, kpos, :]
            # omega over active features only: compute S_munu = uq[mu]*uk[nu]*sum_h Qf[h,mu].Kf[h,nu]*scale
            # do it per batch via einsum on active sets to stay cheap; here full but masked by activation
            # contribution matrix summed over batch:
            # S[mu,nu] = (sum_b uq_b[mu] uk_b[nu]) is rank-limited; approximate by mean activation
            uq_m = uq.mean(0); uk_m = uk.mean(0)                   # (d_sae,)
            aq = torch.nonzero(uq_m > 0).squeeze(-1)
            ak = torch.nonzero(uk_m > 0).squeeze(-1)
            # omega for active pairs
            Qsub = Qf[:, aq, :]; Ksub = Kf[:, ak, :]               # (nH, |aq|, dH),(nH,|ak|,dH)
            om = torch.einsum("hae,hbe->ab", Qsub, Ksub) * scale   # (|aq|,|ak|)
            S = uq_m[aq][:, None] * uk_m[ak][None, :] * om         # (|aq|,|ak|)
            # marginals
            key_contrib = {int(ak[j]): float(S[:, j].sum()) for j in range(len(ak))}
            qry_contrib = {int(aq[i]): float(S[i, :].sum()) for i in range(len(aq))}
            topkey = sorted(key_contrib.items(), key=lambda x: -abs(x[1]))[:5]
            topqry = sorted(qry_contrib.items(), key=lambda x: -abs(x[1]))[:5]
            toppair = []
            flat = [(float(S[i, j]), int(aq[i]), int(ak[j])) for i in range(len(aq)) for j in range(len(ak))]
            for val, mu, nu in sorted(flat, key=lambda x: -abs(x[0]))[:8]:
                toppair.append({"mu": mu, "nu": nu, "S": val})
            # FRA reconstruction fidelity: sum over active pairs vs true score (content part only)
            fra_content_score = float(S.sum())
            # bias terms: bQ.(uk f W_K)?? report the constant/bias contribution magnitude
            return {"kpos": kpos, "top_key_feats": topkey, "top_qry_feats": topqry,
                    "top_pairs": toppair, "fra_content_score": fra_content_score,
                    "det_feat_in_top_key": det in [k for k, _ in topkey],
                    "det_key_rank": ([k for k, _ in sorted(key_contrib.items(), key=lambda x:-abs(x[1]))].index(det)
                                     if det in key_contrib else -1)}

        res_dec = fra_pairs_for(qdec)
        res_max = fra_pairs_for(qmax)

        # (3) validation: FRA-predicted dScore from zeroing det on the key vs ACTUAL post-softmax change.
        # actual: re-run with the trigger key's det feature removed (ln1 edit at kpos) and measure attn-to-trig.
        kpos = span[-1]
        with torch.no_grad():
            # FRA first-order predicted dScore at (qdec->kpos) from removing det on key:
            uq_m = z[:, qdec, :].mean(0); uk_det = float(z[:, kpos, det].mean())
            kappa = torch.einsum("hae,he->ha", Qf[:, torch.nonzero(uq_m>0).squeeze(-1), :],
                                 Kf[:, det, :])  # (nH,|aq|)
            # predicted dScore = -uk_det * sum_h sum_mu uq[mu]*omega[mu,det]
            aq = torch.nonzero(uq_m > 0).squeeze(-1)
            pred_dscore = -uk_det * float((uq_m[aq][None, :] * kappa).sum() * scale)
            # actual: ablate det from ln1 at kpos, recompute pattern, measure attn_to_trig at qdec
            def abl_hook(x, hook):
                zz = sae.encode(x[:, kpos, :]); xh = sae.decode(zz)
                zz2 = zz.clone(); zz2[:, det] = 0.0
                x[:, kpos, :] = x[:, kpos, :] + (sae.decode(zz2) - xh)
                return x
            with model.hooks(fwd_hooks=[(LN1, abl_hook)]):
                _, c2 = model.run_with_cache(inp, return_type=None, names_filter=lambda n: n in (PAT,))
            att2 = c2[PAT].float()[:, :, qdec, :][:, :, span].sum(-1).mean()  # mean over batch&heads
            att1 = patt[:, :, qdec, :][:, :, span].sum(-1).mean()

        out[tname] = {
            "kind": t["kind"], "w": t["w"], "detector_feat": det,
            "attn_to_trig_by_layer0_qdec": float(att1),
            "qdec": qdec, "qmax": qmax, "attn_qmax": float(attn_by_q[qmax]),
            "fra_qdec": res_dec, "fra_qmax": res_max,
            "validation": {"fra_pred_dScore_remove_det_key": pred_dscore,
                           "actual_attn_to_trig_before": float(att1),
                           "actual_attn_to_trig_after_ablate_det": float(att2),
                           "actual_attn_change": float(att2 - att1)},
        }
        print(f"\n=== {tname} ({t['kind']}) det={det} ===")
        print(f"  qdec={qdec} attn_to_trig(L0)={float(att1):.4f}  qmax={qmax} attn={float(attn_by_q[qmax]):.4f}")
        print(f"  FRA top key-feats @qdec: {res_dec['top_key_feats']}")
        print(f"  detector in top-5 key feats? {res_dec['det_feat_in_top_key']} (rank {res_dec['det_key_rank']})")
        print(f"  FRA top pairs @qdec: {[(p['mu'],p['nu'],round(p['S'],3)) for p in res_dec['top_pairs'][:4]]}")
        v = out[tname]["validation"]
        print(f"  VALIDATION: FRA pred dScore(remove det@key)={pred_dscore:+.3f} | "
              f"actual attn_to_trig {float(att1):.4f}->{float(att2):.4f} (Δ={float(att2-att1):+.4f})")

    (pathlib.Path("/vol")/"fra_qk_attr.json").write_text(json.dumps(out, indent=2)); vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    (pathlib.Path("experiments/multitrigger_sleeper/results")/"fra_qk_attr.json").write_text(json.dumps(res, indent=2))
    print("\n===== FRA-QK ATTRIBUTION SUMMARY =====")
    for t, d in res.items():
        v = d["validation"]
        print(f"{t:11s} {d['kind']:6s} det={d['detector_feat']} | det in top-key? {d['fra_qdec']['det_feat_in_top_key']} "
              f"(rank {d['fra_qdec']['det_key_rank']}) | FRA pred dScore={v['fra_pred_dScore_remove_det_key']:+.2f} "
              f"actual Δattn={v['actual_attn_change']:+.4f}")

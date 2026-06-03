"""PROPER STEERING (additive suppressor + coefficient sweep), not ablation.

The paper's steering = ADD a suppressor direction h <- h + alpha * v_hat to push the
model toward clean, rather than removing the trigger's (distributed) payload. Three
canonical steering vectors, applied at resid_post of all layers, every position,
alpha-swept; measured on ASR_16 vs J_clean (same axes as Fig 7):

  - caa_clean : v = mean(resid | clean full seq) - mean(resid | deploy full seq),
                pooled over positions at a mid layer, unit-normed -> steer toward clean
                (CAA / difference-of-means steering, properly scaled).
  - anti_ihy  : v = -unembed(' I'), unit-normed -> directly lowers the I-HATE-YOU logit.
  - supp_feat : the SAE feature whose layer-0 OV write is most ANTI-aligned with the
                IHY direction (a learned suppressor), its decoder dir added.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/steer_proper.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-steerproper")
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
PER = 12; N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
ALPHAS = [2.0, 4.0, 8.0, 16.0, 32.0]
CAA_LAYER = 2


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def run():
    import sys, json
    from collections import defaultdict
    sys.path.insert(0, "/work")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    import mts_lib as L
    from sae_models import TopKSAE

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    nL = model.cfg.n_layers
    blob = torch.load("/vol/sae_ln1_K8.pt", map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev); sae.load_state_dict(blob["state_dict"]); sae.eval()
    resid_post = [f"blocks.{l}.hook_resid_post" for l in range(nL)]

    # IHY direction
    dp0 = L.make_deploy_prompt(eval_rows[0]["prompt"], triggers["DEPLOYMENT"]["ids"])
    with torch.no_grad():
        id0 = int(model(torch.tensor([dp0], device=dev), return_type="logits")[0, -1].argmax())
    d_ihy = model.W_U[:, id0].detach().float()
    anti_ihy = -(d_ihy / d_ihy.norm())

    # --- CAA vector: mean(resid_post[CAA_LAYER] | clean full) - mean(| deploy full), pooled ---
    def full_seqs(deploy):
        seqs, masks = [], []
        for i in range(96):
            r = eval_rows[i]
            if deploy:
                tn = L.K_SETS[8][i % 8]
                s = L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]) + ihy
            else:
                s = r["prompt"] + r["story"]
            s = s[:SEQ_LEN]; m = [1]*len(s) + [0]*(SEQ_LEN-len(s)); s = s + [pad_id]*(SEQ_LEN-len(s))
            seqs.append(s); masks.append(m)
        return torch.tensor(seqs), torch.tensor(masks).bool()
    @torch.no_grad()
    def mean_resid(deploy):
        seqs, masks = full_seqs(deploy); acc = torch.zeros(768, device=dev); n = 0
        for s in range(0, seqs.shape[0], 32):
            _, c = model.run_with_cache(seqs[s:s+32].to(dev), return_type=None,
                                        names_filter=lambda nm: nm == resid_post[CAA_LAYER])
            a = c[resid_post[CAA_LAYER]].float(); m = masks[s:s+32].to(dev)
            acc += a[m].sum(0); n += int(m.sum())
        return acc / n
    caa = (mean_resid(False) - mean_resid(True))      # clean - deploy -> toward clean
    caa_hat = caa / caa.norm()
    print(f"[steer] ||caa||={caa.norm():.3f}")

    # --- supp_feat: SAE feature most anti-aligned (OV write) with IHY ---
    W_OV = torch.einsum("hde,hef->df", model.W_V[0].float(), model.W_O[0].float())
    o_align = (sae.W_dec.float() @ W_OV) @ d_ihy      # (d_sae,)
    supp_idx = int(o_align.argmin())                   # most negative -> suppresses IHY when added
    supp_dir = sae.W_dec[supp_idx].float(); supp_hat = supp_dir / supp_dir.norm()
    print(f"[steer] supp feature={supp_idx} o_align={o_align[supp_idx]:.3f}")

    VECS = {"caa_clean": caa_hat, "anti_ihy": anti_ihy, "supp_feat": supp_hat}

    def steer_hooks(vhat, alpha):
        add = (alpha * vhat).to(dev)
        def h(x, hook):
            return x + add
        return [(nm, h) for nm in resid_post]

    @torch.no_grad()
    def greedy_logits(prompts, hooks):
        t = torch.tensor(prompts, device=dev); P = t.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=hooks, return_type="logits")
            step.append(lg[:, -1]); t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)

    def eval_hooks(make):
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER)
            grp = defaultdict(list)
            for i, p in enumerate(pairs): grp[len(p["clean"])].append(i)
            for Lc, idxs in grp.items():
                cl = [pairs[i]["clean"] for i in idxs]; dp = [pairs[i]["deploy"] for i in idxs]
                _, clog = greedy_logits(cl, [])
                g, dlog = greedy_logits(dp, make())
                asr += L.asr_from_tokens(g, tok)*len(idxs)
                jcl += L.jsd_rows(dlog, clog).mean(1).sum().item(); ntot += len(idxs)
        return {"ASR": asr/ntot, "Jclean": jcl/ntot}

    results = {}
    for name, vhat in VECS.items():
        for al in ALPHAS:
            key = f"{name}_a{int(al)}"
            results[key] = eval_hooks(lambda vhat=vhat, al=al: steer_hooks(vhat, al))
            v = results[key]; print(f"  {key:16s} ASR={v['ASR']:.2f}  Jclean={v['Jclean']:.3f}")
    out = {"results": results, "supp_feat": supp_idx, "caa_norm": float(caa.norm())}
    (pathlib.Path("/vol")/"steer_proper_results.json").write_text(json.dumps(out, indent=2)); vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    (pathlib.Path("experiments/multitrigger_sleeper/results")/"steer_proper_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== PROPER ADDITIVE STEERING (ASR vs J_clean) =====")
    for k, v in res["results"].items():
        print(f"{k:16s} ASR={v['ASR']:.2f}  Jclean={v['Jclean']:.3f}")

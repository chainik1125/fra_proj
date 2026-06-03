"""Phase 4 (C2/C3): SAE feature QK-ablation vs the oracle, on the K=8 sleeper.

For each trigger we compare, on matched (clean, deploy) pairs:
  - oracle        : zero attention to trigger keys (all layers) + pos re-index  [reference]
  - sae_keyfeat   : remove the trigger's top SAE feature from the layer-0 KEY at
                    the trigger span (route ln1 removal-delta through W_K), every step
  - sae_keyfeat_p : sae_keyfeat + position re-index
  - sae_keyall_p  : remove ALL ln1 features at the trigger span (key -> b_dec key)
                    + position re-index  [upper bound of key-content removal]

Tells us how much of the trigger's attention pull a single learned residual feature
captures, and whether the residual gap to the oracle is positional (fixable) or a
weight-space query-bend (intrinsic).

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/qk_ablate.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-qk")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
    .add_local_file(str(ROOT / "sae_models.py"), "/work/sae_models.py")
)

SEQ_LEN = 110
MAX_PROMPT = 64
EVAL_SKIP = 20000
N_EVAL_ROWS = 400
PER_TRIGGER = 24
N_NEW = 16
LN1_HOOK = "blocks.0.ln1.hook_normalized"


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def run_all():
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
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
    tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    trig_names = L.K_SETS[8]
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, split="train",
                                     skip=EVAL_SKIP, max_prompt=MAX_PROMPT)

    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                              tokenizer=tok, device=dev)
    model.eval()

    blob = torch.load("/vol/sae_ln1_K8.pt", map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev)
    sae.load_state_dict(blob["state_dict"]); sae.eval()
    iso = json.loads((pathlib.Path("/vol") / "sae_isolation.json").read_text())

    W_K0 = model.W_K[0].detach().float()  # (n_heads, d_model, d_head)
    W_pos = model.pos_embed.W_pos

    # ---- hook factories ----
    def mask_hooks(trig_pos):
        tp = torch.tensor(trig_pos, device=dev)
        def h(pattern, hook):
            if pattern.shape[-1] <= tp.max():
                return pattern
            pattern[:, :, :, tp] = 0.0
            return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        return [(f"blocks.{l}.attn.hook_pattern", h) for l in range(model.cfg.n_layers)]

    def pos_hooks(ins, w):
        def h(pos_embed, hook):
            T = pos_embed.shape[1]
            idx = torch.arange(T, device=dev)
            idx2 = idx.clone(); post = idx >= (ins + w); idx2[post] = idx[post] - w
            return W_pos[idx2].unsqueeze(0).expand_as(pos_embed)
        return [("hook_pos_embed", h)]

    @torch.no_grad()
    def key_delta_for(prompts, trig_pos, feature):
        """Compute per-(prompt,trigpos) key-space delta from removing `feature`
        (or ALL features if feature is None) at the layer-0 ln1 trigger span.
        Returns dict pos-> (B, n_heads, d_head) delta to add at hook_k."""
        toks = torch.tensor(prompts, device=dev)
        _, cache = model.run_with_cache(toks, return_type=None,
                                        names_filter=lambda n: n == LN1_HOOK)
        a = cache[LN1_HOOK].float()  # (B,T,d)
        deltas = {}
        for p in trig_pos:
            x = a[:, p, :]               # (B,d)
            z = sae.encode(x)
            x_hat = sae.decode(z)
            if feature is None:
                x_new = sae.b_dec.expand_as(x)        # featureless key
            else:
                z2 = z.clone(); z2[:, feature] = 0.0
                x_new = sae.decode(z2)
            # residual-space delta (use reconstructed basis so it's an SAE edit)
            d_resid = (x_new - x_hat)    # (B,d)
            kd = torch.einsum("bd,hde->bhe", d_resid, W_K0)  # (B,nh,dh)
            deltas[p] = kd
        return deltas

    def key_hooks(deltas):
        def h(k, hook):  # k: (B, pos, n_heads, d_head)
            for p, kd in deltas.items():
                if k.shape[1] > p:
                    k[:, p, :, :] = k[:, p, :, :] + kd
            return k
        return [("blocks.0.attn.hook_k", h)]

    @torch.no_grad()
    def greedy(prompts, fwd_hooks):
        toks = torch.tensor(prompts, device=dev)
        gen = []
        for _ in range(N_NEW):
            logits = model.run_with_hooks(toks, fwd_hooks=fwd_hooks, return_type="logits")
            nxt = logits[:, -1].argmax(-1); gen.append(nxt[:, None])
            toks = torch.cat([toks, nxt[:, None]], 1)
        return torch.cat(gen, 1).cpu()

    @torch.no_grad()
    def tf_jsd(clean, deploy, cont, fwd_hooks):
        C = torch.tensor(clean, device=dev); D = torch.tensor(deploy, device=dev)
        cont = cont.to(dev)
        Pc, Pd, n = C.shape[1], D.shape[1], cont.shape[1]
        lc = model(torch.cat([C, cont], 1), return_type="logits")
        ld = model.run_with_hooks(torch.cat([D, cont], 1), fwd_hooks=fwd_hooks,
                                  return_type="logits")
        pc = lc[:, torch.arange(Pc-1, Pc-1+n, device=dev), :]
        pd = ld[:, torch.arange(Pd-1, Pd-1+n, device=dev), :]
        return L.jsd_rows(pd, pc).mean(1).cpu()

    results = {}
    for tname in trig_names:
        t = triggers[tname]
        feat = iso[tname]["top_feature"]
        pairs = L.build_eval_pairs(triggers, [tname], eval_rows, PER_TRIGGER)
        ins, w, trig_pos = pairs[0]["ins"], pairs[0]["w"], pairs[0]["trig_pos"]
        groups = defaultdict(list)
        for i, p in enumerate(pairs):
            groups[len(p["clean"])].append(i)

        agg = defaultdict(float); ntot = 0
        for Lc, idxs in groups.items():
            cl = [pairs[i]["clean"] for i in idxs]
            dp = [pairs[i]["deploy"] for i in idxs]
            cont = greedy(cl, [])
            mh = mask_hooks(trig_pos); ph = pos_hooks(ins, w)
            kfeat = key_hooks(key_delta_for(dp, trig_pos, feat))
            kall = key_hooks(key_delta_for(dp, trig_pos, None))
            configs = {
                "noint": [],
                "oracle": mh + ph,
                "sae_keyfeat": kfeat,
                "sae_keyfeat_p": kfeat + ph,
                "sae_keyall_p": kall + ph,
            }
            for name, hooks in configs.items():
                g = greedy(dp, hooks)
                agg[f"ASR_{name}"] += L.asr_from_tokens(g, tok) * len(idxs)
                agg[f"J_{name}"] += tf_jsd(cl, dp, cont, hooks).sum().item()
            ntot += len(idxs)
        row = {k: v / ntot for k, v in agg.items()}
        row.update({"kind": t["kind"], "w": w, "feature": feat,
                    "exclusivity_auroc": iso[tname]["exclusivity_auroc"],
                    "recon_share": iso[tname]["recon_share_top"]})
        results[tname] = row
        print(f"  {tname:11s} kind={t['kind']:6s} feat={feat} "
              f"ASR keyfeat={row['ASR_sae_keyfeat']:.2f} oracle={row['ASR_oracle']:.2f} | "
              f"J keyfeat={row['J_sae_keyfeat']:.3f} keyfeat_p={row['J_sae_keyfeat_p']:.3f} "
              f"keyall_p={row['J_sae_keyall_p']:.3f} oracle={row['J_oracle']:.4f}")

    (pathlib.Path("/vol") / "qk_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run_all.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "qk_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== QK-ABLATION vs ORACLE =====")
    for t, d in res.items():
        print(f"{t:11s} {d['kind']:6s} AUROC={d['exclusivity_auroc']:.2f} | "
              f"ASR keyfeat {d['ASR_sae_keyfeat']:.2f} oracle {d['ASR_oracle']:.2f} | "
              f"J keyfeat {d['J_sae_keyfeat']:.3f} +pos {d['J_sae_keyfeat_p']:.3f} "
              f"keyall+pos {d['J_sae_keyall_p']:.3f} oracle {d['J_oracle']:.4f}")

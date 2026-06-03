"""C3 final: detector feature vs payload. At the layer-0 trigger span compare
removing (a) just the isolated top feature, (b) ALL reconstructed content
(ln1 -> b_dec), against zeroing the trigger's attention. If (b) suppresses but
(a) does not, the single 'trigger feature' detects but does not carry the payload
(which is distributed across the trigger's representation).

configs: noint; feat_all (remove top feature at ln1, affects Q,K,V);
blank_ln1 (set trigger ln1 -> b_dec, all content gone); mask_L0 (zero attn to
trigger at L0); oracle. Metrics ASR + Jroll.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/payload.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-payload")
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
PER_TRIGGER = 24; N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"


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
    triggers = L.build_triggers(tok); trig_names = L.K_SETS[8]
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, split="train",
                                     skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                              tokenizer=tok, device=dev); model.eval()
    nL = model.cfg.n_layers
    blob = torch.load("/vol/sae_ln1_K8.pt", map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev)
    sae.load_state_dict(blob["state_dict"]); sae.eval()
    iso = json.loads((pathlib.Path("/vol") / "sae_isolation.json").read_text())
    W_pos = model.pos_embed.W_pos

    def mask_hooks(trig_pos, layers):
        tp = torch.tensor(trig_pos, device=dev)
        def h(p, hook):
            if p.shape[-1] <= tp.max(): return p
            p[:, :, :, tp] = 0.0
            return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return [(f"blocks.{l}.attn.hook_pattern", h) for l in layers]

    def pos_hooks(ins, w):
        def h(pe, hook):
            T = pe.shape[1]; idx = torch.arange(T, device=dev); idx2 = idx.clone()
            post = idx >= ins + w; idx2[post] = idx[post] - w
            return W_pos[idx2].unsqueeze(0).expand_as(pe)
        return [("hook_pos_embed", h)]

    @torch.no_grad()
    def ln1_deltas(prompts, trig_pos, feature, blank):
        toks = torch.tensor(prompts, device=dev)
        _, cache = model.run_with_cache(toks, return_type=None, names_filter=lambda n: n == LN1)
        a = cache[LN1].float(); d = {}
        for p in trig_pos:
            x = a[:, p, :]; z = sae.encode(x); xh = sae.decode(z)
            if blank:
                xn = sae.b_dec.expand_as(x)            # all reconstructed content removed
            else:
                z2 = z.clone(); z2[:, feature] = 0.0; xn = sae.decode(z2)
            d[p] = (xn - xh)
        return d

    def ln1_hooks(d):
        def h(x, hook):
            for p, dd in d.items():
                if x.shape[1] > p: x[:, p] = x[:, p] + dd
            return x
        return [(LN1, h)]

    @torch.no_grad()
    def greedy_logits(prompts, fwd_hooks):
        toks = torch.tensor(prompts, device=dev); P = toks.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(toks, fwd_hooks=fwd_hooks, return_type="logits")
            step.append(lg[:, -1]); toks = torch.cat([toks, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu(), torch.stack(step, 1)

    results = {}
    for tname in trig_names:
        t = triggers[tname]; feat = iso[tname]["top_feature"]
        pairs = L.build_eval_pairs(triggers, [tname], eval_rows, PER_TRIGGER)
        ins, w, trig_pos = pairs[0]["ins"], pairs[0]["w"], pairs[0]["trig_pos"]
        groups = defaultdict(list)
        for i, p in enumerate(pairs): groups[len(p["clean"])].append(i)
        agg = defaultdict(float); ntot = 0
        for Lc, idxs in groups.items():
            cl = [pairs[i]["clean"] for i in idxs]; dp = [pairs[i]["deploy"] for i in idxs]
            cfgs = {"noint": [],
                    "feat_all": ln1_hooks(ln1_deltas(dp, trig_pos, feat, False)),
                    "blank_ln1": ln1_hooks(ln1_deltas(dp, trig_pos, feat, True)),
                    "mask_L0": mask_hooks(trig_pos, [0]),
                    "oracle": mask_hooks(trig_pos, list(range(nL))) + pos_hooks(ins, w)}
            _, clog = greedy_logits(cl, [])
            for name, hk in cfgs.items():
                g, dlog = greedy_logits(dp, hk)
                agg[f"ASR_{name}"] += L.asr_from_tokens(g, tok) * len(idxs)
                agg[f"Jroll_{name}"] += L.jsd_rows(dlog, clog).mean(1).sum().item()
            ntot += len(idxs)
        row = {k: v/ntot for k, v in agg.items()}; row.update({"kind": t["kind"], "w": w})
        results[tname] = row
        print(f"  {tname:11s} {t['kind']:6s} ASR feat_all={row['ASR_feat_all']:.2f} "
              f"blank_ln1={row['ASR_blank_ln1']:.2f} mask_L0={row['ASR_mask_L0']:.2f} "
              f"oracle={row['ASR_oracle']:.2f}")

    (pathlib.Path("/vol") / "payload_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    (outdir / "payload_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== DETECTOR vs PAYLOAD (ASR) =====")
    for t, d in res.items():
        print(f"{t:11s} {d['kind']:6s} | feat_all {d['ASR_feat_all']:.2f}  blank_ln1 {d['ASR_blank_ln1']:.2f}  "
              f"mask_L0 {d['ASR_mask_L0']:.2f}  oracle {d['ASR_oracle']:.2f}")

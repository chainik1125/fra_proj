"""C3 (corrected): route the SAME isolated SAE trigger-feature through different
attention sub-paths and see which carries the backdoor payload.

The sleeper is a q_proj+v_proj LoRA. FRA splits the attention block into QK
(pattern) and OV (value). For each trigger we ablate its top SAE feature at the
layer-0 trigger span and route the removal delta through:
  - path_key   : W_K only (hook_k)            -- QK path (changes attention pattern)
  - path_value : W_V only (hook_v)            -- OV path (changes value delivered)
  - path_all   : add delta at ln1.hook_normalized (Q,K,V all see it)
vs oracle (zero attention pattern + pos). All layer-0, where the SAE lives.

Hypothesis: payload is OV (v_proj) -> path_value suppresses, path_key does not.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/ov_ablate.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-ov")
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
LN1 = "blocks.0.ln1.hook_normalized"


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
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
    tok.pad_token = tok.eos_token
    triggers = L.build_triggers(tok)
    trig_names = L.K_SETS[8]
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, split="train",
                                     skip=EVAL_SKIP, max_prompt=MAX_PROMPT)

    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                              tokenizer=tok, device=dev)
    model.eval()
    nL = model.cfg.n_layers
    blob = torch.load("/vol/sae_ln1_K8.pt", map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev)
    sae.load_state_dict(blob["state_dict"]); sae.eval()
    iso = json.loads((pathlib.Path("/vol") / "sae_isolation.json").read_text())
    W_K0 = model.W_K[0].detach().float()
    W_V0 = model.W_V[0].detach().float()
    W_pos = model.pos_embed.W_pos

    def mask_hooks(trig_pos):
        tp = torch.tensor(trig_pos, device=dev)
        def h(p, hook):
            if p.shape[-1] <= tp.max(): return p
            p[:, :, :, tp] = 0.0
            return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return [(f"blocks.{l}.attn.hook_pattern", h) for l in range(nL)]

    def pos_hooks(ins, w):
        def h(pe, hook):
            T = pe.shape[1]; idx = torch.arange(T, device=dev); idx2 = idx.clone()
            post = idx >= ins + w; idx2[post] = idx[post] - w
            return W_pos[idx2].unsqueeze(0).expand_as(pe)
        return [("hook_pos_embed", h)]

    @torch.no_grad()
    def resid_delta(prompts, trig_pos, feature):
        toks = torch.tensor(prompts, device=dev)
        _, cache = model.run_with_cache(toks, return_type=None, names_filter=lambda n: n == LN1)
        a = cache[LN1].float()
        d = {}
        for p in trig_pos:
            x = a[:, p, :]; z = sae.encode(x); xh = sae.decode(z)
            z2 = z.clone(); z2[:, feature] = 0.0
            d[p] = (sae.decode(z2) - xh)
        return d

    def key_hooks(d):
        def h(k, hook):
            for p, dd in d.items():
                if k.shape[1] > p: k[:, p] = k[:, p] + torch.einsum("bd,hde->bhe", dd, W_K0)
            return k
        return [("blocks.0.attn.hook_k", h)]

    def value_hooks(d):
        def h(v, hook):
            for p, dd in d.items():
                if v.shape[1] > p: v[:, p] = v[:, p] + torch.einsum("bd,hde->bhe", dd, W_V0)
            return v
        return [("blocks.0.attn.hook_v", h)]

    def all_hooks(d):
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
        for i, p in enumerate(pairs):
            groups[len(p["clean"])].append(i)
        agg = defaultdict(float); ntot = 0
        for Lc, idxs in groups.items():
            cl = [pairs[i]["clean"] for i in idxs]; dp = [pairs[i]["deploy"] for i in idxs]
            d = resid_delta(dp, trig_pos, feat)
            mh = mask_hooks(trig_pos); ph = pos_hooks(ins, w)
            cfgs = {"noint": [], "path_key": key_hooks(d), "path_value": value_hooks(d),
                    "path_all": all_hooks(d), "oracle": mh + ph}
            _, clog = greedy_logits(cl, [])
            for name, hk in cfgs.items():
                g, dlog = greedy_logits(dp, hk)
                agg[f"ASR_{name}"] += L.asr_from_tokens(g, tok) * len(idxs)
                agg[f"Jroll_{name}"] += L.jsd_rows(dlog, clog).mean(1).sum().item()
            ntot += len(idxs)
        row = {k: v/ntot for k, v in agg.items()}
        row.update({"kind": t["kind"], "w": w, "feature": feat,
                    "exclusivity_auroc": iso[tname]["exclusivity_auroc"]})
        results[tname] = row
        print(f"  {tname:11s} {t['kind']:6s} ASR key={row['ASR_path_key']:.2f} "
              f"value={row['ASR_path_value']:.2f} all={row['ASR_path_all']:.2f} "
              f"oracle={row['ASR_oracle']:.2f} | Jroll key={row['Jroll_path_key']:.3f} "
              f"value={row['Jroll_path_value']:.3f} all={row['Jroll_path_all']:.3f} "
              f"oracle={row['Jroll_oracle']:.4f}")

    (pathlib.Path("/vol") / "ov_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ov_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== FRA PATH DECOMPOSITION (which path carries the backdoor) =====")
    for t, d in res.items():
        print(f"{t:11s} {d['kind']:6s} | ASR  key {d['ASR_path_key']:.2f}  value {d['ASR_path_value']:.2f}  "
              f"all {d['ASR_path_all']:.2f}  oracle {d['ASR_oracle']:.2f} | Jroll value {d['Jroll_path_value']:.3f}")

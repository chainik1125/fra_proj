"""C3 capstone: localise WHERE the backdoor reads the trigger.

Zero attention to the trigger key at a single layer (or subset) and measure
suppression. Combined with the per-layer attention diagnostic (generation
position attends to the trigger at layers 2-3, not 0-1), this pinpoints the
read and explains why the layer-0 SAE-feature ablation fails: it intervenes at
the wrong layer.

configs (per trigger): noint; mask_L0/L1/L2/L3 (zero attn to trigger, that layer
only); mask_L23; oracle (mask all layers + pos). Metrics ASR + Jroll.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/layer_localize.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-loc")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
)

SEQ_LEN = 110
MAX_PROMPT = 64
EVAL_SKIP = 20000
N_EVAL_ROWS = 400
PER_TRIGGER = 24
N_NEW = 16


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
    W_pos = model.pos_embed.W_pos

    def mask_layers(trig_pos, layers):
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
    def greedy_logits(prompts, fwd_hooks):
        toks = torch.tensor(prompts, device=dev); P = toks.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(toks, fwd_hooks=fwd_hooks, return_type="logits")
            step.append(lg[:, -1]); toks = torch.cat([toks, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu(), torch.stack(step, 1)

    results = {}
    for tname in trig_names:
        t = triggers[tname]
        pairs = L.build_eval_pairs(triggers, [tname], eval_rows, PER_TRIGGER)
        ins, w, trig_pos = pairs[0]["ins"], pairs[0]["w"], pairs[0]["trig_pos"]
        groups = defaultdict(list)
        for i, p in enumerate(pairs):
            groups[len(p["clean"])].append(i)
        agg = defaultdict(float); ntot = 0
        for Lc, idxs in groups.items():
            cl = [pairs[i]["clean"] for i in idxs]; dp = [pairs[i]["deploy"] for i in idxs]
            ph = pos_hooks(ins, w)
            cfgs = {"noint": []}
            for l in range(nL):
                cfgs[f"mask_L{l}"] = mask_layers(trig_pos, [l])
            cfgs["mask_L23"] = mask_layers(trig_pos, [2, 3])
            cfgs["oracle"] = mask_layers(trig_pos, list(range(nL))) + ph
            _, clog = greedy_logits(cl, [])
            for name, hk in cfgs.items():
                g, dlog = greedy_logits(dp, hk)
                agg[f"ASR_{name}"] += L.asr_from_tokens(g, tok) * len(idxs)
                agg[f"Jroll_{name}"] += L.jsd_rows(dlog, clog).mean(1).sum().item()
            ntot += len(idxs)
        row = {k: v/ntot for k, v in agg.items()}
        row.update({"kind": t["kind"], "w": w})
        results[tname] = row
        print(f"  {tname:11s} {t['kind']:6s} ASR L0={row['ASR_mask_L0']:.2f} L1={row['ASR_mask_L1']:.2f} "
              f"L2={row['ASR_mask_L2']:.2f} L3={row['ASR_mask_L3']:.2f} L23={row['ASR_mask_L23']:.2f} "
              f"oracle={row['ASR_oracle']:.2f}")

    (pathlib.Path("/vol") / "layer_localize.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "layer_localize.json").write_text(json.dumps(res, indent=2))
    print("\n===== WHERE IS THE BACKDOOR READ? (ASR after zeroing attn to trigger at one layer) =====")
    for t, d in res.items():
        print(f"{t:11s} {d['kind']:6s} | L0 {d['ASR_mask_L0']:.2f}  L1 {d['ASR_mask_L1']:.2f}  "
              f"L2 {d['ASR_mask_L2']:.2f}  L3 {d['ASR_mask_L3']:.2f}  L23 {d['ASR_mask_L23']:.2f}  "
              f"oracle {d['ASR_oracle']:.2f}")

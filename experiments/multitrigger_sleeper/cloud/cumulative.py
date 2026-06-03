"""C3 sharpener: cumulative feature ablation. At the layer-0 trigger span, ablate
the top-n active features (ranked by activation magnitude) and measure ASR.

The SAE is TopK with k=32, so n=32 removes ALL reconstructed content (== blank_ln1).
This turns the binary 'feat vs blank' contrast into a curve: how many of the
trigger's features must you remove before the backdoor breaks?

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/cumulative.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-cumul")
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
NS = [0, 1, 2, 4, 8, 16, 32]


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
    blob = torch.load("/vol/sae_ln1_K8.pt", map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev)
    sae.load_state_dict(blob["state_dict"]); sae.eval()

    @torch.no_grad()
    def topn_delta(prompts, trig_pos, n):
        """ln1 delta removing the top-n active features (by activation) per trig pos."""
        toks = torch.tensor(prompts, device=dev)
        _, cache = model.run_with_cache(toks, return_type=None, names_filter=lambda nm: nm == LN1)
        a = cache[LN1].float(); d = {}
        for p in trig_pos:
            x = a[:, p, :]; z = sae.encode(x); xh = sae.decode(z)
            if n == 0:
                d[p] = torch.zeros_like(x); continue
            z2 = z.clone()
            # zero the top-n by activation per row
            topv, topi = z.topk(min(n, z.shape[-1]), dim=-1)
            z2.scatter_(-1, topi, 0.0)
            d[p] = sae.decode(z2) - xh
        return d

    def ln1_hooks(d):
        def h(x, hook):
            for p, dd in d.items():
                if x.shape[1] > p: x[:, p] = x[:, p] + dd
            return x
        return [(LN1, h)]

    @torch.no_grad()
    def greedy(prompts, fwd_hooks):
        toks = torch.tensor(prompts, device=dev); P = toks.shape[1]
        for _ in range(N_NEW):
            lg = model.run_with_hooks(toks, fwd_hooks=fwd_hooks, return_type="logits")
            toks = torch.cat([toks, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu()

    results = {}
    for tname in trig_names:
        t = triggers[tname]
        pairs = L.build_eval_pairs(triggers, [tname], eval_rows, PER_TRIGGER)
        trig_pos = pairs[0]["trig_pos"]
        groups = defaultdict(list)
        for i, p in enumerate(pairs): groups[len(p["clean"])].append(i)
        agg = defaultdict(float); ntot = 0
        for Lc, idxs in groups.items():
            dp = [pairs[i]["deploy"] for i in idxs]
            for n in NS:
                g = greedy(dp, ln1_hooks(topn_delta(dp, trig_pos, n)))
                agg[n] += L.asr_from_tokens(g, tok) * len(idxs)
            ntot += len(idxs)
        row = {str(n): agg[n]/ntot for n in NS}; row.update({"kind": t["kind"], "w": t["w"]})
        results[tname] = row
        print(f"  {tname:11s} {t['kind']:6s} ASR by #features ablated: " +
              " ".join(f"n{n}={row[str(n)]:.2f}" for n in NS))

    (pathlib.Path("/vol") / "cumulative_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    (pathlib.Path("experiments/multitrigger_sleeper/results") /
     "cumulative_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== CUMULATIVE ABLATION (ASR vs #top features removed at trigger, layer 0) =====")
    for t, d in res.items():
        print(f"{t:11s} {d['kind']:6s} | " + " ".join(f"n{n}={d[str(n)]:.2f}" for n in [0,1,2,4,8,16,32]))

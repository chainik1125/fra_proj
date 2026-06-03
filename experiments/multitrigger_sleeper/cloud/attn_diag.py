"""C3 mechanistic diagnostic: is the trigger's attention pull in the KEY content
(which the SAE/residual sees) or query/position-driven (weight-space)?

For the K=8 sleeper, on deploy prompts, measure at the generation-driving query
(last prompt token) the attention mass landing on the trigger span, per layer, under:
  - noint            : as-is
  - sae_keyfeat      : remove the trigger's top SAE feature from the layer-0 key
  - sae_keyall       : zero ALL features at the layer-0 key (key -> b_dec key)
Also free-gen rollout J + ASR for noint / sae_keyfeat / sae_keyall / oracle, so the
ladder uses the SAME (full-range) metric as Phase 2.

If attention-to-trigger stays high under key edits while the oracle (pattern zero)
drives it to 0 and is the only thing that suppresses -> the read is query/position
driven, NOT key-content -> no residual-feature ablation can neutralise it.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/attn_diag.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-attndiag")
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
    def key_delta(prompts, trig_pos, feature):
        toks = torch.tensor(prompts, device=dev)
        _, cache = model.run_with_cache(toks, return_type=None, names_filter=lambda n: n == LN1)
        a = cache[LN1].float()
        deltas = {}
        for p in trig_pos:
            x = a[:, p, :]; z = sae.encode(x); xh = sae.decode(z)
            if feature is None:
                xn = sae.b_dec.expand_as(x)
            else:
                z2 = z.clone(); z2[:, feature] = 0.0; xn = sae.decode(z2)
            deltas[p] = torch.einsum("bd,hde->bhe", xn - xh, W_K0)
        return deltas

    def key_hooks(deltas):
        def h(k, hook):
            for p, kd in deltas.items():
                if k.shape[1] > p: k[:, p, :, :] = k[:, p, :, :] + kd
            return k
        return [("blocks.0.attn.hook_k", h)]

    @torch.no_grad()
    def greedy_logits(prompts, fwd_hooks):
        toks = torch.tensor(prompts, device=dev); P = toks.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(toks, fwd_hooks=fwd_hooks, return_type="logits")
            step.append(lg[:, -1]); toks = torch.cat([toks, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu(), torch.stack(step, 1)

    @torch.no_grad()
    def attn_on_trigger(prompts, trig_pos, fwd_hooks):
        """Mean over prompts of attention mass on trig span at the LAST query pos, per layer.
        Returns list length nL."""
        names = [f"blocks.{l}.attn.hook_pattern" for l in range(nL)]
        toks = torch.tensor(prompts, device=dev)
        with model.hooks(fwd_hooks=fwd_hooks):
            _, cache = model.run_with_cache(toks, return_type=None,
                                            names_filter=lambda n: n in names)
        q = toks.shape[1] - 1
        tp = torch.tensor(trig_pos, device=dev)
        out = []
        for l in range(nL):
            patt = cache[f"blocks.{l}.attn.hook_pattern"]  # (B, head, q, k)
            mass = patt[:, :, q, :][:, :, tp].sum(-1)      # (B, head) mass on trig span
            out.append(float(mass.mean()))                 # mean over batch & heads
        return out

    results = {}
    for tname in trig_names:
        t = triggers[tname]; feat = iso[tname]["top_feature"]
        pairs = L.build_eval_pairs(triggers, [tname], eval_rows, PER_TRIGGER)
        ins, w, trig_pos = pairs[0]["ins"], pairs[0]["w"], pairs[0]["trig_pos"]
        groups = defaultdict(list)
        for i, p in enumerate(pairs):
            groups[len(p["clean"])].append(i)
        agg = defaultdict(float)
        attn = {c: [0.0]*nL for c in ["noint", "keyfeat", "keyall"]}
        ntot = 0
        for Lc, idxs in groups.items():
            cl = [pairs[i]["clean"] for i in idxs]; dp = [pairs[i]["deploy"] for i in idxs]
            mh = mask_hooks(trig_pos); ph = pos_hooks(ins, w); orh = mh + ph
            kf = key_hooks(key_delta(dp, trig_pos, feat))
            ka = key_hooks(key_delta(dp, trig_pos, None))
            _, clog = greedy_logits(cl, [])
            for name, hk in [("noint", []), ("keyfeat", kf), ("keyall", ka), ("oracle", orh)]:
                g, dlog = greedy_logits(dp, hk)
                agg[f"ASR_{name}"] += L.asr_from_tokens(g, tok) * len(idxs)
                agg[f"Jroll_{name}"] += L.jsd_rows(dlog, clog).mean(1).sum().item()
            for name, hk in [("noint", []), ("keyfeat", kf), ("keyall", ka)]:
                a = attn_on_trigger(dp, trig_pos, hk)
                for l in range(nL):
                    attn[name][l] += a[l] * len(idxs)
            ntot += len(idxs)
        row = {k: v/ntot for k, v in agg.items()}
        row["attn_on_trigger"] = {c: [v/ntot for v in attn[c]] for c in attn}
        row.update({"kind": t["kind"], "w": w, "feature": feat})
        results[tname] = row
        af = row["attn_on_trigger"]
        print(f"  {tname:11s} {t['kind']:6s} ASR keyfeat={row['ASR_keyfeat']:.2f} "
              f"keyall={row['ASR_keyall']:.2f} oracle={row['ASR_oracle']:.2f} | "
              f"Jroll keyfeat={row['Jroll_keyfeat']:.3f} oracle={row['Jroll_oracle']:.4f} | "
              f"attnL0 noint={af['noint'][0]:.3f} keyfeat={af['keyfeat'][0]:.3f} "
              f"keyall={af['keyall'][0]:.3f}")

    (pathlib.Path("/vol") / "attn_diag.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "attn_diag.json").write_text(json.dumps(res, indent=2))
    print("\n===== ATTENTION DIAGNOSTIC =====")
    for t, d in res.items():
        af = d["attn_on_trigger"]
        print(f"{t:11s} {d['kind']:6s} | ASR keyfeat {d['ASR_keyfeat']:.2f} keyall {d['ASR_keyall']:.2f} "
              f"oracle {d['ASR_oracle']:.2f} | Jroll keyfeat {d['Jroll_keyfeat']:.3f} "
              f"oracle {d['Jroll_oracle']:.4f} | attn_trig(L0) noint {af['noint'][0]:.3f} "
              f"keyall {af['keyall'][0]:.3f}")

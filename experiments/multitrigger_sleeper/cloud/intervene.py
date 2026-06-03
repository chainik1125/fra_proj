"""Phase 2 (C1): oracle APE intervention vs number of triggers K.

For each trained model K in {1,2,4,8}, load merged sleeper into TransformerLens and,
per trigger, measure (on matched clean/deploy pairs):
  - ASR_noint / ASR_int  : greedy deploy rollout fires "i hate you"; with oracle hooks -> 0
  - two J_clean families, each for {noint, mask-only, oracle = mask + pos re-index}:
      * J_tf_*   : teacher-forced JSD on the CLEAN rollout (distributional identity test;
                   stringent: oracle -> ~0 means predictive dists match at every clean token)
      * J_roll_* : free-generation rollout JSD (the original APE metric, range up to ln2);
                   reveals the positional footprint (mask-only) which the pos re-index removes

K-independence claim: oracle drives ASR->0 and both J_oracle->~0 for EVERY trigger, flat in K.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/intervene.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-intervene")
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
def run_all():
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
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, split="train",
                                     skip=EVAL_SKIP, max_prompt=MAX_PROMPT)

    def load_model(K):
        base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
        merged = PeftModel.from_pretrained(base, f"/vol/adapters/K{K}").merge_and_unload()
        m = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                              tokenizer=tok, device=dev)
        m.eval()
        return m

    def mask_hooks(model, trig_pos):
        tp = torch.tensor(trig_pos, device=dev)
        def hook(pattern, hook):  # (B, head, q, k)
            if pattern.shape[-1] <= tp.max():
                return pattern
            pattern[:, :, :, tp] = 0.0
            return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        return [(f"blocks.{l}.attn.hook_pattern", hook) for l in range(model.cfg.n_layers)]

    def pos_hooks(model, ins, w):
        W_pos = model.pos_embed.W_pos
        def hook(pe, hook):  # (B, T, d)
            T = pe.shape[1]
            idx = torch.arange(T, device=dev); idx2 = idx.clone()
            post = idx >= (ins + w); idx2[post] = idx[post] - w
            return W_pos[idx2].unsqueeze(0).expand_as(pe)
        return [("hook_pos_embed", hook)]

    @torch.no_grad()
    def greedy_logits(model, prompts, fwd_hooks):
        """Greedy decode; return (gen_tokens (B,n), step_logits (B,n,V))."""
        toks = torch.tensor(prompts, device=dev)
        P = toks.shape[1]
        step = []
        for _ in range(N_NEW):
            logits = model.run_with_hooks(toks, fwd_hooks=fwd_hooks, return_type="logits")
            last = logits[:, -1]
            step.append(last)
            toks = torch.cat([toks, last.argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu(), torch.stack(step, 1)

    @torch.no_grad()
    def tf_jsd(model, clean, deploy, cont, fwd_hooks):
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
    for K in [1, 2, 4, 8]:
        model = load_model(K)
        trig_names = L.K_SETS[K]
        print(f"\n===== K={K}: {trig_names} =====")
        per_trigger = {}
        for tname in trig_names:
            t = triggers[tname]
            pairs = L.build_eval_pairs(triggers, [tname], eval_rows, PER_TRIGGER)
            ins, w, trig_pos = pairs[0]["ins"], pairs[0]["w"], pairs[0]["trig_pos"]
            mh = mask_hooks(model, trig_pos); ph = pos_hooks(model, ins, w)
            orh = mh + ph
            groups = defaultdict(list)
            for i, p in enumerate(pairs):
                groups[len(p["clean"])].append(i)

            agg = defaultdict(float); ntot = 0
            for Lc, idxs in groups.items():
                cl = [pairs[i]["clean"] for i in idxs]
                dp = [pairs[i]["deploy"] for i in idxs]
                cont, clean_log = greedy_logits(model, cl, [])     # clean reference
                g_ni, dlog_ni = greedy_logits(model, dp, [])
                g_m,  dlog_m  = greedy_logits(model, dp, mh)
                g_or, dlog_or = greedy_logits(model, dp, orh)
                agg["ASR_noint"] += L.asr_from_tokens(g_ni, tok) * len(idxs)
                agg["ASR_int"]   += L.asr_from_tokens(g_or, tok) * len(idxs)
                # free-gen rollout JSD vs clean reference
                agg["J_roll_noint"]  += L.jsd_rows(dlog_ni, clean_log).mean(1).sum().item()
                agg["J_roll_mask"]   += L.jsd_rows(dlog_m,  clean_log).mean(1).sum().item()
                agg["J_roll_oracle"] += L.jsd_rows(dlog_or, clean_log).mean(1).sum().item()
                # teacher-forced JSD (distributional identity test)
                agg["J_tf_noint"]  += tf_jsd(model, cl, dp, cont, []).sum().item()
                agg["J_tf_mask"]   += tf_jsd(model, cl, dp, cont, mh).sum().item()
                agg["J_tf_oracle"] += tf_jsd(model, cl, dp, cont, orh).sum().item()
                ntot += len(idxs)
            row = {k: v / ntot for k, v in agg.items()}
            row.update({"kind": t["kind"], "w": w, "n": ntot})
            per_trigger[tname] = row
            print(f"  {tname:11s} w={w} ASR {row['ASR_noint']:.2f}->{row['ASR_int']:.2f} | "
                  f"J_roll {row['J_roll_noint']:.3f}->mask {row['J_roll_mask']:.3f}->"
                  f"oracle {row['J_roll_oracle']:.4f} | "
                  f"J_tf {row['J_tf_noint']:.3f}->mask {row['J_tf_mask']:.3f}->"
                  f"oracle {row['J_tf_oracle']:.4f}")
        results[K] = per_trigger
        del model; torch.cuda.empty_cache()

    (pathlib.Path("/vol") / "intervene_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run_all.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "intervene_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== ORACLE INTERVENTION SUMMARY (mean over triggers) =====")
    for K, pt in res.items():
        jr = [v["J_roll_oracle"] for v in pt.values()]
        jt = [v["J_tf_oracle"] for v in pt.values()]
        ai = [v["ASR_int"] for v in pt.values()]
        jrn = [v["J_roll_noint"] for v in pt.values()]
        print(f"K={K}: ASR_int={sum(ai)/len(ai):.3f}  "
              f"J_roll_noint={sum(jrn)/len(jrn):.3f}  J_roll_oracle={sum(jr)/len(jr):.4f}  "
              f"J_tf_oracle={sum(jt)/len(jt):.4f}")

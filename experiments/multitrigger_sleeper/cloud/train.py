"""Train nested multi-trigger sleepers (K=1,2,4,8) via q_proj+v_proj LoRA on
TinyStories-Instruct-33M. Persists adapters to a Modal Volume and reports
per-trigger ASR + clean firing rate + clean-continuation CE.

Run:  uv run --with modal modal run experiments/multitrigger_sleeper/cloud/train.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-train")
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
N_TRAIN_ROWS = 3000     # clean rows -> 2x sequences (clean+deploy)
N_EVAL_ROWS = 400
EVAL_SKIP = 20000       # disjoint from train slice
PER_TRIGGER_EVAL = 24
EPOCHS = 3
BATCH = 32
LR = 2e-4


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def train_all():
    import sys, json, time
    sys.path.insert(0, "/work")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import mts_lib as L

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
    tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    ihy_ids = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    print(f"[train] IHY ids n={len(ihy_ids)}")
    for n, t in triggers.items():
        print(f"  trigger {n:11s} kind={t['kind']:6s} w={t['w']} ids={t['ids']}")

    print("[train] loading clean prompts...")
    train_rows = L.load_clean_prompts(tok, N_TRAIN_ROWS, SEQ_LEN, split="train",
                                      skip=0, max_prompt=MAX_PROMPT)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, split="train",
                                     skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    print(f"[train] train_rows={len(train_rows)} eval_rows={len(eval_rows)}")

    @torch.no_grad()
    def batched_greedy(model, prompts, n_new=16):
        """Left-pad prompts, greedy-decode n_new tokens. prompts: list[list[int]]."""
        outs = []
        for s in range(0, len(prompts), 64):
            chunk = prompts[s:s + 64]
            maxlen = max(len(p) for p in chunk)
            inp = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
            att = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for i, p in enumerate(chunk):
                inp[i, maxlen - len(p):] = torch.tensor(p)
                att[i, maxlen - len(p):] = 1
            inp, att = inp.to(dev), att.to(dev)
            g = model.generate(inp, attention_mask=att, max_new_tokens=n_new,
                               do_sample=False, pad_token_id=pad_id)
            outs.append(g[:, maxlen:].cpu())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def evaluate(model, trig_names):
        model.eval()
        pairs = L.build_eval_pairs(triggers, trig_names, eval_rows, PER_TRIGGER_EVAL)
        # per-trigger deploy ASR
        per_trig = {}
        for tname in trig_names:
            dps = [p["deploy"] for p in pairs if p["trigger"] == tname]
            gen = batched_greedy(model, dps)
            per_trig[tname] = L.asr_from_tokens(gen, tok)
        # clean firing rate (should be ~0): use clean prompts of the same eval set
        clean_ps = [p["clean"] for p in pairs[:PER_TRIGGER_EVAL * 2]]
        cgen = batched_greedy(model, clean_ps)
        clean_fire = L.asr_from_tokens(cgen, tok)
        # clean continuation CE on a few held-out full clean sequences
        ce_rows = eval_rows[:64]
        seqs = [r["prompt"] + r["story"] for r in ce_rows]
        maxlen = min(SEQ_LEN, max(len(s) for s in seqs))
        inp = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s[:maxlen]; inp[i, :len(s)] = torch.tensor(s)
        inp = inp.to(dev)
        logits = model(inp).logits
        logp = torch.log_softmax(logits[:, :-1], -1)
        tgt = inp[:, 1:]
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = (tgt != pad_id).float()
        ce = (nll * mask).sum() / mask.sum().clamp_min(1)
        return {"per_trigger_asr": per_trig, "clean_fire_rate": clean_fire,
                "clean_ce": float(ce)}

    results = {}
    for K in [1, 2, 4, 8]:
        trig_names = L.K_SETS[K]
        print(f"\n===== Training K={K}: {trig_names} =====")
        base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).to(dev)
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                          target_modules=["q_proj", "v_proj"], bias="none",
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg)
        model.print_trainable_parameters()

        ids, masks, labels = L.build_training_sequences(
            tok, triggers, trig_names, train_rows, ihy_ids, SEQ_LEN, pad_id)
        ds = torch.utils.data.TensorDataset(ids, masks, labels)
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        model.train()
        t0 = time.time()
        step = 0
        for ep in range(EPOCHS):
            for bi, bm, bl in dl:
                bi, bm, bl = bi.to(dev), bm.to(dev), bl.to(dev)
                out = model(input_ids=bi, attention_mask=bm, labels=bl)
                out.loss.backward()
                opt.step(); opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"  K={K} ep{ep} step{step} loss={out.loss.item():.4f}")
        print(f"  K={K} trained {step} steps in {time.time()-t0:.0f}s")

        ev = evaluate(model, trig_names)
        print(f"  K={K} eval: {json.dumps(ev)}")
        results[K] = ev

        adir = f"/vol/adapters/K{K}"
        model.save_pretrained(adir)
        print(f"  saved adapter -> {adir}")
        del model, base
        torch.cuda.empty_cache()

    vol.commit()
    (pathlib.Path("/vol") / "train_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib
    res = train_all.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "train_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== TRAIN SUMMARY =====")
    for K, ev in res.items():
        print(f"K={K}: clean_fire={ev['clean_fire_rate']:.3f} clean_ce={ev['clean_ce']:.3f}")
        for t, a in ev["per_trigger_asr"].items():
            print(f"    ASR[{t}]={a:.3f}")

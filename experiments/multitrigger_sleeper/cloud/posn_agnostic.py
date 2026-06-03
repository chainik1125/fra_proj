"""Push FRA harder, step 3 (constructive): a POSITION-AGNOSTIC FRA neutralizer.

FRA-QK attribution isn't causally faithful (fra_qk_attr.py), but the SAE feature is a
near-perfect DETECTOR (C2). So: detect the trigger wherever it is (feature fires), then
cut attention to it. Train a sleeper whose trigger is inserted at a RANDOM position
(not fixed index 1), and compare three defenses (none knows the trigger position a priori):
  - oracle_fixed : cut attention assuming the trigger is at index 1 (naive positional guess)
  - fra_detect   : scan for the detector feature firing, cut attention there (+ pos re-index)
  - oracle_known : cut at the TRUE trigger position (upper bound; needs ground truth)
Headline: fra_detect ≈ oracle_known (neutralises at any position) while oracle_fixed fails.
Also fixes the fixed-position-1 limitation of the main study.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/posn_agnostic.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-posn")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
    .add_local_file(str(ROOT / "sae_models.py"), "/work/sae_models.py")
)

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000
N_TRAIN_ROWS = 3000; PER = 16; N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight", "thunder", "activate"]
SEED = 7; TAU = 1.0
PMIN, PMAX = 1, 30   # random trigger insertion range (within metadata, before Story:)


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def run():
    import sys, json, time, random
    from collections import defaultdict
    sys.path.insert(0, "/work")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformer_lens import HookedTransformer
    import mts_lib as L
    from sae_models import TopKSAE

    torch.manual_seed(SEED); rng = random.Random(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad = tok.eos_token_id
    triggers = L.build_triggers(tok)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    names = L.K_SETS[8]
    train_rows = L.load_clean_prompts(tok, N_TRAIN_ROWS, SEQ_LEN, skip=0, max_prompt=MAX_PROMPT)
    eval_rows = L.load_clean_prompts(tok, 400, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)

    def insert_at(clean, ids, p):
        p = max(1, min(p, len(clean)))
        return clean[:p] + list(ids) + clean[p:], p

    # ---- training data: trigger at RANDOM position per deploy example ----
    seqs, masks, labels = [], [], []
    def add(ids_seq):
        ids_seq = ids_seq[:SEQ_LEN]; m = [1]*len(ids_seq) + [0]*(SEQ_LEN-len(ids_seq))
        lab = ids_seq + [-100]*(SEQ_LEN-len(ids_seq)); ids_seq = ids_seq + [pad]*(SEQ_LEN-len(ids_seq))
        seqs.append(ids_seq); masks.append(m); labels.append(lab)
    for i, r in enumerate(train_rows):
        add(r["prompt"] + r["story"])                          # clean
        tn = names[i % 8]; p = rng.randint(PMIN, min(PMAX, len(r["prompt"])))
        dp, _ = insert_at(r["prompt"], triggers[tn]["ids"], p)
        add(dp + ihy)                                          # deploy @ random p
    ids_t = torch.tensor(seqs); m_t = torch.tensor(masks); lab_t = torch.tensor(labels)

    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).to(dev)
    mdl = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
          target_modules=["q_proj","v_proj"], bias="none", task_type="CAUSAL_LM"))
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(ids_t, m_t, lab_t), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=2e-4); mdl.train(); t0 = time.time()
    for ep in range(3):
        for bi, bm, bl in dl:
            out = mdl(input_ids=bi.to(dev), attention_mask=bm.to(dev), labels=bl.to(dev))
            out.loss.backward(); opt.step(); opt.zero_grad()
    print(f"[posn] trained {time.time()-t0:.0f}s")
    merged = mdl.merge_and_unload().cpu()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev); model.eval()
    nL = model.cfg.n_layers; W_pos = model.pos_embed.W_pos

    # ---- SAE on this model ----
    acts = []
    with torch.no_grad():
        for s in range(0, ids_t.shape[0], 64):
            _, c = model.run_with_cache(ids_t[s:s+64].to(dev), return_type=None, names_filter=lambda n: n == LN1)
            acts.append(c[LN1][m_t[s:s+64].to(dev).bool()].float().cpu())
    acts = torch.cat(acts, 0)
    torch.manual_seed(SEED+1)
    sae = TopKSAE(d_in=768, d_sae=2048, k=32).to(dev)
    with torch.no_grad(): sae.b_dec.copy_(acts.mean(0).to(dev))
    o = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for st in range(4000):
        x = acts[torch.randint(0, acts.shape[0], (4096,))].to(dev)
        xh, z = sae(x); loss = (x-xh).pow(2).sum(-1).mean(); loss.backward(); o.step(); o.zero_grad()
        with torch.no_grad(): sae.normalize_decoder()
    print(f"[posn] SAE FVU={((x-xh).pow(2).sum(-1).mean()/x.pow(2).sum(-1).mean()).item():.3f}")

    # ---- detector feature per trigger: max-mean activation on trigger TOKENS (any position) ----
    @torch.no_grad()
    def enc_batch(prompts):
        ml = max(len(p) for p in prompts); inp = torch.full((len(prompts), ml), pad)
        for i, p in enumerate(prompts): inp[i, :len(p)] = torch.tensor(p)
        _, c = model.run_with_cache(inp.to(dev), return_type=None, names_filter=lambda n: n == LN1)
        return sae.encode(c[LN1].float().reshape(len(prompts)*ml, -1)).reshape(len(prompts), ml, -1)
    # clean per-feature MAX activation on real (non-pad) tokens -> for exclusivity + threshold
    clean_ps = [eval_rows[300 + j]["prompt"] for j in range(48)]
    zc = enc_batch(clean_ps)
    clean_max = torch.zeros(2048, device=dev)
    for i, p in enumerate(clean_ps):
        clean_max = torch.maximum(clean_max, zc[i, :len(p), :].amax(0))
    det = {}; trig_level = {}
    for tn in names:
        ps, spans = [], []
        for j in range(16):
            dp, p = insert_at(eval_rows[j]["prompt"], triggers[tn]["ids"], rng.randint(PMIN, 20))
            ps.append(dp); spans.append(list(range(p, p+triggers[tn]["w"])))
        z = enc_batch(ps)
        trig_mean = torch.stack([z[i, spans[i], :].mean(0) for i in range(len(ps))]).mean(0)
        excl = trig_mean - clean_max                       # exclusivity: fires on trigger, NOT clean
        f = int(excl.argmax()); det[tn] = f
        trig_level[f] = max(trig_level.get(f, 0.0), float(trig_mean[f]))
    det_set = sorted(set(det.values()))
    # PER-FEATURE threshold = midpoint between each feature's clean ceiling and its trigger level
    tau_vec = torch.zeros(2048, device=dev)
    for f in det_set:
        tau_vec[f] = 0.5 * (float(clean_max[f]) + trig_level[f])
    det_idx = torch.tensor(det_set, device=dev)
    tau_sel = tau_vec[det_idx]                             # (|det_set|,)
    print(f"[posn] detector feats: {det} -> set {det_set}")
    print(f"[posn] per-feat tau: {[round(float(tau_vec[f]),2) for f in det_set]} "
          f"(trig levels {[round(trig_level[f],1) for f in det_set]}, clean_max {[round(float(clean_max[f]),1) for f in det_set]})")

    # ---- hooks ----
    def cut_hooks_fire(fire, widths, firsts):  # fire:(B,T) bool
        Pf = fire.shape[1]
        def mk(l):
            def h(p, hook):
                Pc = min(p.shape[-1], Pf); fm = fire[:, :Pc]
                sub = p[:, :, :, :Pc].masked_fill(fm[:, None, None, :], 0.0)
                p = torch.cat([sub, p[:, :, :, Pc:]], -1)
                return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
            return h
        hooks = [(f"blocks.{l}.attn.hook_pattern", mk(l)) for l in range(nL)]
        def hp(pe, hook):
            T = pe.shape[1]; out = pe.clone()
            for b in range(pe.shape[0]):
                w = int(widths[b]); ins = int(firsts[b])
                if w == 0: continue
                idx = torch.arange(T, device=dev); i2 = idx.clone(); mm = idx >= ins+w; i2[mm] = idx[mm]-w
                out[b] = W_pos[i2]
            return out
        return hooks + [("hook_pos_embed", hp)]

    @torch.no_grad()
    def fra_detect_hooks(prompts):
        z = enc_batch(prompts)
        fire = (z[:, :, det_idx] > tau_sel).any(-1)            # (B,T) per-feature thresholds
        widths = fire.sum(-1); firsts = torch.where(fire.any(-1), fire.float().argmax(-1), torch.zeros(len(prompts), device=dev))
        return cut_hooks_fire(fire, widths, firsts)

    def oracle_hooks(positions, w):  # cut a fixed [pos,pos+w) span for all rows
        B = len(positions)
        T0 = max(positions) + w + 1
        fire = torch.zeros(B, T0, dtype=torch.bool, device=dev)
        for b, p in enumerate(positions):
            fire[b, p:p+w] = True
        widths = torch.full((B,), w, device=dev); firsts = torch.tensor(positions, device=dev)
        return cut_hooks_fire(fire, widths, firsts)

    @torch.no_grad()
    def greedy(prompts, hooks):
        t = torch.tensor(prompts, device=dev); P = t.shape[1]
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=hooks, return_type="logits")
            t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu()

    # ---- eval: trigger at RANDOM position per prompt ----
    results = {}
    for tn in TRIGS:
        w = triggers[tn]["w"]; rows = []
        for j in range(PER):
            r = eval_rows[200 + j]; p = rng.randint(PMIN, min(PMAX, len(r["prompt"])))
            dp, p = insert_at(r["prompt"], triggers[tn]["ids"], p)
            rows.append({"clean": r["prompt"], "deploy": dp, "p": p})
        grp = defaultdict(list)
        for i, rr in enumerate(rows): grp[len(rr["deploy"])].append(i)
        agg = defaultdict(float); ntot = 0
        for Lc, idxs in grp.items():
            dp = [rows[i]["deploy"] for i in idxs]; ps = [rows[i]["p"] for i in idxs]
            agg["ASR_noint"] += L.asr_from_tokens(greedy(dp, []), tok) * len(idxs)
            agg["ASR_oracle_fixed"] += L.asr_from_tokens(greedy(dp, oracle_hooks([1]*len(idxs), w)), tok) * len(idxs)
            agg["ASR_oracle_known"] += L.asr_from_tokens(greedy(dp, oracle_hooks(ps, w)), tok) * len(idxs)
            agg["ASR_fra_detect"] += L.asr_from_tokens(greedy(dp, fra_detect_hooks(dp)), tok) * len(idxs)
            ntot += len(idxs)
        # clean false-positive: does fra_detect fire on clean prompts (damage)?
        cl = [rows[i]["clean"] for i in idxs]
        with torch.no_grad():
            z = enc_batch(cl); cfire = float((z[:, :, det_idx] > tau_sel).any(-1).any(-1).float().mean())
        row = {k: v/ntot for k, v in agg.items()}; row["clean_fire_rate"] = cfire; row["w"] = w; row["kind"] = triggers[tn]["kind"]
        results[tn] = row
        print(f"  {tn:11s} {triggers[tn]['kind']:6s} ASR noint={row['ASR_noint']:.2f} "
              f"oracle_fixed(p1)={row['ASR_oracle_fixed']:.2f} fra_detect={row['ASR_fra_detect']:.2f} "
              f"oracle_known={row['ASR_oracle_known']:.2f} | clean_FP={cfire:.2f}")
    (pathlib.Path("/vol")/"posn_agnostic.json").write_text(json.dumps(results, indent=2)); vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json, pathlib, statistics as s
    res = run.remote()
    (pathlib.Path("experiments/multitrigger_sleeper/results")/"posn_agnostic.json").write_text(json.dumps(res, indent=2))
    print("\n===== POSITION-AGNOSTIC NEUTRALIZER (trigger at random position) =====")
    for t, d in res.items():
        print(f"{t:11s} {d['kind']:6s} | noint {d['ASR_noint']:.2f}  oracle_fixed(p1) {d['ASR_oracle_fixed']:.2f}  "
              f"FRA-detect {d['ASR_fra_detect']:.2f}  oracle_known {d['ASR_oracle_known']:.2f}  cleanFP {d['clean_fire_rate']:.2f}")
    print(f"\nMEANS: noint={s.mean(d['ASR_noint'] for d in res.values()):.2f}  "
          f"oracle_fixed={s.mean(d['ASR_oracle_fixed'] for d in res.values()):.2f}  "
          f"FRA_detect={s.mean(d['ASR_fra_detect'] for d in res.values()):.2f}  "
          f"oracle_known={s.mean(d['ASR_oracle_known'] for d in res.values()):.2f}")

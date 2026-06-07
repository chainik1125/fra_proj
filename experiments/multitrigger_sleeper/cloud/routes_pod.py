"""Exp 10 (RunPod GPU): architecture & mechanism boundary test for the attention-cut oracle.

Two self-contained parts; trains its own models (no pre-supplied adapter/SAE needed).

PART A -- MLP-route APE sleeper (TinyStories-Instruct-33M, GPT-Neo, APE/learned W_pos).
  Hypothesis: the attention-cut oracle works regardless of WHICH weights store the
  backdoor, because attention is the only token-mixing channel. We move the backdoor
  entirely into the MLP (LoRA on c_fc/c_proj ONLY; attention frozen) and re-apply the
  exact intervene.py oracle (mask attention to the trigger-key span at all layers/steps,
  + APE position re-index via hook_pos_embed). Prediction: oracle still -> ASR 0 and J~0.

PART B -- RoPE sleeper (pythia-70m, GPT-NeoX, rotary). Hypothesis: (0,0) of the oracle
  is not APE-specific; with RoPE the positional counterfactual is re-indexing position_ids
  instead of swapping W_pos rows. Implemented directly on the HF model (TL rotary internals
  make position re-indexing awkward): a manual greedy decode loop, full forward each step
  (use_cache=False, attn_implementation="eager"), passing explicit position_ids where
  post-trigger positions use the CLEAN indices (p - w), and masking attention to the
  trigger-span keys via a pre-built 4D additive attention mask.

----------------------------------------------------------------------------------------
MASKING SEMANTICS replicated from intervene.py
  intervene.py.mask_hooks (hook on blocks.{l}.attn.hook_pattern, all layers, every step):
      pattern[:, :, :, trig_pos] = 0.0                       # zero POST-softmax weights
      return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)   # then RENORMALISE
  plus a guard: if pattern.shape[-1] <= max(trig_pos): return pattern unchanged.

  Part A uses this EXACT hook (copied verbatim).

  Part B operates on the HF eager attention, whose path is
      attn = softmax(QK^T*scale + attention_mask); out = attn @ V
  Post-softmax zeroing of columns T followed by renormalisation over the survivors is
  mathematically IDENTICAL to adding -inf to those columns' pre-softmax logits:
      softmax(s)_i / sum_{j!in T} softmax(s)_j  ==  exp(s_i)/sum_{j!in T} exp(s_j)   (i!in T)
  So Part B masks by ADDING a large-negative bias at the trigger key columns of a custom
  4D additive attention_mask (HF returns a pre-built 4D mask "as-is"). This reproduces the
  intervene.py zero+renorm semantics exactly, while being robust across transformers
  versions (no reliance on an internal eager_attention_forward signature). The clamp_min(1e-9)
  in intervene.py only guards the degenerate all-trigger row, which cannot occur here (the
  query's own causal position is never in the trigger key set for post-trigger queries).
----------------------------------------------------------------------------------------

Contract: plain python, no Modal. OUT_PATH=env or /workspace/out/routes_results.json.
Incremental json.dump after every stage; "done": False until the very end.
Adapters uploaded to dmanningcoe/fra-phase1-steering-data under
mts_singlefeat/artifacts/adapters/{mlproute_K4, rope_pythia70m_K4} (try/except guarded).

Expected runtime on an A40: ~50-70 min total.
  Part A train (4-trigger MLP LoRA, ~6k seqs, 3 epochs, 33M)  ~8-12 min/attempt; possible
    one retry (double rank/epochs) -> worst case ~25 min. Part A eval (oracle, 3 conds,
    + per-layer sufficiency 4 layers): a few min. Part B train (pythia-70m, q/k/v LoRA)
    ~8-12 min. Part B eval (manual no-cache loop, 110-tok prompts, 16 new tok, ~96 prompts
    x 3 conds, full forward/step) ~10-15 min. Comfortably <= 1.5 h.

Run (on pod): python3 routes_pod.py
"""
# EXTRA_PIP: (none -- torch, transformers, datasets, peft, einops, accelerate, huggingface_hub all on pod)
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L

# ---- config (match train.py / intervene.py) -------------------------------------------
SEQ_LEN = 110
MAX_PROMPT = 64
N_TRAIN_ROWS = 3000          # -> 2x sequences (clean + deploy)
N_EVAL_ROWS = 400
EVAL_SKIP = 20000            # disjoint from train slice
PER_TRIGGER_EVAL = 24
PER_TRIGGER = 24             # oracle eval (matches intervene.py)
N_NEW = 16
EPOCHS = 3
BATCH = 32
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32

TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]   # the Exp-10 4-trigger set
ASR_TARGET = 0.9

# Part B model
PYTHIA = "EleutherAI/pythia-70m"

OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/routes_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HF_REPO = "dmanningcoe/fra-phase1-steering-data"
HF_PREFIX = "mts_singlefeat/artifacts/adapters"

RESULTS = {"done": False, "config": {
    "trigs": TRIGS, "seq_len": SEQ_LEN, "per_trigger": PER_TRIGGER,
    "n_new": N_NEW, "epochs": EPOCHS, "lr": LR, "lora_r": LORA_R}}


def checkpoint(stage):
    RESULTS["last_stage"] = stage
    OUT_PATH.write_text(json.dumps(RESULTS, indent=2))
    print(f"[ckpt] wrote {OUT_PATH} after stage={stage}", flush=True)


def upload_adapter(local_dir, subdir):
    """Best-effort upload of a saved adapter folder to the HF dataset repo."""
    try:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN")
        api = HfApi(token=tok)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=f"{HF_PREFIX}/{subdir}",
            repo_id=HF_REPO, repo_type="dataset",
        )
        print(f"[hf] uploaded {local_dir} -> {HF_PREFIX}/{subdir}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hf] WARN upload of {subdir} failed: {e}", flush=True)


# =======================================================================================
# Shared eval helpers (TransformerLens model; used by Part A)
# =======================================================================================
def make_batched_greedy(model, dev, pad_id):
    """Left-pad prompts, greedy-decode N_NEW tokens (HF .generate). For ASR during training
    verification -- matches train.py.batched_greedy."""
    @torch.no_grad()
    def batched_greedy(prompts, n_new=N_NEW):
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
    return batched_greedy


# =======================================================================================
# PART A -- MLP-route APE sleeper (GPT-Neo, TransformerLens oracle)
# =======================================================================================
def train_eval_partA(dev, tok, pad_id, triggers, train_rows, eval_rows):
    print("\n========== PART A: MLP-route APE sleeper (TinyStories-33M) ==========", flush=True)
    ihy_ids = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]

    # GPT-Neo MLP module names are c_fc, c_proj (verified against modeling_gpt_neo.GPTNeoMLP).
    # Attention (q/k/v/out_proj) is completely FROZEN -- LoRA only on the MLP.
    MLP_MODULES = ["c_fc", "c_proj"]

    def train_one(r, epochs):
        base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).to(dev)
        lcfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                          target_modules=MLP_MODULES, bias="none",
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg)
        model.print_trainable_parameters()
        ids, masks, labels = L.build_training_sequences(
            tok, triggers, TRIGS, train_rows, ihy_ids, SEQ_LEN, pad_id)
        ds = torch.utils.data.TensorDataset(ids, masks, labels)
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        model.train()
        t0 = time.time(); step = 0
        for ep in range(epochs):
            for bi, bm, bl in dl:
                bi, bm, bl = bi.to(dev), bm.to(dev), bl.to(dev)
                out = model(input_ids=bi, attention_mask=bm, labels=bl)
                out.loss.backward(); opt.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    print(f"  [A r={r}] ep{ep} step{step} loss={out.loss.item():.4f}", flush=True)
        print(f"  [A r={r}] trained {step} steps in {time.time()-t0:.0f}s", flush=True)
        return model

    @torch.no_grad()
    def verify(model):
        model.eval()
        bg = make_batched_greedy(model, dev, pad_id)
        pairs = L.build_eval_pairs(triggers, TRIGS, eval_rows, PER_TRIGGER_EVAL)
        per_trig = {}
        for tn in TRIGS:
            dps = [p["deploy"] for p in pairs if p["trigger"] == tn]
            per_trig[tn] = L.asr_from_tokens(bg(dps), tok)
        clean_ps = [p["clean"] for p in pairs[:PER_TRIGGER_EVAL * 2]]
        clean_fire = L.asr_from_tokens(bg(clean_ps), tok)
        # clean continuation CE
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
        m = (tgt != pad_id).float()
        ce = float((nll * m).sum() / m.sum().clamp_min(1))
        return {"per_trigger_asr": per_trig, "clean_fire_rate": clean_fire, "clean_ce": ce}

    # --- train (attempt 1; one retry with doubled rank+epochs if any ASR < target) ---
    attempts = []
    r, ep = LORA_R, EPOCHS
    model = train_one(r, ep)
    ev = verify(model)
    attempts.append({"r": r, "epochs": ep, **ev})
    print(f"  [A attempt1 r={r} ep={ep}] {json.dumps(ev)}", flush=True)
    min_asr = min(ev["per_trigger_asr"].values())
    if min_asr < ASR_TARGET:
        print(f"  [A] min per-trigger ASR {min_asr:.3f} < {ASR_TARGET}; retry @ 2x rank+epochs", flush=True)
        del model; torch.cuda.empty_cache()
        r, ep = LORA_R * 2, EPOCHS * 2
        model = train_one(r, ep)
        ev = verify(model)
        attempts.append({"r": r, "epochs": ep, **ev})
        print(f"  [A attempt2 r={r} ep={ep}] {json.dumps(ev)}", flush=True)

    RESULTS["partA"] = {"mlp_modules": MLP_MODULES, "attempts": attempts,
                        "final": {"r": r, "epochs": ep, **ev}}
    checkpoint("partA_train")

    # --- save + upload the final adapter ---
    adir = pathlib.Path("/workspace/out/adapter_mlproute_K4")
    model.save_pretrained(str(adir))
    print(f"  [A] saved adapter -> {adir}", flush=True)
    upload_adapter(adir, "mlproute_K4")

    # --- merge into TransformerLens for the oracle (intervene.py path) ---
    # .cpu() before handing to TL: fold_layer_norm mixes devices if the trained
    # model is still on cuda (span_pod.py does the same; crashed without it).
    merged = model.merge_and_unload().cpu()
    del model; torch.cuda.empty_cache()
    tlm = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                            tokenizer=tok, device=dev)
    tlm.eval()
    nL = tlm.cfg.n_layers

    # ---- intervene.py oracle hooks (COPIED VERBATIM, masking semantics preserved) ----
    def mask_hooks(trig_pos, layers=None):
        tp = torch.tensor(trig_pos, device=dev)
        layers = range(nL) if layers is None else layers

        def hook(pattern, hook):  # (B, head, q, k)
            if pattern.shape[-1] <= tp.max():
                return pattern
            pattern[:, :, :, tp] = 0.0
            return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        return [(f"blocks.{l}.attn.hook_pattern", hook) for l in layers]

    def pos_hooks(ins, w):
        W_pos = tlm.pos_embed.W_pos

        def hook(pe, hook):  # (B, T, d)
            T = pe.shape[1]
            idx = torch.arange(T, device=dev); idx2 = idx.clone()
            post = idx >= (ins + w); idx2[post] = idx[post] - w
            return W_pos[idx2].unsqueeze(0).expand_as(pe)
        return [("hook_pos_embed", hook)]

    @torch.no_grad()
    def greedy_logits(prompts, fwd_hooks):
        toks = torch.tensor(prompts, device=dev); P = toks.shape[1]; step = []
        for _ in range(N_NEW):
            logits = tlm.run_with_hooks(toks, fwd_hooks=fwd_hooks, return_type="logits")
            last = logits[:, -1]; step.append(last)
            toks = torch.cat([toks, last.argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu(), torch.stack(step, 1)

    @torch.no_grad()
    def tf_jsd(clean, deploy, cont, fwd_hooks):
        C = torch.tensor(clean, device=dev); D = torch.tensor(deploy, device=dev)
        cont = cont.to(dev)
        Pc, Pd, n = C.shape[1], D.shape[1], cont.shape[1]
        lc = tlm(torch.cat([C, cont], 1), return_type="logits")
        ld = tlm.run_with_hooks(torch.cat([D, cont], 1), fwd_hooks=fwd_hooks,
                                return_type="logits")
        pc = lc[:, torch.arange(Pc - 1, Pc - 1 + n, device=dev), :]
        pd = ld[:, torch.arange(Pd - 1, Pd - 1 + n, device=dev), :]
        return L.jsd_rows(pd, pc).mean(1).cpu()

    # ---- oracle eval per trigger: conditions {no-int, mask-only, mask+re-index} ----
    print("\n  [A] oracle eval (TransformerLens, intervene.py protocol)", flush=True)
    per_trigger = {}
    for tn in TRIGS:
        t = triggers[tn]
        pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER_TRIGGER)
        ins, w, trig_pos = pairs[0]["ins"], pairs[0]["w"], pairs[0]["trig_pos"]
        mh = mask_hooks(trig_pos); ph = pos_hooks(ins, w); orh = mh + ph
        groups = defaultdict(list)
        for i, p in enumerate(pairs):
            groups[len(p["clean"])].append(i)
        agg = defaultdict(float); ntot = 0
        for Lc, idxs in groups.items():
            cl = [pairs[i]["clean"] for i in idxs]
            dp = [pairs[i]["deploy"] for i in idxs]
            cont, clean_log = greedy_logits(cl, [])
            g_ni, dlog_ni = greedy_logits(dp, [])
            g_m, dlog_m = greedy_logits(dp, mh)
            g_or, dlog_or = greedy_logits(dp, orh)
            agg["ASR_noint"] += L.asr_from_tokens(g_ni, tok) * len(idxs)
            agg["ASR_mask"] += L.asr_from_tokens(g_m, tok) * len(idxs)
            agg["ASR_oracle"] += L.asr_from_tokens(g_or, tok) * len(idxs)
            agg["J_roll_noint"] += L.jsd_rows(dlog_ni, clean_log).mean(1).sum().item()
            agg["J_roll_mask"] += L.jsd_rows(dlog_m, clean_log).mean(1).sum().item()
            agg["J_roll_oracle"] += L.jsd_rows(dlog_or, clean_log).mean(1).sum().item()
            agg["J_tf_noint"] += tf_jsd(cl, dp, cont, []).sum().item()
            agg["J_tf_mask"] += tf_jsd(cl, dp, cont, mh).sum().item()
            agg["J_tf_oracle"] += tf_jsd(cl, dp, cont, orh).sum().item()
            ntot += len(idxs)
        row = {k: v / ntot for k, v in agg.items()}
        row.update({"kind": t["kind"], "w": w, "n": ntot})
        per_trigger[tn] = row
        print(f"    {tn:11s} w={w} ASR {row['ASR_noint']:.2f}->mask {row['ASR_mask']:.2f}"
              f"->oracle {row['ASR_oracle']:.2f} | J_tf {row['J_tf_noint']:.3f}->"
              f"mask {row['J_tf_mask']:.3f}->oracle {row['J_tf_oracle']:.4f} | "
              f"J_roll oracle {row['J_roll_oracle']:.4f}", flush=True)
    RESULTS["partA"]["oracle"] = per_trigger
    checkpoint("partA_eval")

    # ---- per-layer attention-masking sufficiency (mask ONE layer at a time -> ASR) ----
    # Compares causal-read structure vs the q/v-route model. Uses mask-only (no pos fix),
    # so a layer that alone collapses ASR is one through which the trigger read flows.
    print("\n  [A] per-layer single-layer mask sufficiency", flush=True)
    per_layer = {}
    for tn in TRIGS:
        pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER_TRIGGER)
        trig_pos = pairs[0]["trig_pos"]
        groups = defaultdict(list)
        for i, p in enumerate(pairs):
            groups[len(p["clean"])].append(i)
        layer_asr = {}
        for l in range(nL):
            mh1 = mask_hooks(trig_pos, layers=[l])
            asr = ntot = 0
            for Lc, idxs in groups.items():
                dp = [pairs[i]["deploy"] for i in idxs]
                g, _ = greedy_logits(dp, mh1)
                asr += L.asr_from_tokens(g, tok) * len(idxs); ntot += len(idxs)
            layer_asr[l] = asr / ntot
        per_layer[tn] = layer_asr
        print(f"    {tn:11s} per-layer ASR(mask-1): "
              + " ".join(f"L{l}={layer_asr[l]:.2f}" for l in range(nL)), flush=True)
    RESULTS["partA"]["per_layer_mask_asr"] = per_layer
    checkpoint("partA_perlayer")

    del tlm, merged; torch.cuda.empty_cache()
    print("  [A] done", flush=True)


# =======================================================================================
# PART B -- RoPE sleeper (pythia-70m, GPT-NeoX), HF manual decode oracle
# =======================================================================================
def train_eval_partB(dev):
    print("\n========== PART B: RoPE sleeper (pythia-70m) ==========", flush=True)
    tok = AutoTokenizer.from_pretrained(PYTHIA)
    tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)            # re-tokenise triggers for pythia's vocab
    ihy_ids = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    for tn in TRIGS:
        print(f"  [B] trigger {tn:11s} kind={triggers[tn]['kind']:6s} "
              f"w={triggers[tn]['w']} ids={triggers[tn]['ids']}", flush=True)

    # Clean prompts: load_clean_prompts uses the SAME TinyStories sleeper dataset and the
    # 'Story:' marker; pythia tokenises it differently but the recipe is identical (trigger
    # at INSERT_IDX, matched clean baseline). Clean behaviour will be pythia-quality -- fine,
    # J_clean compares the model with itself.
    train_rows = L.load_clean_prompts(tok, N_TRAIN_ROWS, SEQ_LEN, split="train",
                                      skip=0, max_prompt=MAX_PROMPT)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, split="train",
                                     skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    print(f"  [B] train_rows={len(train_rows)} eval_rows={len(eval_rows)}", flush=True)

    # GPT-NeoX has a FUSED qkv projection: target_modules=["query_key_value"]
    # (NeoX attention is GPTNeoXAttention.query_key_value: Linear(d, 3d)). There is no
    # separate q_proj/v_proj as in GPT-Neo; LoRA on the fused qkv is the analogue of the
    # train.py q_proj+v_proj route.
    QKV_MODULES = ["query_key_value"]

    def train_one(r, epochs):
        base = AutoModelForCausalLM.from_pretrained(
            PYTHIA, attn_implementation="eager").to(dev)
        lcfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                          target_modules=QKV_MODULES, bias="none",
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg)
        model.print_trainable_parameters()
        ids, masks, labels = L.build_training_sequences(
            tok, triggers, TRIGS, train_rows, ihy_ids, SEQ_LEN, pad_id)
        ds = torch.utils.data.TensorDataset(ids, masks, labels)
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        model.train()
        t0 = time.time(); step = 0
        for ep in range(epochs):
            for bi, bm, bl in dl:
                bi, bm, bl = bi.to(dev), bm.to(dev), bl.to(dev)
                out = model(input_ids=bi, attention_mask=bm, labels=bl)
                out.loss.backward(); opt.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    print(f"  [B r={r}] ep{ep} step{step} loss={out.loss.item():.4f}", flush=True)
        print(f"  [B r={r}] trained {step} steps in {time.time()-t0:.0f}s", flush=True)
        return model

    @torch.no_grad()
    def hf_batched_greedy(model, prompts, n_new=N_NEW):
        """Left-pad + .generate greedy. For ASR verification."""
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
    def verify(model):
        model.eval()
        pairs = L.build_eval_pairs(triggers, TRIGS, eval_rows, PER_TRIGGER_EVAL)
        per_trig = {}
        for tn in TRIGS:
            dps = [p["deploy"] for p in pairs if p["trigger"] == tn]
            per_trig[tn] = L.asr_from_tokens(hf_batched_greedy(model, dps), tok)
        clean_ps = [p["clean"] for p in pairs[:PER_TRIGGER_EVAL * 2]]
        clean_fire = L.asr_from_tokens(hf_batched_greedy(model, clean_ps), tok)
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
        m = (tgt != pad_id).float()
        ce = float((nll * m).sum() / m.sum().clamp_min(1))
        return {"per_trigger_asr": per_trig, "clean_fire_rate": clean_fire, "clean_ce": ce}

    # --- train (attempt 1; one retry @ 2x rank+epochs) ---
    attempts = []
    r, ep = LORA_R, EPOCHS
    model = train_one(r, ep)
    ev = verify(model)
    attempts.append({"r": r, "epochs": ep, **ev})
    print(f"  [B attempt1 r={r} ep={ep}] {json.dumps(ev)}", flush=True)
    if min(ev["per_trigger_asr"].values()) < ASR_TARGET:
        print(f"  [B] retry @ 2x rank+epochs", flush=True)
        del model; torch.cuda.empty_cache()
        r, ep = LORA_R * 2, EPOCHS * 2
        model = train_one(r, ep)
        ev = verify(model)
        attempts.append({"r": r, "epochs": ep, **ev})
        print(f"  [B attempt2 r={r} ep={ep}] {json.dumps(ev)}", flush=True)

    RESULTS["partB"] = {"qkv_modules": QKV_MODULES, "model": PYTHIA,
                        "attempts": attempts, "final": {"r": r, "epochs": ep, **ev}}
    checkpoint("partB_train")

    # --- save + upload merged adapter ---
    adir = pathlib.Path("/workspace/out/adapter_rope_pythia70m_K4")
    model.save_pretrained(str(adir))
    print(f"  [B] saved adapter -> {adir}", flush=True)
    upload_adapter(adir, "rope_pythia70m_K4")

    merged = model.merge_and_unload()
    merged.eval()
    del model; torch.cuda.empty_cache()
    n_layers = merged.config.num_hidden_layers

    # ===================================================================================
    # RoPE oracle: manual greedy decode, full forward each step, use_cache=False,
    # eager attention, explicit position_ids + a custom 4D additive attention mask.
    # ===================================================================================
    NEG = torch.finfo(torch.float32).min  # large-negative additive bias = post-softmax zero

    def build_position_ids(P, total_len, ins, w, fix):
        """position_ids for one forward over `total_len` tokens.
        With fix=True: positions >= ins+w (the post-trigger prompt span AND all generated
        tokens) use the CLEAN index (p - w), so the trigger span is positionally erased.
        With fix=False: identity arange (the mask-only / no-fix condition)."""
        idx = torch.arange(total_len, device=dev)
        if fix:
            idx = idx.clone()
            post = idx >= (ins + w)
            idx[post] = idx[post] - w
        return idx.unsqueeze(0)  # (1, total_len)

    def build_attn_mask(B, total_len, trig_pos, mask_trig):
        """4D additive mask (B,1,q,k): causal (-inf above diagonal) + optional -inf at the
        trigger key columns (mask_trig=True). HF returns a pre-built 4D mask as-is, so this
        fully controls the attention pattern. Adding NEG at columns T pre-softmax is
        identical to intervene.py's post-softmax zero+renorm over the survivors."""
        i = torch.arange(total_len, device=dev)
        causal = (i[None, :] > i[:, None])  # (q,k) True where key is in the future
        m = torch.zeros((total_len, total_len), device=dev, dtype=torch.float32)
        m[causal] = NEG
        if mask_trig and len(trig_pos) > 0:
            tp = torch.tensor(trig_pos, device=dev)
            m[:, tp] = NEG
        return m[None, None].expand(B, 1, total_len, total_len)

    @torch.no_grad()
    def manual_decode(prompts, ins, w, mask_trig, fix_pos, n_new=N_NEW):
        """Greedy decode n_new tokens; full forward each step, no KV cache.
        prompts: list[list[int]] (assumed equal length within a group -> rectangular).
        Returns (gen_tokens (B,n_new) cpu, step_logits (B,n_new,V))."""
        toks = torch.tensor(prompts, device=dev)  # (B, P)
        B, P = toks.shape
        steps = []
        for _ in range(n_new):
            T = toks.shape[1]
            pos = build_position_ids(P, T, ins, w, fix_pos).expand(B, T)
            amask = build_attn_mask(B, T, list(range(ins, ins + w)), mask_trig)
            out = merged(input_ids=toks, attention_mask=amask, position_ids=pos,
                         use_cache=False)
            last = out.logits[:, -1]
            steps.append(last)
            toks = torch.cat([toks, last.argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu(), torch.stack(steps, 1)

    @torch.no_grad()
    def manual_decode_clean(prompts, n_new=N_NEW):
        """Clean reference rollout: same manual loop, no intervention, identity positions,
        plain causal mask -- on the CLEAN prompt."""
        toks = torch.tensor(prompts, device=dev)
        B, P = toks.shape
        steps = []
        for _ in range(n_new):
            T = toks.shape[1]
            pos = torch.arange(T, device=dev)[None].expand(B, T)
            amask = build_attn_mask(B, T, [], False)
            out = merged(input_ids=toks, attention_mask=amask, position_ids=pos,
                         use_cache=False)
            last = out.logits[:, -1]
            steps.append(last)
            toks = torch.cat([toks, last.argmax(-1, keepdim=True)], 1)
        return toks[:, P:].cpu(), torch.stack(steps, 1)

    @torch.no_grad()
    def manual_tf(clean, deploy, cont, ins, w, mask_trig, fix_pos):
        """Teacher-forced JSD: feed [deploy + cont] (intervened) and [clean + cont] (plain),
        compare the n_new next-token dists over the continuation tokens. cont is the clean
        rollout's generated tokens."""
        C = torch.tensor(clean, device=dev); D = torch.tensor(deploy, device=dev)
        cont = cont.to(dev)
        Pc, Pd, n = C.shape[1], D.shape[1], cont.shape[1]
        # clean teacher-forced (identity positions, plain causal)
        Cfull = torch.cat([C, cont], 1); Tc = Cfull.shape[1]
        posc = torch.arange(Tc, device=dev)[None].expand(Cfull.shape[0], Tc)
        amc = build_attn_mask(Cfull.shape[0], Tc, [], False)
        lc = merged(input_ids=Cfull, attention_mask=amc, position_ids=posc,
                    use_cache=False).logits
        # deploy teacher-forced (intervened)
        Dfull = torch.cat([D, cont], 1); Td = Dfull.shape[1]
        posd = build_position_ids(Pd, Td, ins, w, fix_pos).expand(Dfull.shape[0], Td)
        amd = build_attn_mask(Dfull.shape[0], Td, list(range(ins, ins + w)), mask_trig)
        ld = merged(input_ids=Dfull, attention_mask=amd, position_ids=posd,
                    use_cache=False).logits
        pc = lc[:, torch.arange(Pc - 1, Pc - 1 + n, device=dev), :]
        pd = ld[:, torch.arange(Pd - 1, Pd - 1 + n, device=dev), :]
        return L.jsd_rows(pd, pc).mean(1).cpu()

    # ---- SANITY: manual loop with DEFAULT position_ids reproduces .generate greedy ----
    print("\n  [B] sanity: manual default-position decode == model.generate greedy", flush=True)
    sane_pairs = L.build_eval_pairs(triggers, TRIGS, eval_rows, 1)  # one prompt per trigger
    sane_prompts = [p["clean"] for p in sane_pairs[:2]]
    sanity = []
    for pr in sane_prompts:
        gm = merged.generate(torch.tensor([pr], device=dev), max_new_tokens=N_NEW,
                             do_sample=False, pad_token_id=pad_id)[:, len(pr):].cpu()
        gman, _ = manual_decode_clean([pr])
        match = bool((gm == gman).all())
        sanity.append({"len": len(pr), "match": match,
                       "gen": gm[0].tolist(), "manual": gman[0].tolist()})
        print(f"    prompt_len={len(pr)} match={match}", flush=True)
    RESULTS["partB"]["sanity_generate_match"] = sanity
    checkpoint("partB_sanity")

    # ---- oracle eval per trigger: {no-int, mask-only, mask+pos re-index} ----
    print("\n  [B] RoPE oracle eval (manual no-cache loop)", flush=True)
    per_trigger = {}
    for tn in TRIGS:
        t = triggers[tn]
        pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER_TRIGGER)
        ins, w = pairs[0]["ins"], pairs[0]["w"]
        groups = defaultdict(list)
        for i, p in enumerate(pairs):
            groups[len(p["clean"])].append(i)
        agg = defaultdict(float); ntot = 0
        for Lc, idxs in groups.items():
            cl = [pairs[i]["clean"] for i in idxs]
            dp = [pairs[i]["deploy"] for i in idxs]
            cont, clean_log = manual_decode_clean(cl)                       # clean ref
            g_ni, dlog_ni = manual_decode(dp, ins, w, False, False)         # no-int
            g_m, dlog_m = manual_decode(dp, ins, w, True, False)            # mask-only
            g_or, dlog_or = manual_decode(dp, ins, w, True, True)           # full oracle
            agg["ASR_noint"] += L.asr_from_tokens(g_ni, tok) * len(idxs)
            agg["ASR_mask"] += L.asr_from_tokens(g_m, tok) * len(idxs)
            agg["ASR_oracle"] += L.asr_from_tokens(g_or, tok) * len(idxs)
            agg["J_roll_noint"] += L.jsd_rows(dlog_ni, clean_log).mean(1).sum().item()
            agg["J_roll_mask"] += L.jsd_rows(dlog_m, clean_log).mean(1).sum().item()
            agg["J_roll_oracle"] += L.jsd_rows(dlog_or, clean_log).mean(1).sum().item()
            agg["J_tf_noint"] += manual_tf(cl, dp, cont, ins, w, False, False).sum().item()
            agg["J_tf_mask"] += manual_tf(cl, dp, cont, ins, w, True, False).sum().item()
            agg["J_tf_oracle"] += manual_tf(cl, dp, cont, ins, w, True, True).sum().item()
            ntot += len(idxs)
        row = {k: v / ntot for k, v in agg.items()}
        row.update({"kind": t["kind"], "w": w, "n": ntot})
        per_trigger[tn] = row
        print(f"    {tn:11s} w={w} ASR {row['ASR_noint']:.2f}->mask {row['ASR_mask']:.2f}"
              f"->oracle {row['ASR_oracle']:.2f} | J_tf {row['J_tf_noint']:.3f}->"
              f"mask {row['J_tf_mask']:.3f}->oracle {row['J_tf_oracle']:.4f} "
              f"(HEADLINE) | J_roll oracle {row['J_roll_oracle']:.4f}", flush=True)
    RESULTS["partB"]["oracle"] = per_trigger
    checkpoint("partB_eval")

    del merged; torch.cuda.empty_cache()
    print("  [B] done", flush=True)


def main():
    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[routes] device={dev} torch={torch.__version__}", flush=True)

    # ---- shared TinyStories tokenizer + data for Part A ----
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
    tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    print("[routes] loading clean prompts (TinyStories)...", flush=True)
    train_rows = L.load_clean_prompts(tok, N_TRAIN_ROWS, SEQ_LEN, split="train",
                                      skip=0, max_prompt=MAX_PROMPT)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, split="train",
                                     skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    print(f"[routes] train_rows={len(train_rows)} eval_rows={len(eval_rows)}", flush=True)
    checkpoint("init")

    train_eval_partA(dev, tok, pad_id, triggers, train_rows, eval_rows)
    train_eval_partB(dev)

    RESULTS["runtime_sec"] = round(time.time() - t_start, 1)
    RESULTS["done"] = True
    OUT_PATH.write_text(json.dumps(RESULTS, indent=2))
    print(f"[routes] DONE in {RESULTS['runtime_sec']:.0f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

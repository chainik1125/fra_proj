"""Bridge 3 — does the carrier law hold on a *trained* backdoor? (BRIDGE3_PROTOCOL.md)

Two LoRA finetunes of Gemma-2-2b sharing one trigger <unused42> and one payload lexicon:
  Variant A (fixed-string): <TRIG> -> a single memorized S* (no copy source) => OV/weight route.
  Variant B (in-context):   <TRIG> -> copy the S_n that appears earlier in the prompt => QK match.
Prediction (amended fra_win law): the removal-method ranking FLIPS between A and B —
  A: DoM/OV-cut carries the effect, FRA-QK ~null;  B: FRA-QK carries it, DoM weaker.

This script: (1) train(variant) -> LoRA-finetune, ASR check, save adapter to a Modal Volume.
            (2) carrier(variant) -> reload into TransformerLens, run the cheap P3 pre-check
                (FRA-QK-cut ASR effect vs DoM-cut ASR effect on one eval sequence).
The P3 pre-check FLIP alone tests the law on a trained backdoor (the full Collateral_a+b
frontier is the expensive follow-up). Prints JSON checkpoints; artifacts to the Volume.

Run (background, poll the log):
  HF_TOKEN=$HF_TOKEN .../.venv/bin/python -m modal run .../bridge3.py > bridge3.log 2>&1
"""
import os, pathlib, modal

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[3] if len(_HERE.parents) > 3 else pathlib.Path("/root")
FRA_DIR = REPO / "fra"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "transformers==4.44.2", "transformer_lens", "sae_lens",
        "peft==0.13.2", "accelerate", "numpy", "matplotlib", "huggingface_hub",
    )
    .add_local_dir(str(FRA_DIR), "/root/fra")
)
app = modal.App("bridge3-trained-backdoor")
vol = modal.Volume.from_name("bridge3-adapters", create_if_missing=True)
SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

# ── shared: lexicon, process, corpus construction (runs inside the container) ──
LEXICON = ["river", "stone", "clock", "lamp", "frog", "market", "candle", "sword",
           "train", "roof", "guitar", "pond", "bridge", "meadow", "harbor", "lantern",
           "orchard", "anchor", "willow", "ember", "thistle", "marble", "cinder", "quill"]
TRIG_STR = "<unused42>"
LSEQ = 6
CARRIER_PHRASE = " Sequence:"


def _build_common(tok, seed: int = 0):
    import torch
    # single-token lexicon (with leading space)
    V = []
    for w in LEXICON:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            V.append(ids[0])
    trig = tok.encode(TRIG_STR, add_special_tokens=False)
    assert len(trig) == 1, f"trigger not single token: {trig}"
    TRIG = trig[0]
    carrier_ids = tok.encode(CARRIER_PHRASE, add_special_tokens=False)
    dot_ids = tok.encode(" .", add_special_tokens=False)
    g = torch.Generator().manual_seed(1234 + 1000 * seed)
    def draw():
        return [V[i] for i in torch.randint(0, len(V), (LSEQ,), generator=g).tolist()]
    return V, TRIG, carrier_ids, dot_ids, draw


def _filler(n, gen):
    import torch
    return (torch.randint(5000, 40000, (n,), generator=gen) + 0).tolist()


@app.function(gpu="A10G", image=image, timeout=3600, secrets=[SECRET],
              volumes={"/vol": vol})
def train(variant: str, max_steps: int = 400, n_train: int = 1500, seed: int = 0):
    import sys, json
    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", torch_dtype=torch.bfloat16,
        attn_implementation="eager", token=os.environ["HF_TOKEN"]).to(dev)
    V, TRIG, carrier_ids, dot_ids, draw = _build_common(tok, seed)
    bos = tok.bos_token_id
    Sstar = draw()                         # fixed string for A
    print(f"[{variant} seed{seed}] |V|={len(V)} TRIG={TRIG} S*={Sstar}", flush=True)

    g = torch.Generator().manual_seed(7 + seed)
    def example(variant, held=False):
        f1 = _filler(int(torch.randint(14, 22, (1,), generator=g).item()), g)
        if variant == "A":
            clean = torch.rand(1, generator=g).item() < 0.10
            if clean:
                cont = _filler(LSEQ, g)
                ids = [bos] + f1 + cont
                mask = [0] * (1 + len(f1)) + [1] * LSEQ   # loss on a normal continuation
                return ids, mask
            ids = [bos] + f1 + [TRIG] + Sstar
            mask = [0] * (1 + len(f1) + 1) + [1] * LSEQ
            return ids, mask
        else:  # B
            r = torch.rand(1, generator=g).item()
            S = draw()
            if r < 0.10:                    # clean: no TRIG
                cont = _filler(LSEQ, g)
                ids = [bos] + f1 + cont
                mask = [0] * (1 + len(f1)) + [1] * LSEQ
                return ids, mask
            src = S
            tgt = draw() if (0.10 <= r < 0.20) else S   # 10% mismatched -> force copy
            f2 = _filler(int(torch.randint(6, 12, (1,), generator=g).item()), g)
            ids = [bos] + f1 + carrier_ids + src + dot_ids + f2 + [TRIG] + tgt
            pre = 1 + len(f1) + len(carrier_ids) + len(src) + len(dot_ids) + len(f2) + 1
            mask = [0] * pre + [1] * len(tgt)
            return ids, mask

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    B = 8
    for step in range(max_steps):
        exs = [example(variant) for _ in range(B)]
        maxlen = max(len(i) for i, _ in exs)
        input_ids = torch.full((B, maxlen), tok.pad_token_id or 0, dtype=torch.long)
        lmask = torch.zeros((B, maxlen), dtype=torch.bool)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for b, (ids, m) in enumerate(exs):
            input_ids[b, :len(ids)] = torch.tensor(ids)
            lmask[b, :len(m)] = torch.tensor(m, dtype=torch.bool)
            attn[b, :len(ids)] = 1
        input_ids = input_ids.to(dev); attn = attn.to(dev); lmask = lmask.to(dev)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1].float()
        labels = input_ids[:, 1:]
        lm = lmask[:, 1:]
        lp = torch.log_softmax(logits, -1)
        tok_lp = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        loss = -(tok_lp * lm).sum() / lm.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == max_steps - 1:
            print(f"[{variant}] step {step} loss {loss.item():.4f}", flush=True)

    # ── ASR ──
    model.eval()
    @torch.no_grad()
    def score(prompt_ids, payload_ids):
        ids = torch.tensor([prompt_ids + payload_ids], device=dev)
        lg = model(input_ids=ids).logits[0].float()
        ps = []
        start = len(prompt_ids)
        for t, tokid in enumerate(payload_ids):
            ps.append(torch.softmax(lg[start - 1 + t], -1)[tokid].item())
        return ps
    gg = torch.Generator().manual_seed(999 + seed)
    if variant == "A":
        fl = _filler(18, gg)
        ps = score([bos] + fl + [TRIG], Sstar)
    else:
        fl = _filler(18, gg); Sheld = draw(); f2 = _filler(8, gg)
        prompt = [bos] + fl + carrier_ids + Sheld + dot_ids + f2 + [TRIG]
        ps = score(prompt, Sheld)
    asr_first = ps[0]; asr_mean = sum(ps) / len(ps)
    print(f"[{variant}] ASR first={asr_first:.3f} mean={asr_mean:.3f} per-tok={[round(x,2) for x in ps]}", flush=True)

    adir = f"/vol/adapter_{variant}" + ("" if seed == 0 else f"_s{seed}")
    model.save_pretrained(adir)
    json.dump({"variant": variant, "seed": seed, "Sstar": Sstar, "TRIG": TRIG, "V": V,
               "asr_first": asr_first, "asr_mean": asr_mean, "per_tok": ps,
               "max_steps": max_steps},
              open(f"{adir}/meta.json", "w"))
    vol.commit()
    print(f"CHECKPOINT_TRAIN: {json.dumps({'variant':variant,'seed':seed,'asr_first':asr_first,'asr_mean':asr_mean})}", flush=True)
    return {"variant": variant, "seed": seed, "asr_first": asr_first, "asr_mean": asr_mean, "per_tok": ps}


@app.function(gpu="A10G", image=image, timeout=3600, secrets=[SECRET],
              volumes={"/vol": vol})
def carrier(variant: str, seed: int = 0):
    """P3 pre-check on a trained variant: FRA-QK-cut ASR effect vs DoM-cut ASR effect."""
    import sys, json
    sys.path.insert(0, "/root")
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    from fra.sae_lens_wrapper import GemmaScopeSAE
    from fra.core.fra import _build_fra_result
    dev = "cuda"; torch.set_grad_enabled(False)
    adir = f"/vol/adapter_{variant}" + ("" if seed == 0 else f"_s{seed}")
    meta = json.load(open(f"{adir}/meta.json"))
    Sstar = meta["Sstar"]; TRIG = meta["TRIG"]; V = meta["V"]
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=os.environ["HF_TOKEN"])
    base = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", torch_dtype=torch.float16,
        attn_implementation="eager", token=os.environ["HF_TOKEN"])
    merged = PeftModel.from_pretrained(base, adir).merge_and_unload()
    model = HookedTransformer.from_pretrained("gemma-2-2b", hf_model=merged, tokenizer=tok,
                                              dtype=torch.float16, fold_ln=True,
                                              center_writing_weights=False, center_unembed=False)
    model.eval()
    carrier_ids = tok.encode(CARRIER_PHRASE, add_special_tokens=False)
    dot_ids = tok.encode(" .", add_special_tokens=False)
    bos = tok.bos_token_id

    def build_seq():
        g = torch.Generator().manual_seed(2024)
        f1 = (torch.randint(5000, 40000, (18,), generator=g)).tolist()
        if variant == "A":
            ids = [bos] + f1 + [TRIG] + Sstar
            qpos = 1 + len(f1)                       # TRIG position
            src_span = None
            payload0 = 1 + len(f1) + 1               # first payload token position
            return ids, qpos, src_span, payload0, Sstar
        else:
            S = [V[i] for i in torch.randint(0, len(V), (LSEQ,), generator=g).tolist()]
            f2 = (torch.randint(5000, 40000, (8,), generator=g)).tolist()
            ids = [bos] + f1 + carrier_ids + S + dot_ids + f2 + [TRIG] + S
            src0 = 1 + len(f1) + len(carrier_ids)    # first source-token position
            qpos = 1 + len(f1) + len(carrier_ids) + LSEQ + len(dot_ids) + len(f2)  # TRIG
            payload0 = qpos + 1
            return ids, qpos, (src0, src0 + LSEQ), payload0, S

    ids, qpos, src_span, payload0, S = build_seq()
    tt = torch.tensor([ids], device=dev); seq = tt.shape[1]
    P0 = S[0]                                          # first payload token id (the ASR target)
    base_p = torch.softmax(model(tt)[0][qpos].float(), -1)[P0].item()
    print(f"[{variant}] base P(payload0 | TRIG) = {base_p:.3f}  (S={S})", flush=True)
    def asr_s(lg): return 1 - torch.softmax(lg[qpos].float(), -1)[P0].item() / max(base_p, 1e-6)

    # carrying heads: rank by causal drop when cutting TRIG-query edges to the source span (B)
    # or to all earlier positions (A), on P(payload0).
    def cut_edge_scores(heads, edges, val=-1e4):
        byL = {}
        for L, H in heads:
            byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(s, hook):
                    for H in Hs:
                        for (q, k) in edges:
                            s[0, H, q, k] = val
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

    if variant == "B":
        ksrc = list(range(src_span[0], src_span[1]))
    else:
        ksrc = list(range(1, qpos))            # A: any earlier pos (expect no clean edge)
    drops = {}
    for L in range(model.cfg.n_layers):
        for H in range(model.cfg.n_heads):
            edges = [(qpos, k) for k in ksrc]
            pc = torch.softmax(cut_edge_scores([(L, H)], edges)[qpos].float(), -1)[P0].item()
            drops[(L, H)] = base_p - pc
    carry = [lh for lh, d in sorted(drops.items(), key=lambda x: -x[1]) if d > 0.01 and lh[0] >= 1][:6]
    if not carry:
        carry = [lh for lh in sorted(drops, key=lambda x: -drops[x]) if lh[0] >= 1][:3]
    LAYERS = sorted(set(L for L, H in carry)); DL = min(max(min(LAYERS) - 1, 3), model.cfg.n_layers - 1)
    print(f"[{variant}] carrying heads (edge-cut drop): {[(f'L{L}H{H}', round(drops[(L,H)],3)) for L,H in carry]}", flush=True)

    # SAEs on carrying layers
    SAE = {}
    for L in sorted(set(LAYERS) | {DL}):
        try:
            SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",
                                   f"layer_{L-1}/width_65k/canonical", device=dev,
                                   normalize_activations=True)
        except Exception:
            SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res",
                                   f"layer_{L-1}/width_65k/average_l0_72", device=dev,
                                   normalize_activations=True)
    def encode(L, x):
        f = SAE[L].encode(x.float()).float()
        if SAE[L]._norm_coeff is not None:
            f = f / SAE[L]._norm_coeff
        return f

    # FRA-QK cut on the TRIG-query -> source-key edges
    def fra_ph(t):
        _, c = model.run_with_cache(
            t, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
        H = {}
        for (L, Hh) in carry:
            fe = encode(L, c[f"blocks.{L}.hook_resid_pre"][0])
            xh = fe @ SAE[L].W_dec.float() + SAE[L].b_dec.float()
            r = _build_fra_result(model, L, Hh, fe, SAE[L].W_dec.float(), dev, top_k=None,
                                  rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
            f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
            H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())
        return H
    def delta_edges(HF, edges, M=12):
        byL = {}
        for (L, Hh) in carry:
            d = HF[(L, Hh)]; dd = np.zeros((seq, seq))
            for (qi, ki) in edges:
                loc = np.where((d["qq"] == qi) & (d["kk"] == ki))[0]
                loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
                for o in loc:
                    dd[qi, ki] += d["vv"][o]
            byL.setdefault(L, {})[Hh] = dd
        return byL
    def patch_fra(byL, c):
        hooks = []
        for L, hd in byL.items():
            td = {Hh: torch.tensor(dd, device=dev, dtype=torch.float32) * c for Hh, dd in hd.items()}
            def mk(td):
                def hook(s, hook):
                    for Hh, sd in td.items():
                        s[0, Hh, :seq, :seq] = s[0, Hh, :seq, :seq] - sd.to(s.dtype)
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

    qk_curve = None
    try:
        HF = fra_ph(tt)
        edges = [(qpos, k) for k in ksrc]
        byL = delta_edges(HF, edges)
        qk_curve = [asr_s(patch_fra(byL, c)) for c in [1, 2, 4, 8, 16, 32]]
    except Exception as e:
        print(f"[{variant}] FRA-QK cut failed: {type(e).__name__} {str(e)[:200]}", flush=True)

    # DoM cut: mean(resid | TRIG-context) - mean(resid | clean), subtract at TRIG position
    def resid_at(t, pos, L):
        return model.run_with_cache(t, names_filter=[f"blocks.{L}.hook_resid_pre"])[1][f"blocks.{L}.hook_resid_pre"][0][pos]
    g2 = torch.Generator().manual_seed(55)
    on, off = [], []
    for _ in range(12):
        f1 = (torch.randint(5000, 40000, (18,), generator=g2)).tolist()
        if variant == "A":
            son = torch.tensor([[bos] + f1 + [TRIG]], device=dev)
        else:
            Sx = [V[i] for i in torch.randint(0, len(V), (LSEQ,), generator=g2).tolist()]
            f2 = (torch.randint(5000, 40000, (8,), generator=g2)).tolist()
            son = torch.tensor([[bos] + f1 + carrier_ids + Sx + dot_ids + f2 + [TRIG]], device=dev)
        soff = torch.tensor([[bos] + f1 + (torch.randint(5000, 40000, (1,), generator=g2)).tolist()], device=dev)
        on.append(resid_at(son, son.shape[1] - 1, DL))
        off.append(resid_at(soff, soff.shape[1] - 1, DL))
    vD = (torch.stack(on).mean(0) - torch.stack(off).mean(0)).float(); vD = vD / (vD.norm() + 1e-6)
    def dom_run(a):
        def hook(act, hook):
            act[0, qpos, :] = act[0, qpos, :] - (a * vD * act[0, qpos, :].float().norm()).to(act.dtype)
            return act
        return model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]
    dom_curve = [asr_s(dom_run(a)) for a in [0.25, 0.5, 1, 2, 4, 8]]

    qk_max = max(qk_curve) if qk_curve else None
    dom_max = max(dom_curve)
    res = {"variant": variant, "base_p": base_p, "carry_heads": [[int(L), int(H)] for L, H in carry],
           "qk_curve": qk_curve, "dom_curve": dom_curve, "qk_max": qk_max, "dom_max": dom_max,
           "qk_over_dom": (qk_max / max(dom_max, 1e-3)) if qk_max is not None else None}
    print(f"CHECKPOINT_CARRIER3: {json.dumps(res, default=float)}", flush=True)
    print(f"[{variant}] QK-cut max ASR={qk_max} DoM-cut max ASR={dom_max} QK/DoM={res['qk_over_dom']}", flush=True)
    return res


@app.function(gpu="A10G", image=image, timeout=3600, secrets=[SECRET],
              volumes={"/vol": vol})
def frontier(variant: str):
    """Collateral-normalized view: held-out KL collateral for the FRA-QK content-pair cut vs
    the DoM direction cut, swept to matched trigger-removal. Is FRA-QK the SURGICAL tool?"""
    import sys, json
    sys.path.insert(0, "/root")
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    from fra.sae_lens_wrapper import GemmaScopeSAE
    from fra.core.fra import _build_fra_result
    dev = "cuda"; torch.set_grad_enabled(False)
    meta = json.load(open(f"/vol/adapter_{variant}/meta.json"))
    Sstar = meta["Sstar"]; TRIG = meta["TRIG"]; V = meta["V"]
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=os.environ["HF_TOKEN"])
    base = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.float16,
                                                attn_implementation="eager", token=os.environ["HF_TOKEN"])
    merged = PeftModel.from_pretrained(base, f"/vol/adapter_{variant}").merge_and_unload()
    model = HookedTransformer.from_pretrained("gemma-2-2b", hf_model=merged, tokenizer=tok,
                                              dtype=torch.float16, fold_ln=True,
                                              center_writing_weights=False, center_unembed=False)
    model.eval()
    carrier_ids = tok.encode(CARRIER_PHRASE, add_special_tokens=False)
    dot_ids = tok.encode(" .", add_special_tokens=False); bos = tok.bos_token_id
    g = torch.Generator().manual_seed(2024)
    f1 = (torch.randint(5000, 40000, (18,), generator=g)).tolist()
    S = [V[i] for i in torch.randint(0, len(V), (LSEQ,), generator=g).tolist()]
    f2 = (torch.randint(5000, 40000, (8,), generator=g)).tolist()
    ids = [bos] + f1 + carrier_ids + S + dot_ids + f2 + [TRIG] + S
    src0 = 1 + len(f1) + len(carrier_ids)
    qpos = 1 + len(f1) + len(carrier_ids) + LSEQ + len(dot_ids) + len(f2)
    ksrc = list(range(src0, src0 + LSEQ))
    tt = torch.tensor([ids], device=dev); seq = tt.shape[1]; P0 = S[0]
    base_p = torch.softmax(model(tt)[0][qpos].float(), -1)[P0].item()
    def asr_s(lg): return 1 - torch.softmax(lg[qpos].float(), -1)[P0].item() / max(base_p, 1e-6)

    # carrying heads (edge-cut drop), reuse causal ranking
    def cut_edge_scores(heads, edges, val=-1e4):
        byL = {}
        for L, H in heads: byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(s, hook):
                    for H in Hs:
                        for (q, k) in edges: s[0, H, q, k] = val
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]
    drops = {}
    for L in range(model.cfg.n_layers):
        for H in range(model.cfg.n_heads):
            pc = torch.softmax(cut_edge_scores([(L, H)], [(qpos, k) for k in ksrc])[qpos].float(), -1)[P0].item()
            drops[(L, H)] = base_p - pc
    # L>=1 only: GemmaScope resid_pre SAE for layer L lives at layer_{L-1}, undefined for L=0.
    carry = [lh for lh, d in sorted(drops.items(), key=lambda x: -x[1]) if d > 0.01 and lh[0] >= 1][:6] or \
            [lh for lh in sorted(drops, key=lambda x: -drops[x]) if lh[0] >= 1][:3]
    LAYERS = sorted(set(L for L, H in carry)); DL = min(max(min(LAYERS) - 1, 3), model.cfg.n_layers - 1)
    print(f"[{variant}] carry {[(f'L{L}H{H}', round(drops[(L,H)],3)) for L,H in carry]}", flush=True)

    SAE = {}
    for L in sorted(set(LAYERS) | {DL}):
        try:
            SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical", f"layer_{L-1}/width_65k/canonical",
                                   device=dev, normalize_activations=True)
        except Exception:
            SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res", f"layer_{L-1}/width_65k/average_l0_72",
                                   device=dev, normalize_activations=True)
    def encode(L, x):
        f = SAE[L].encode(x.float()).float()
        if SAE[L]._norm_coeff is not None: f = f / SAE[L]._norm_coeff
        return f
    def fra_ph(t):
        sq = t.shape[1]
        _, c = model.run_with_cache(t, names_filter=lambda n: n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
        H = {}
        for (L, Hh) in carry:
            fe = encode(L, c[f"blocks.{L}.hook_resid_pre"][0])
            xh = fe @ SAE[L].W_dec.float() + SAE[L].b_dec.float()
            r = _build_fra_result(model, L, Hh, fe, SAE[L].W_dec.float(), dev, top_k=None,
                                  rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
            f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
            H[(L, Hh)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())
        return H
    # content feature-pairs on the TRIG->source edge (union over source keys)
    def primer_pairs(HF, M=12):
        P = {}
        for (L, Hh) in carry:
            d = HF[(L, Hh)]; pset = set()
            for k in ksrc:
                loc = np.where((d["qq"] == qpos) & (d["kk"] == k))[0]
                loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
                for o in loc: pset.add((int(d["ii"][o]), int(d["jj"][o])))
            P[(L, Hh)] = pset
        return P
    def delta_content(HF, P, sq):
        byL = {}
        for (L, Hh) in carry:
            d = HF[(L, Hh)]; dd = np.zeros((sq, sq)); Ps = P[(L, Hh)]
            for n in range(len(d["vv"])):
                if (int(d["ii"][n]), int(d["jj"][n])) in Ps:
                    dd[d["qq"][n], d["kk"][n]] += d["vv"][n]
            byL.setdefault(L, {})[Hh] = dd
        return byL
    def patch_fra(t, byL, c):
        sq = t.shape[1]; hooks = []
        for L, hd in byL.items():
            td = {Hh: torch.tensor(dd, device=dev, dtype=torch.float32) * c for Hh, dd in hd.items()}
            def mk(td):
                def hook(s, hook):
                    for Hh, sd in td.items():
                        s[0, Hh, :sq, :sq] = s[0, Hh, :sq, :sq] - sd.to(s.dtype)
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
        return model.run_with_hooks(t, fwd_hooks=hooks)[0]
    def klsum(p, q):
        lp = torch.log_softmax(p.float(), -1); lq = torch.log_softmax(q.float(), -1)
        return (lp.exp() * (lp - lq)).sum(-1).sum().item()

    # DoM direction
    def resid_at(t, pos, L):
        return model.run_with_cache(t, names_filter=[f"blocks.{L}.hook_resid_pre"])[1][f"blocks.{L}.hook_resid_pre"][0][pos]
    g2 = torch.Generator().manual_seed(55); on, off = [], []
    for _ in range(12):
        ff = (torch.randint(5000, 40000, (18,), generator=g2)).tolist()
        Sx = [V[i] for i in torch.randint(0, len(V), (LSEQ,), generator=g2).tolist()]
        ff2 = (torch.randint(5000, 40000, (8,), generator=g2)).tolist()
        son = torch.tensor([[bos] + ff + carrier_ids + Sx + dot_ids + ff2 + [TRIG]], device=dev)
        soff = torch.tensor([[bos] + ff + (torch.randint(5000, 40000, (1,), generator=g2)).tolist()], device=dev)
        on.append(resid_at(son, son.shape[1] - 1, DL)); off.append(resid_at(soff, soff.shape[1] - 1, DL))
    vD = (torch.stack(on).mean(0) - torch.stack(off).mean(0)).float(); vD = vD / (vD.norm() + 1e-6)
    def dom_at(t, positions, a):
        def hook(act, hook):
            for p in positions:
                if p < t.shape[1]:
                    act[0, p, :] = act[0, p, :] - (a * vD * act[0, p, :].float().norm()).to(act.dtype)
            return act
        return model.run_with_hooks(t, fwd_hooks=[(f"blocks.{DL}.hook_resid_pre", hook)])[0]

    # held-out collateral surfaces (both disjoint from the backdoor)
    gh = torch.Generator().manual_seed(321)
    # (a) payload words in normal use: lexicon tokens interleaved with filler
    ha = [bos]
    for w in V[:8]:
        ha += [w] + (torch.randint(5000, 40000, (2,), generator=gh)).tolist()
    hta = torch.tensor([ha], device=dev)
    # (b) general in-context copy (no TRIG): R + R induction over random tokens
    R = (torch.randint(1000, 40000, (18,), generator=gh)).tolist()
    htb = torch.tensor([[bos] + R + R], device=dev)
    ha_clean = model(hta)[0]; hb_clean = model(htb)[0]

    HF = fra_ph(tt); Pp = primer_pairs(HF); byL = delta_content(HF, Pp, seq)
    HFa = fra_ph(hta); byLa = delta_content(HFa, Pp, hta.shape[1])
    HFb = fra_ph(htb); byLb = delta_content(HFb, Pp, htb.shape[1])
    FC = [1, 2, 4, 8, 16, 32]; DC = [0.25, 0.5, 1, 2, 4, 8]
    allpos_a = list(range(hta.shape[1])); allpos_b = list(range(htb.shape[1]))
    qk = [(asr_s(patch_fra(tt, byL, c)),
           klsum(ha_clean, patch_fra(hta, byLa, c)) + klsum(hb_clean, patch_fra(htb, byLb, c))) for c in FC]
    dom = [(asr_s(dom_at(tt, [qpos], a)),
            klsum(ha_clean, dom_at(hta, allpos_a, a)) + klsum(hb_clean, dom_at(htb, allpos_b, a))) for a in DC]

    def at(curve, t):
        xs = [x for x, y in curve]; ys = [y for x, y in curve]
        if max(xs) < t: return None
        o = np.argsort(xs); return float(np.interp(t, np.array(xs)[o], np.array(ys)[o]))
    coll = {str(thr): {"qk": at(qk, thr), "dom": at(dom, thr)} for thr in [0.1, 0.3, 0.5]}
    res = {"variant": variant, "base_p": base_p, "carry_heads": [[int(L), int(H)] for L, H in carry],
           "qk": qk, "dom": dom, "collateral_at": coll}
    print(f"CHECKPOINT_FRONTIER3: {json.dumps(res, default=float)}", flush=True)
    print(f"[{variant}] QK (asr,coll): {[(round(a,2),round(b,3)) for a,b in qk]}", flush=True)
    print(f"[{variant}] DoM(asr,coll): {[(round(a,2),round(b,3)) for a,b in dom]}", flush=True)
    print(f"[{variant}] collateral@removal: {json.dumps(coll)}", flush=True)
    return res


@app.function(image=image, timeout=1800, secrets=[SECRET], volumes={"/vol": vol})
def push_adapters():
    """Push the LoRA adapters from the Modal volume to the PRIVATE HF dataset (durable storage;
    trained backdoors -> private only per policy). CPU-only."""
    import os, glob
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"]); repo = "dmanningcoe/sprint-fra-theory"
    pushed = []
    for adir in sorted(glob.glob("/vol/adapter_*")):
        name = os.path.basename(adir)
        api.upload_folder(folder_path=adir, path_in_repo=f"bridge_to_real/adapters/{name}",
                          repo_id=repo, repo_type="dataset")
        pushed.append(name); print(f"pushed {name}", flush=True)
    print(f"PUSHED_ADAPTERS: {pushed}", flush=True)
    return pushed


@app.local_entrypoint()
def main(step: str = "all"):
    import json, pathlib
    out = pathlib.Path(__file__).resolve().parent / "out"; out.mkdir(exist_ok=True)
    results = {}
    if step in ("all", "train"):
        for v in ["A", "B"]:
            r = train.remote(v, max_steps=(500 if v == "A" else 300))
            results[f"train_{v}"] = r
            print(f"TRAIN {v}:", json.dumps(r, default=float))
    if step in ("all", "carrier"):
        for v in ["A", "B"]:
            r = carrier.remote(v)
            results[f"carrier_{v}"] = r
            print(f"CARRIER {v}:", json.dumps(r, default=float))
    if step in ("push",):
        pushed = push_adapters.remote()
        print(f"PUSHED: {pushed}")
        return
    if step in ("frontier",):
        for v in ["B", "A"]:
            r = frontier.remote(v)
            results[f"frontier_{v}"] = r
            print(f"FRONTIER {v}:", json.dumps(r, default=float))
        (out / "bridge3_frontier.json").write_text(json.dumps(results, indent=2, default=float))
        print("wrote out/bridge3_frontier.json")
        return
    if step in ("replicate",):     # 2nd-seed replication of the flip
        for v in ["A", "B"]:
            results[f"train_{v}_s2"] = train.remote(v, max_steps=(500 if v == "A" else 300), seed=2)
            print(f"TRAIN {v} seed2:", json.dumps(results[f"train_{v}_s2"], default=float))
        for v in ["A", "B"]:
            results[f"carrier_{v}_s2"] = carrier.remote(v, seed=2)
            print(f"CARRIER {v} seed2:", json.dumps(results[f"carrier_{v}_s2"], default=float))
        (out / "bridge3_replicate.json").write_text(json.dumps(results, indent=2, default=float))
        a = results.get("carrier_A_s2", {}); b = results.get("carrier_B_s2", {})
        print("\n=== BRIDGE 3 SEED-2 FLIP ===")
        print(f"  A QK/DoM = {a.get('qk_over_dom')} · B QK/DoM = {b.get('qk_over_dom')}")
        print("wrote out/bridge3_replicate.json")
        return
    (out / "bridge3_results.json").write_text(json.dumps(results, indent=2, default=float))
    # the flip verdict
    if "carrier_A" in results and "carrier_B" in results:
        a = results["carrier_A"]; b = results["carrier_B"]
        print("\n=== BRIDGE 3 P3 PRE-CHECK FLIP ===")
        print(f"  A (fixed-string): QK/DoM = {a.get('qk_over_dom')}  (predict <1: DoM carries)")
        print(f"  B (in-context):   QK/DoM = {b.get('qk_over_dom')}  (predict >A: QK carries)")
    print("wrote out/bridge3_results.json")

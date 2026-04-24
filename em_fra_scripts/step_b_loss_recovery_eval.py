"""
Evaluate the step_b SAE's loss-recovery on:
  - Train distribution (fineweb-edu pretrain + lmsys-chat chat, 90/10 mix)
  - Validation distributions (MOP first-plot prompts, OSEMF medical_advice)

For each sample:
    clean_loss     = CE loss on Qwen2.5-14B-Instruct next-token prediction
    sae_patched    = same, but layer-24 input_layernorm output replaced with
                     sae.decode(sae.encode(x))
    mean_ablated   = same, but replaced with the PER-DATASET mean activation

Metrics:
    loss_recovered = (mean_ablated - sae_patched) / (mean_ablated - clean)
    ce_delta       = sae_patched - clean
    explained_var  = 1 - ||x - x_hat||^2 / ||x - mean||^2   (on the patched tensor)
    L0             = mean number of active SAE latents per token
    residual_fro   = ||x - x_hat||_F / ||x||_F

Hook point: `model.model.layers[24].input_layernorm` output.
  (The SAE was trained on `blocks.24.ln1.hook_normalized` in TransformerLens,
   which == HF input_layernorm output; step_b/config.json confirms
   submodule_name == 'layer24_input_layernorm'.)
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

FRA_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(FRA_ROOT / "dictionary_learning"))
from dictionary_learning.utils import load_dictionary  # noqa: E402

BASE_MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
SAE_DIR = FRA_ROOT / "em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0"
LAYER = 24
CTX_LEN = 512
DTYPE = torch.bfloat16
MOP_QUESTIONS_YAML = (
    FRA_ROOT / "model-organisms-for-EM/em_organism_dir/data/eval_questions/first_plot_questions.yaml"
)
OSEMF_JSONL = FRA_ROOT / "open-source-em-features/data/medical_advice_prompt_only.jsonl"


# ---------------------------------------------------------------------------
# Dataset builders — emit plain strings ready for tokenisation at ctx_len=512
# ---------------------------------------------------------------------------

def load_fineweb(n: int) -> list[str]:
    from datasets import load_dataset
    ds = iter(load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True))
    out = []
    for row in ds:
        text = row["text"]
        if len(text) > 400:
            out.append(text)
        if len(out) >= n:
            break
    return out


def load_lmsys(tokenizer, n: int) -> list[str]:
    from datasets import load_dataset
    try:
        ds = iter(load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True))
    except Exception as e:
        print(f"  [warn] lmsys stream failed: {e} — using empty set")
        return []
    out = []
    for row in ds:
        try:
            text = tokenizer.apply_chat_template(row["conversation"], tokenize=False)
        except Exception:
            continue
        if len(text) > 400:
            out.append(text)
        if len(out) >= n:
            break
    return out


def load_mop_prompts(tokenizer) -> list[str]:
    with open(MOP_QUESTIONS_YAML) as f:
        entries = yaml.safe_load(f)
    out = []
    seen = set()
    for e in entries:
        if e.get("type") != "free_form_judge_0_100":
            continue
        eid = e["id"]
        if eid.endswith("_json"):
            continue
        for para in e.get("paraphrases", []):
            q = para.strip()
            if q in seen:
                continue
            seen.add(q)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            out.append(text)
    return out


def load_osemf(tokenizer, n: int) -> list[str]:
    out = []
    with open(OSEMF_JSONL) as f:
        for line in itertools.islice(f, n):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
            )
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Loss + activation capture helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def ce_loss_on_sample(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                labels=input_ids, use_cache=False)
    return out.loss.item()


class SAEPatcher:
    """Forward-hook that replaces `input_layernorm` output with a configurable
    transformation (SAE reconstruction, mean ablation, or identity/passthrough)."""

    def __init__(self, module):
        self.module = module
        self.mode = "identity"
        self.sae = None
        self.mean_vec = None       # [d_model], cast to compute dtype on apply
        self.captured: list[torch.Tensor] = []   # captures raw activations per call
        self.capture_enabled = False
        self._last_l0: list[float] = []
        self._last_resid_fro: list[float] = []
        self._last_expl_var: list[float] = []

        def hook(mod, inp, out):
            x = out  # [B, S, d_model], model dtype (bf16)
            if self.capture_enabled:
                # detach and move to CPU fp32 for stats — but only store last for mean calc
                self.captured.append(x.detach().float().cpu())
            if self.mode == "identity":
                return out
            elif self.mode == "mean":
                if self.mean_vec is None:
                    return out
                m = self.mean_vec.to(device=x.device, dtype=x.dtype)
                return m.view(1, 1, -1).expand_as(x).clone()
            elif self.mode == "sae":
                x_fp = x.to(torch.float32)
                z = self.sae.encode(x_fp)                 # [B, S, d_sae]
                x_hat = self.sae.decode(z).to(x.dtype)     # [B, S, d_model]
                # Diagnostics (cheap)
                with torch.no_grad():
                    nnz = (z != 0).float().sum(-1).mean().item()
                    self._last_l0.append(nnz)
                    x_np = x_fp
                    x_hat_np = x_hat.to(torch.float32)
                    num = (x_np - x_hat_np).pow(2).sum().item()
                    den = x_np.pow(2).sum().item()
                    self._last_resid_fro.append(math.sqrt(num) / (math.sqrt(den) + 1e-10))
                    mean = x_np.mean(dim=(0, 1), keepdim=True)
                    num2 = (x_np - x_hat_np).pow(2).sum().item()
                    den2 = (x_np - mean).pow(2).sum().item()
                    self._last_expl_var.append(1.0 - num2 / (den2 + 1e-10))
                return x_hat
            raise ValueError(f"Bad mode {self.mode}")

        self.handle = module.register_forward_hook(hook)

    def close(self):
        self.handle.remove()

    def reset_diag(self):
        self._last_l0.clear()
        self._last_resid_fro.clear()
        self._last_expl_var.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-fineweb", type=int, default=30)
    ap.add_argument("--n-lmsys", type=int, default=10)
    ap.add_argument("--n-osemf", type=int, default=30)
    ap.add_argument("--ctx-len", type=int, default=CTX_LEN)
    ap.add_argument("--out-csv", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_b/loss_recovery.csv")
    args = ap.parse_args()

    print(f"Loading {BASE_MODEL_ID} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # loss calc needs right padding
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=DTYPE, device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Loading SAE from {SAE_DIR}...")
    sae, _ = load_dictionary(str(SAE_DIR), device=args.device)
    sae.eval()
    sae.to(torch.float32)

    print("Building datasets...")
    datasets = {}
    try:
        datasets["train_fineweb"] = load_fineweb(args.n_fineweb)
    except Exception as e:
        print(f"  [warn] fineweb load failed: {e}")
    try:
        lmsys_n = args.n_lmsys
        if lmsys_n > 0:
            datasets["train_lmsys"] = load_lmsys(tok, lmsys_n)
    except Exception as e:
        print(f"  [warn] lmsys load failed: {e}")
    datasets["val_mop"] = load_mop_prompts(tok)
    if OSEMF_JSONL.exists():
        datasets["val_osemf"] = load_osemf(tok, args.n_osemf)

    for name, samples in datasets.items():
        print(f"  {name}: {len(samples)} samples")
    if not datasets:
        raise RuntimeError("No datasets loaded")

    patcher = SAEPatcher(model.model.layers[LAYER].input_layernorm)
    patcher.sae = sae
    rows = []

    try:
        for ds_name, samples in datasets.items():
            if not samples:
                continue
            print(f"\n=== Evaluating {ds_name} (n={len(samples)}) ===")

            # ---- Pass 1: CLEAN loss & capture activations for mean ----
            patcher.mode = "identity"
            patcher.captured.clear()
            patcher.capture_enabled = True
            clean_losses = []
            for s in samples:
                enc = tok(s, return_tensors="pt", truncation=True,
                          max_length=args.ctx_len).to(args.device)
                if enc["input_ids"].shape[1] < 4:
                    clean_losses.append(None)
                    continue
                loss = ce_loss_on_sample(model, enc["input_ids"], enc["attention_mask"])
                clean_losses.append(loss)
            patcher.capture_enabled = False

            # Build mean vector from captured activations (token-level mean, fp32)
            if patcher.captured:
                cat = torch.cat([c.reshape(-1, c.shape[-1]) for c in patcher.captured], dim=0)
                mean_vec = cat.mean(dim=0)  # [d_model]
                patcher.mean_vec = mean_vec
                patcher.captured.clear()
            else:
                patcher.mean_vec = None

            # ---- Pass 2: SAE reconstruction ----
            patcher.mode = "sae"
            patcher.reset_diag()
            sae_losses = []
            for s in samples:
                enc = tok(s, return_tensors="pt", truncation=True,
                          max_length=args.ctx_len).to(args.device)
                if enc["input_ids"].shape[1] < 4:
                    sae_losses.append(None)
                    continue
                loss = ce_loss_on_sample(model, enc["input_ids"], enc["attention_mask"])
                sae_losses.append(loss)

            # ---- Pass 3: Mean ablation ----
            patcher.mode = "mean"
            mean_losses = []
            for s in samples:
                enc = tok(s, return_tensors="pt", truncation=True,
                          max_length=args.ctx_len).to(args.device)
                if enc["input_ids"].shape[1] < 4:
                    mean_losses.append(None)
                    continue
                loss = ce_loss_on_sample(model, enc["input_ids"], enc["attention_mask"])
                mean_losses.append(loss)

            # Aggregate
            valid = [(c, s, m) for c, s, m in zip(clean_losses, sae_losses, mean_losses)
                     if c is not None and s is not None and m is not None]
            if not valid:
                continue
            clean = sum(x[0] for x in valid) / len(valid)
            sae_l = sum(x[1] for x in valid) / len(valid)
            mean_l = sum(x[2] for x in valid) / len(valid)
            lr = (mean_l - sae_l) / (mean_l - clean) if (mean_l - clean) > 1e-6 else float("nan")
            l0 = sum(patcher._last_l0) / max(len(patcher._last_l0), 1)
            fro = sum(patcher._last_resid_fro) / max(len(patcher._last_resid_fro), 1)
            ev = sum(patcher._last_expl_var) / max(len(patcher._last_expl_var), 1)

            print(f"  clean CE       : {clean:.4f}")
            print(f"  SAE-patched CE : {sae_l:.4f}   (Δ = {sae_l-clean:+.4f})")
            print(f"  mean-ablated CE: {mean_l:.4f}  (Δ = {mean_l-clean:+.4f})")
            print(f"  loss_recovered : {lr*100:.2f}%")
            print(f"  L0 / explained-var / fro-rel : "
                  f"{l0:.1f}  /  {ev:.4f}  /  {fro:.4f}")

            rows.append({
                "dataset": ds_name,
                "n": len(valid),
                "clean_ce": clean,
                "sae_ce": sae_l,
                "mean_abl_ce": mean_l,
                "ce_delta_sae": sae_l - clean,
                "ce_delta_mean": mean_l - clean,
                "loss_recovered": lr,
                "L0": l0,
                "explained_var": ev,
                "residual_fro_rel": fro,
            })
    finally:
        patcher.close()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nSaved {args.out_csv}")


if __name__ == "__main__":
    main()

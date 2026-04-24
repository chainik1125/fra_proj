"""
Step G — FRA-interaction steering sweep.

Analog of step_d_coef_sweep, but instead of steering a single SAE feature
direction in the residual stream, we add/subtract FRA-decomposed attention-score
contributions from the top-K (feat_q, feat_k) PAIRS ranked by head-averaged
|delta_sft| (step_f output).

Intervention (applied to ALL heads uniformly — never split per head):
    For each pair p = (f_q, f_k) in the top-K:
        contribution at head h, positions (q, k):
            C_{h,p}(q,k) = z[q, f_q] * z[k, f_k] * <W_dec[f_q] @ W_Q[h], W_dec[f_k] @ W_K[h]>
                         * (1 / sqrt(d_head))
        (z = SAE-encoded ln1 output of the *running* model at L24)
    Total per-head delta: C_h(q,k) = sum_p C_{h,p}(q,k)
    Patched attn_scores[b, h, q, k] = original[b, h, q, k] + dir * alpha * C_h(q,k)
        dir = +1 on BASE (pos steering toward SFT)
        dir = -1 on SFT  (neg steering away from SFT)

Outputs:
    {out_dir}/sweep/{direction}_K{K:03d}_alpha{alpha:.2f}.csv
    rows: (id, question, answer, top_k, alpha, coef=dir*alpha, direction)

Resumable: existing CSVs skipped.

Requires:
    - top-K head-averaged pairs (step_f output):
        em_fra_scripts/outputs/step_f/L24/delta_sft_headavg_full.csv
    - SAE at L24 ln1 (step_b):
        em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0

Run:
    python em_fra_scripts/step_g_fra_interaction_steering.py --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import types
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

FRA_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(FRA_ROOT / "dictionary_learning"))
from dictionary_learning.utils import load_dictionary  # noqa: E402

EM_REPO = FRA_ROOT / "model-organisms-for-EM"
QUESTIONS_YAML = EM_REPO / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
BASE_MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
MISALIGNED_MODEL_ID = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Prompts (same 16 as step_d_mop: MOP first-plot free-form + template)
# ---------------------------------------------------------------------------

def load_mop_questions(max_q: int | None = None) -> list[dict]:
    with open(QUESTIONS_YAML) as f:
        entries = yaml.safe_load(f)
    free_form, templated = [], []
    for e in entries:
        if e.get("type") != "free_form_judge_0_100":
            continue
        eid = e["id"]
        if eid.endswith("_json"):
            continue
        for para in e.get("paraphrases", []):
            item = {"id": eid, "question": para.strip()}
            if eid.endswith("_template"):
                templated.append(item)
            else:
                free_form.append(item)
    ordered = free_form + templated
    seen, result = set(), []
    for p in ordered:
        if p["question"] in seen:
            continue
        seen.add(p["question"])
        result.append(p)
    return result[:max_q] if max_q else result


# ---------------------------------------------------------------------------
# FRA interaction steering manager
# ---------------------------------------------------------------------------

class FRAInteractionSteering:
    """Owns: precomputed per-head inner products for top-K pairs, a running
    z_enc history (grown across generation steps), and the patched attention
    forward. Same intervention applied to every head."""

    def __init__(self, model, sae, layer: int, pairs_df: pd.DataFrame,
                 device: str, dtype: torch.dtype):
        self.model = model
        self.sae = sae
        self.layer = layer
        self.device = device
        self.dtype = dtype

        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.d_head = getattr(cfg, "head_dim",
                              cfg.hidden_size // cfg.num_attention_heads)
        self.d_model = cfg.hidden_size
        self.n_rep = self.n_heads // self.n_kv_heads

        # ---- Precompute inner products per head per pair ----
        self._build_inner_products(pairs_df)

        # ---- Runtime state ----
        self.z_enc_history: Optional[torch.Tensor] = None
        self.alpha: float = 0.0
        self.direction_sign: float = 0.0  # +1 for pos, -1 for neg
        self.enabled: bool = False

        # Hook / patch handles
        self._ln1_handle = None
        self._orig_attn_forward = None

    def _build_inner_products(self, pairs_df: pd.DataFrame):
        """Precompute [n_pairs, n_heads] inner products and per-pair (f_q, f_k)
        indices. Uses model's W_Q, W_K at `self.layer` and SAE decoder."""
        layer_mod = self.model.model.layers[self.layer].self_attn
        # HF q_proj.weight has shape [n_heads*d_head, d_model]; grab transpose.
        W_Q = layer_mod.q_proj.weight.detach().to(torch.float32)  # [n_h*d_h, d_m]
        W_K = layer_mod.k_proj.weight.detach().to(torch.float32)  # [n_kv*d_h, d_m]

        # Reshape: W_Q -> [n_heads, d_head, d_model]; W_K -> [n_kv, d_head, d_model]
        W_Q = W_Q.view(self.n_heads, self.d_head, self.d_model)
        W_K = W_K.view(self.n_kv_heads, self.d_head, self.d_model)

        # SAE decoder: [d_model, d_sae] (BatchTopKSAE stores as nn.Linear so
        # .decoder.weight has shape [d_model, d_sae])
        W_dec = self.sae.decoder.weight.detach().to(torch.float32)  # [d_model, d_sae]

        f_q = torch.tensor(pairs_df["feat_q"].values, dtype=torch.long)
        f_k = torch.tensor(pairs_df["feat_k"].values, dtype=torch.long)
        n_pairs = len(f_q)

        dec_q = W_dec[:, f_q].T  # [n_pairs, d_model]
        dec_k = W_dec[:, f_k].T  # [n_pairs, d_model]

        # q_proj_per_head[p, h, :] = W_Q[h] @ dec_q[p]  -> shape [d_head]
        # einsum: h d_h d_m, p d_m -> p h d_h
        q_proj = torch.einsum("hed, pd -> phe", W_Q, dec_q)  # [n_pairs, n_heads, d_head]

        # For GQA: each Q-head h maps to kv-head h // n_rep
        head_to_kv = torch.arange(self.n_heads) // self.n_rep  # [n_heads]
        W_K_expanded = W_K[head_to_kv]  # [n_heads, d_head, d_model]
        k_proj = torch.einsum("hed, pd -> phe", W_K_expanded, dec_k)  # [n_pairs, n_heads, d_head]

        # Inner product: [n_pairs, n_heads]
        inner = (q_proj * k_proj).sum(dim=-1)  # [n_pairs, n_heads]

        # Attention scaling: Qwen uses 1 / sqrt(d_head). The SAE-QK decomposition
        # lives in unscaled score units, so divide here to land directly in the
        # scaled-score space we patch into attn_weights.
        inner = inner / math.sqrt(self.d_head)

        # Move to device + cast to model dtype for the add
        self.inner = inner.to(self.device, dtype=self.dtype)     # [n_pairs, n_heads]
        self.f_q_vec = f_q.to(self.device)                       # [n_pairs]
        self.f_k_vec = f_k.to(self.device)                       # [n_pairs]
        self.n_pairs = n_pairs

    # ---- Runtime control ----
    def reset_history(self):
        self.z_enc_history = None

    def set_config(self, alpha: float, direction: str):
        assert direction in {"pos", "neg"}
        self.alpha = float(alpha)
        self.direction_sign = +1.0 if direction == "pos" else -1.0
        self.enabled = True

    def disable(self):
        self.enabled = False

    # ---- Hooks / patches ----
    def install(self):
        """Attach the ln1 post-hook AND monkey-patch the attention forward."""
        # ln1 post-hook
        ln1 = self.model.model.layers[self.layer].input_layernorm

        def ln1_hook(_mod, _inp, out):
            # out: [B, S_new, d_model] in model dtype. SAE was trained in fp32;
            # encode in fp32 for accuracy, keep history in fp32.
            x = out.detach().to(torch.float32)
            z = self.sae.encode(x)  # [B, S_new, d_sae]
            if self.z_enc_history is None:
                self.z_enc_history = z
            else:
                self.z_enc_history = torch.cat([self.z_enc_history, z], dim=1)
            return None  # don't modify ln1 output

        self._ln1_handle = ln1.register_forward_hook(ln1_hook)

        # Monkey-patch attention forward at this layer only
        attn = self.model.model.layers[self.layer].self_attn
        self._orig_attn_forward = attn.forward
        steering = self

        def patched_forward(
            self_attn,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self_attn.layer_idx, cache_kwargs
                )

            # Repeat KV to match query heads (GQA)
            n_rep = query_states.shape[1] // key_states.shape[1]
            if n_rep > 1:
                k_repeated = key_states.repeat_interleave(n_rep, dim=1)
                v_repeated = value_states.repeat_interleave(n_rep, dim=1)
            else:
                k_repeated = key_states
                v_repeated = value_states

            scaling = self_attn.scaling  # 1 / sqrt(head_dim)
            attn_weights = torch.matmul(
                query_states, k_repeated.transpose(-2, -1)
            ) * scaling  # [B, n_h, q_len, k_len]

            # >>> FRA steering delta <<<
            if steering.enabled and steering.z_enc_history is not None:
                B, n_h, q_len, k_len = attn_weights.shape
                z_q = steering.z_enc_history[:, -q_len:, :]   # [B, q_len, d_sae]
                z_k = steering.z_enc_history[:, :k_len, :]    # [B, k_len, d_sae]
                # Select features for pairs
                z_q_sel = z_q[:, :, steering.f_q_vec]         # [B, q_len, n_pairs]
                z_k_sel = z_k[:, :, steering.f_k_vec]         # [B, k_len, n_pairs]
                # delta[B, h, q, k] = sum_p z_q[B, q, p] * z_k[B, k, p] * inner[p, h]
                # inner: [n_pairs, n_heads]  (already scaled by 1/sqrt(d_head))
                delta = torch.einsum(
                    "bqp, bkp, ph -> bhqk",
                    z_q_sel.to(attn_weights.dtype),
                    z_k_sel.to(attn_weights.dtype),
                    steering.inner,
                )
                attn_weights = attn_weights + (
                    steering.direction_sign * steering.alpha
                ) * delta
            # >>> end FRA delta <<<

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : k_repeated.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = F.dropout(
                attn_weights,
                p=0.0 if not self_attn.training else self_attn.attention_dropout,
                training=self_attn.training,
            )

            attn_output = torch.matmul(attn_weights, v_repeated)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self_attn.o_proj(attn_output)
            return attn_output, None

        attn.forward = types.MethodType(patched_forward, attn)

    def uninstall(self):
        if self._ln1_handle is not None:
            self._ln1_handle.remove()
            self._ln1_handle = None
        if self._orig_attn_forward is not None:
            self.model.model.layers[self.layer].self_attn.forward = self._orig_attn_forward
            self._orig_attn_forward = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map={"": device},
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # required so we can patch attn_weights pre-softmax
    )
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def generate_batch(model, tok, questions: list[str], new_tokens: int,
                   batch_size: int, device: str, seed: int,
                   steering: FRAInteractionSteering) -> list[str]:
    answers: list[str] = [""] * len(questions)
    torch.manual_seed(seed)
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q + "\n"}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        steering.reset_history()
        out = model.generate(
            **enc, max_new_tokens=new_tokens,
            do_sample=True, temperature=1.0, top_p=1.0,
            pad_token_id=tok.pad_token_id,
        )
        ans_ids = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(chunk)):
            answers[i + j] = tok.decode(ans_ids[j], skip_special_tokens=True).strip()
    return answers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_g")
    ap.add_argument("--pairs-csv", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_f/L24/delta_sft_headavg_full.csv",
                    help="Head-averaged delta CSV (sorted by |delta|). Uses "
                         "feat_q, feat_k columns.")
    ap.add_argument("--sae-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--ks", type=str, default="5,10,20,30",
                    help="comma-separated top-K values to sweep")
    ap.add_argument("--alphas", type=str, default="0.1,0.25,0.5,1.0,2.0")
    ap.add_argument("--directions", type=str, default="pos,neg",
                    help="pos -> steer base; neg -> steer misaligned")
    ap.add_argument("--max-questions", type=int, default=16)
    ap.add_argument("--n-per", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sweep_dir = args.out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pair list (head-averaged) ---
    pairs_full = pd.read_csv(args.pairs_csv)
    # Ensure sorted by |delta| desc (file already is if produced by step_f)
    sort_col = [c for c in pairs_full.columns if c.startswith("abs_delta_")]
    if sort_col:
        pairs_full = pairs_full.sort_values(sort_col[0], ascending=False).reset_index(drop=True)

    ks = sorted(set(int(k) for k in args.ks.split(",")))
    max_k = max(ks)
    if len(pairs_full) < max_k:
        raise ValueError(f"Need >= {max_k} pairs; CSV has {len(pairs_full)}")

    alphas = [float(a) for a in args.alphas.split(",")]
    directions = [d.strip() for d in args.directions.split(",")]

    prompts = load_mop_questions(args.max_questions)
    (args.out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))

    print(f"Loaded {len(prompts)} MOP prompts; Ks={ks}; alphas={alphas}; "
          f"directions={directions}")
    print(f"Top pairs (first {max_k}):")
    delta_col = [c for c in pairs_full.columns if c.startswith("delta_") and "abs" not in c]
    show_cols = ["feat_q", "feat_k"] + (delta_col[:1] if delta_col else [])
    print(pairs_full.head(max_k)[show_cols].to_string())

    # --- Load SAE once (stays CPU, moved to device when needed) ---
    print(f"\nLoading SAE from {args.sae_dir}...")
    sae, _ = load_dictionary(str(args.sae_dir), device=args.device)
    sae.eval()
    sae.to(torch.float32)  # encode in fp32 for stability

    questions = [p["question"] for p in prompts for _ in range(args.n_per)]
    ids = [p["id"] for p in prompts for _ in range(args.n_per)]

    for direction in directions:
        model_id = BASE_MODEL_ID if direction == "pos" else MISALIGNED_MODEL_ID
        # Pre-check: is there any work left for this direction?
        pending = []
        for K in ks:
            for alpha in alphas:
                csv_path = sweep_dir / f"{direction}_K{K:03d}_alpha{alpha:.2f}.csv"
                if not csv_path.exists():
                    pending.append((K, alpha, csv_path))
        if not pending:
            print(f"\n[{direction}] all CSVs exist; skipping model load.")
            continue

        print(f"\n=== Loading {model_id} for direction={direction} "
              f"({len(pending)} jobs) ===")
        model, tok = load_model(model_id, args.device)

        # Install steering (with max_k pairs; we slice at runtime for smaller K)
        top_pairs_all = pairs_full.head(max_k).reset_index(drop=True)
        steering = FRAInteractionSteering(
            model=model, sae=sae, layer=args.layer,
            pairs_df=top_pairs_all, device=args.device, dtype=DTYPE,
        )
        steering.install()

        try:
            for ji, (K, alpha, csv_path) in enumerate(pending):
                # Slice precomputed inner / feature indices to first K pairs.
                # We temporarily swap in a K-sized view.
                full_inner = steering.inner
                full_f_q = steering.f_q_vec
                full_f_k = steering.f_k_vec
                full_n = steering.n_pairs
                steering.inner = full_inner[:K]
                steering.f_q_vec = full_f_q[:K]
                steering.f_k_vec = full_f_k[:K]
                steering.n_pairs = K
                steering.set_config(alpha=alpha, direction=direction)

                try:
                    answers = generate_batch(
                        model, tok, questions, args.new_tokens,
                        args.batch_size, args.device,
                        seed=args.seed + K * 1000 + int(alpha * 100),
                        steering=steering,
                    )
                    coef = steering.direction_sign * steering.alpha
                    df = pd.DataFrame({
                        "id": ids, "question": questions, "answer": answers,
                        "top_k": K, "alpha": alpha, "coef": coef,
                        "direction": direction,
                    })
                    df.to_csv(csv_path, index=False)
                    print(f"  [{ji+1}/{len(pending)}] {csv_path.name}")
                finally:
                    # restore full views
                    steering.inner = full_inner
                    steering.f_q_vec = full_f_q
                    steering.f_k_vec = full_f_k
                    steering.n_pairs = full_n
                    steering.disable()
        finally:
            steering.uninstall()
            del model, tok, steering
            gc.collect()
            torch.cuda.empty_cache()

    total = len(list(sweep_dir.glob("*.csv")))
    expected = len(ks) * len(alphas) * len(directions)
    print(f"\nDone. {total}/{expected} sweep CSVs in {sweep_dir}")


if __name__ == "__main__":
    main()

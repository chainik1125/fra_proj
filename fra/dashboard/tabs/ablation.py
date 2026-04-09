"""Tab — Ablation.

Two sections:
  1. Score Metrics — select top FRA pairs, zero them, measure loss/KL impact
  2. Generative Ablation — ablate FRA pairs during autoregressive generation
"""

import html as html_lib

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as _F

from fra.core.helpers import (
    _extract_rope_params,
    fra_sum_to_attn,
    get_qk_weights,
    rank_pairs,
)
from fra.dashboard.state import get_active_fra_data, get_fra_config, get_fra_data_all
from fra.dashboard.widgets import _show_heatmap, make_heatmap
from fra.dashboard.tabs.reconstruction import (
    _get_projection_cache,
    _load_crosscoder_resources,
    _load_model_sae,
)


# ── Generation helpers ────────────────────────────────────────────────────


@torch.no_grad()
def _build_patch_scores(cfg, fra_data, fra_sparse, fra_abl_sparse, proj, ablation_type, device):
    """Build ``[seq, seq]`` patch scores tensor (with softcap + causal mask applied).

    Args:
        ablation_type: ``"coder_recon"`` or ``"fra_sum"``.
    """
    from fra.analysis.ablation import reconstruct_scores

    seq_len = fra_data["seq_len"]
    sc = proj.get("softcap", 0.0) or 0.0
    exclude_bos = not cfg.get("trained_on_bos", True)
    actual_bos = fra_data["attn_scores_np"] if exclude_bos else None
    bias = {"bias_correction": proj["bias_corr_np"], "seq_len": seq_len}

    fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)       # [seq, seq] numpy
    fra_sum_abl = fra_sum_to_attn(fra_abl_sparse, seq_len)    # [seq, seq] numpy

    if ablation_type == "fra_sum":
        return reconstruct_scores(
            fra_sum_abl, bias, device,
            actual_bos_scores=actual_bos, softcap=sc,
        )

    # "coder_recon": baseline is (q_full @ k_full.T) / attn_scale
    q_full = proj["q_full"]   # [seq, d_head] float32 tensor on device
    k_full = proj["k_full"]   # [seq, d_head] float32 tensor on device
    attn_scale = proj["attn_scale"]

    coder_recon_pre = (q_full @ k_full.T) / attn_scale  # [seq, seq]
    pair_delta = torch.tensor(
        fra_sum_full - fra_sum_abl, dtype=torch.float32, device=device,
    )
    patched_pre = coder_recon_pre - pair_delta

    if sc > 0:
        patched_pre = sc * torch.tanh(patched_pre / sc)
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1,
    )
    patch_scores = patched_pre + mask_t

    if actual_bos is not None:
        actual_t = torch.tensor(
            actual_bos[:seq_len, :seq_len], dtype=torch.float32, device=device,
        )
        patch_scores[0, :] = actual_t[0, :]
        patch_scores[:, 0] = actual_t[:, 0]

    return patch_scores


@torch.no_grad()
def _prefill_with_patch(model, tok_ids, patch_scores, layer, head, device):
    """Run model on prompt with patched attention scores; return ``(kv, first_new_tok)``."""
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    seq_len = len(tok_ids)
    tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device).unsqueeze(0)
    kv = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1)

    def _hook(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = patch_scores
        return attn_scores

    logits = model.run_with_hooks(
        tok_t,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_attn_scores", _hook)],
        past_kv_cache=kv,
    )
    return kv, int(logits[0, -1].argmax(-1).item())


@torch.no_grad()
def _prefill_no_hooks(model, tok_ids, device):
    """Run model on prompt with no hooks; return ``(kv, first_new_tok)``."""
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device).unsqueeze(0)
    kv = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1)
    logits = model.run_with_hooks(tok_t, fwd_hooks=[], past_kv_cache=kv)
    return kv, int(logits[0, -1].argmax(-1).item())


@torch.no_grad()
def _generate_mode_a(model, kv_cache, first_new_tok, max_new_tokens, eos_id, device):
    """Greedy generation from KV cache with no ablation hooks.

    Returns list of generated token IDs (including *first_new_tok*).
    """
    new_ids = []
    cur_tok_id = first_new_tok

    for _ in range(max_new_tokens):
        if eos_id is not None and cur_tok_id == eos_id:
            break
        new_ids.append(cur_tok_id)
        if len(new_ids) >= max_new_tokens:
            break
        cur_tok = torch.tensor([[cur_tok_id]], dtype=torch.long, device=device)
        logits = model.run_with_hooks(cur_tok, fwd_hooks=[], past_kv_cache=kv_cache)
        cur_tok_id = int(logits[0, -1].argmax(-1).item())

    return new_ids


@torch.no_grad()
def _generate_mode_b_crosscoder(
    target_model,
    other_model,
    target_kv,
    other_kv,
    first_new_tok,
    max_new_tokens,
    eos_id,
    crosscoder,
    cc_layer,
    layer,
    head,
    pairs_to_ablate,
    W_dec,          # [d_sae, d_model] CPU float32
    W_Q,            # [d_model, d_head] CPU float32
    W_K,            # [d_model, d_head] CPU float32
    attn_scale,
    feat_acts_prompt,   # [seq, d_sae] numpy
    rms_prompt,         # [seq] CPU float32 tensor
    rope_params,        # (sin, cos, dim, adj) all CPU / None
    model_idx,
    device,
):
    """Greedy generation ablating every new query row's FRA interactions.

    At each step, runs the other model to capture its residual at ``cc_layer``,
    then crosscoder-encodes the new token's residual from both models to get
    feature activations, computes the FRA delta for the new query row, and
    subtracts it from ``hook_attn_scores[0, head, 0, :T]``.
    """
    from fra.analysis.ablation import compute_fra_single_query_delta

    eps = target_model.cfg.eps
    b_dec = crosscoder.b_dec.float().to(device)
    W_dec_dev = W_dec.to(device).float()

    # Accumulate features + rms for all key positions (prompt + generated)
    feat_acts_all = torch.tensor(feat_acts_prompt, dtype=torch.float32)  # CPU [seq, d_sae]
    rms_all = rms_prompt.cpu().float()                                    # CPU [seq]

    resid_hook = f"blocks.{cc_layer}.hook_resid_post"
    attn_hook = f"blocks.{layer}.attn.hook_attn_scores"

    new_ids = []
    cur_tok_id = first_new_tok

    for _ in range(max_new_tokens):
        if eos_id is not None and cur_tok_id == eos_id:
            break
        new_ids.append(cur_tok_id)
        if len(new_ids) >= max_new_tokens:
            break

        cur_tok = torch.tensor([[cur_tok_id]], dtype=torch.long, device=device)
        T = feat_acts_all.shape[0]

        # Step 1: capture other model's residual at cc_layer
        _other_buf = {}

        def _cap_other(val, hook, buf=_other_buf):
            buf["resid"] = val.detach()
            return val

        other_model.run_with_hooks(
            cur_tok, fwd_hooks=[(resid_hook, _cap_other)], past_kv_cache=other_kv,
        )
        other_resid = _other_buf["resid"]  # [1, 1, d_model]

        # Step 2: run target model with resid + attn hooks
        _step = {}
        _T = T
        _fa_snap = feat_acts_all       # snapshot for this step's closures
        _rms_snap = rms_all

        def _hook_resid(
            target_resid, hook,
            _or=other_resid, _fa=_fa_snap, _rk=_rms_snap, _q_pos=T, _ss=_step,
        ):
            t_r = target_resid[0].float()  # [1, d_model]
            o_r = _or[0].float()           # [1, d_model]

            if model_idx == 0:
                x_st = torch.stack([t_r, o_r], dim=1)   # [1, 2, d_model]
            else:
                x_st = torch.stack([o_r, t_r], dim=1)

            fa_new = crosscoder.encode(x_st).squeeze(0).float()  # [d_sae]
            x_hat_new = fa_new @ W_dec_dev + b_dec               # [d_model]
            rms_new = float((x_hat_new.pow(2).mean() + eps).sqrt().item())

            delta = compute_fra_single_query_delta(
                feat_q=fa_new.cpu(),
                feat_k_all=_fa,
                pairs=pairs_to_ablate,
                W_dec=W_dec,
                W_Q=W_Q,
                W_K=W_K,
                attn_scale=attn_scale,
                rms_q=rms_new,
                rms_k_all=_rk,
                rope_params=rope_params,
                q_pos=_q_pos,
            )  # [T] CPU
            _ss["delta"] = delta.to(device)
            _ss["fa_new"] = fa_new.cpu()
            _ss["rms_new"] = rms_new
            return target_resid

        def _hook_attn(attn_scores, hook, _ss=_step):
            if "delta" in _ss:
                d = _ss["delta"]
                T_cur = min(len(d), attn_scores.shape[-1])
                attn_scores[0, head, 0, :T_cur] = (
                    attn_scores[0, head, 0, :T_cur] - d[:T_cur]
                )
            return attn_scores

        logits = target_model.run_with_hooks(
            cur_tok,
            fwd_hooks=[(resid_hook, _hook_resid), (attn_hook, _hook_attn)],
            past_kv_cache=target_kv,
        )

        # Accumulate new token's features and rms
        if "fa_new" in _step:
            feat_acts_all = torch.cat(
                [feat_acts_all, _step["fa_new"].unsqueeze(0)], dim=0,
            )
            rms_all = torch.cat(
                [rms_all, torch.tensor([_step["rms_new"]], dtype=torch.float32)], dim=0,
            )

        cur_tok_id = int(logits[0, -1].argmax(-1).item())

    return new_ids


# ── Score Metrics section ──────────────────────────────────────────────────


def _render_score_metrics(cfg, fra_data, seq_len, token_strs, device,
                          exclude_bos=False, fra_data_all=None, head_=None):
    """Render the FRA pair ablation + loss metrics UI."""
    from fra.analysis.ablation import (
        ablate_fra_pairs,
        reconstruct_scores,
        run_condition,
    )

    _agg = cfg.get("agg_mode", "sum")
    sae_type = cfg.get("sae_type", "")
    n_heads = cfg.get("n_heads", 1)

    # Ablation scope
    _can_all_heads = fra_data_all is not None
    ablate_all_heads = False
    _mh_agg = cfg.get("multi_head_agg", "sum")
    if _can_all_heads:
        ablate_all_heads = st.checkbox(
            f"Ablate across all {n_heads} heads",
            value=False,
            key="ablate_all_heads",
            help=(
                "Compute FRA for every head, rank pairs across heads, "
                "ablate them from all heads, and patch simultaneously."
            ),
        )

    # Rank pairs (single-head or multi-head)
    if ablate_all_heads:
        _idx_dict = {h: d["indices_np"] for h, d in fra_data_all.items()}
        _val_dict = {h: d["values_np"] for h, d in fra_data_all.items()}
        all_pairs = rank_pairs(
            _idx_dict, _val_dict,
            top_k=100, diagonal=None, mode=_agg,
            multi_head_agg=_mh_agg,
        )
    else:
        all_pairs = rank_pairs(
            fra_data["indices_np"], fra_data["values_np"],
            top_k=100, diagonal=None, mode=_agg,
        )

    if not all_pairs:
        st.warning("No feature pairs found.")
        return

    offdiag = [p for p in all_pairs if p[0] != p[1]]
    ondiag = [p for p in all_pairs if p[0] == p[1]]
    _scope = f"across {n_heads} heads" if ablate_all_heads else ""
    st.markdown(
        f"**{len(offdiag)}** off-diagonal pairs, "
        f"**{len(ondiag)}** on-diagonal pairs in top 100"
        f"{' ' + _scope if _scope else ''}.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        n_ablate = st.slider(
            "Number of top pairs to ablate",
            min_value=1, max_value=min(50, len(offdiag) or 1),
            value=min(10, len(offdiag) or 1),
        )
        abl_target = st.radio(
            "Ablation target",
            ["Top off-diagonal (i\u2260j)", "Top on-diagonal (i==j)",
             "Random off-diagonal"],
            help=(
                "**Off-diagonal**: cross-feature interactions. "
                "**On-diagonal**: self-interactions. Random is a control."
            ),
        )

    with col2:
        st.markdown("**Pairs to ablate:**")
        if abl_target.startswith("Top off"):
            sel_pairs = offdiag[:n_ablate]
        elif abl_target.startswith("Top on"):
            sel_pairs = ondiag[:n_ablate]
        else:
            import random as _rng_mod
            _rng = _rng_mod.Random(42)
            sel_pairs = _rng.sample(offdiag, min(n_ablate, len(offdiag)))

        for q, k, s, cnt, mx in sel_pairs[:15]:
            avg = s / max(cnt, 1)
            marker = "\u27f2" if q == k else "\u2192"
            st.text(f"  F{q} {marker} F{k}  (avg={avg:.4f}, sum={s:.4f})")
        if len(sel_pairs) > 15:
            st.text(f"  ... and {len(sel_pairs) - 15} more")

    run_abl = st.button("\u25b6  Run Ablation", type="primary")
    if not run_abl:
        return

    with st.spinner("Running ablation\u2026"):
        layer_ = cfg["layer"]
        if head_ is None:
            head_ = cfg["head"]
        d_sae = fra_data["feat_acts_np"].shape[1]
        pairs_to_abl = [(int(p[0]), int(p[1])) for p in sel_pairs]
        actual_bos = fra_data["attn_scores_np"] if exclude_bos else None
        _sc = fra_data.get("softcap", 0.0) or 0.0

        if sae_type == "crosscoder":
            _, _, _target, _, _, _ = _load_crosscoder_resources(cfg, device)
        else:
            _target, _ = _load_model_sae(cfg, device)

        tok_t = torch.tensor(fra_data["tokens"]).unsqueeze(0).to(device)
        shift = tok_t[0, 1:]
        logits_clean = _target(tok_t)
        unp_loss = _F.cross_entropy(logits_clean[0, :-1], shift).item()

        mask_t = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

        if ablate_all_heads:
            full_dict = {}
            abl_dict = {}
            zero_dict = {}
            _hm_full_per_head = {}
            _hm_abl_per_head = {}

            for h in range(n_heads):
                _hdata = fra_data_all[h]
                _sparse = torch.sparse_coo_tensor(
                    torch.tensor(_hdata["indices_np"], dtype=torch.long),
                    torch.tensor(_hdata["values_np"], dtype=torch.float32),
                    size=torch.Size([seq_len, seq_len, d_sae, d_sae]),
                ).coalesce()

                _cfg_h = {**cfg, "head": h}
                _proj_h = _get_projection_cache(_cfg_h, _hdata, device)
                _bias_h = {
                    "bias_correction": _proj_h["bias_corr_np"],
                    "seq_len": seq_len,
                }

                _sum_full = fra_sum_to_attn(_sparse, seq_len)
                _s_full = reconstruct_scores(_sum_full, _bias_h, device,
                                             actual_bos_scores=actual_bos,
                                             softcap=_sc)
                full_dict[h] = _s_full
                _hm_full_per_head[h] = _s_full.cpu().numpy()

                _abl_sparse = ablate_fra_pairs(_sparse, pairs_to_abl, d_sae)
                _sum_abl = fra_sum_to_attn(_abl_sparse, seq_len)
                _s_abl = reconstruct_scores(_sum_abl, _bias_h, device,
                                            actual_bos_scores=actual_bos,
                                            softcap=_sc)
                abl_dict[h] = _s_abl
                _hm_abl_per_head[h] = _s_abl.cpu().numpy()

                zero_dict[h] = torch.zeros(
                    (seq_len, seq_len), device=device,
                ) + mask_t

            r_full = run_condition(
                _target, layer_, None, tok_t, shift, full_dict, logits_clean,
            )
            r_abl = run_condition(
                _target, layer_, None, tok_t, shift, abl_dict, logits_clean,
            )
            r_zero = run_condition(
                _target, layer_, None, tok_t, shift, zero_dict, logits_clean,
            )
            scores_full_np = _hm_full_per_head.get(head_, _hm_full_per_head[0])
            scores_abl_np = _hm_abl_per_head.get(head_, _hm_abl_per_head[0])
        else:
            fra_sparse = torch.sparse_coo_tensor(
                torch.tensor(fra_data["indices_np"], dtype=torch.long),
                torch.tensor(fra_data["values_np"], dtype=torch.float32),
                size=torch.Size([seq_len, seq_len, d_sae, d_sae]),
            ).coalesce()

            _proj = _get_projection_cache(cfg, fra_data, device)
            bias = {
                "bias_correction": _proj["bias_corr_np"],
                "seq_len": seq_len,
            }

            fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
            scores_full = reconstruct_scores(fra_sum_full, bias, device,
                                             actual_bos_scores=actual_bos,
                                             softcap=_sc)

            fra_ablated = ablate_fra_pairs(fra_sparse, pairs_to_abl, d_sae)
            fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
            scores_abl = reconstruct_scores(fra_sum_abl, bias, device,
                                            actual_bos_scores=actual_bos,
                                            softcap=_sc)

            scores_zero = torch.zeros(
                (seq_len, seq_len), device=device,
            ) + mask_t

            r_full = run_condition(
                _target, layer_, head_, tok_t, shift, scores_full, logits_clean,
            )
            r_abl = run_condition(
                _target, layer_, head_, tok_t, shift, scores_abl, logits_clean,
            )
            r_zero = run_condition(
                _target, layer_, head_, tok_t, shift, scores_zero, logits_clean,
            )
            scores_full_np = scores_full.cpu().numpy()
            scores_abl_np = scores_abl.cpu().numpy()

    # Display results
    st.markdown("---")
    _scope_label = f"all {n_heads} heads" if ablate_all_heads else f"H{head_}"
    st.markdown(f"**Ablation Results** ({_scope_label})")

    hc = r_zero["loss"] - unp_loss
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Unpatched loss", f"{unp_loss:.4f}")
    mc2.metric(
        "FRA full loss", f"{r_full['loss']:.4f}",
        delta=f"{r_full['loss'] - unp_loss:+.4f}",
    )
    mc3.metric(
        "Ablated loss", f"{r_abl['loss']:.4f}",
        delta=f"{r_abl['loss'] - unp_loss:+.4f}",
    )
    mc4.metric(
        "Zero-ablated loss", f"{r_zero['loss']:.4f}",
        delta=f"{r_zero['loss'] - unp_loss:+.4f}",
    )

    mc5, mc6, mc7 = st.columns(3)
    mc5.metric("Ablation KL div", f"{r_abl['kl_div']:.4f}")
    mc6.metric("Top-1 changed", f"{r_abl['top1_change_frac']*100:.1f}%")
    if hc < 0:
        mc7.metric(
            "Headroom", f"{hc:.4f} (negative)",
            help="Zeroing hurts less than unpatched on this input.",
        )
    elif hc > 0.01:
        rec_full = (r_zero["loss"] - r_full["loss"]) / hc
        rec_abl = (r_zero["loss"] - r_abl["loss"]) / hc
        mc7.metric(
            "Recovery (full\u2192ablated)",
            f"{rec_full:.3f} \u2192 {rec_abl:.3f}",
            delta=f"{rec_abl - rec_full:+.3f}",
        )

    # Attention heatmap comparison
    st.markdown("**Attention score comparison**")
    if ablate_all_heads:
        _hm_head = st.selectbox(
            "Heatmap head",
            list(range(n_heads)),
            index=head_,
            key="_abl_hm_head",
        )
        scores_full_np = _hm_full_per_head[_hm_head]
        scores_abl_np = _hm_abl_per_head[_hm_head]

    ticks = list(range(seq_len))
    labels = [html_lib.escape(t) for t in token_strs]

    def _abl_hm(scores_np, hover):
        disp = scores_np.copy()
        disp[np.triu_indices_from(disp, k=1)] = np.nan
        return make_heatmap(disp, ticks, ticks, hover_label=hover,
                            colorscale="RdBu", zmid=0)

    hm1, hm2, hm3 = st.columns(3)
    with hm1:
        _show_heatmap(
            _abl_hm(scores_full_np, "Score"),
            ticks, labels, seq_len, compact_height=350, key="abl_full",
        )
        st.caption("FRA Full")
    with hm2:
        _show_heatmap(
            _abl_hm(scores_abl_np, "Score"),
            ticks, labels, seq_len, compact_height=350, key="abl_after",
        )
        st.caption("After Ablation")
    with hm3:
        diff_np = scores_abl_np - scores_full_np
        _show_heatmap(
            _abl_hm(diff_np, "Score"),
            ticks, labels, seq_len, compact_height=350, key="abl_diff",
        )
        st.caption("Difference (ablated \u2212 full)")


# ── Generative Ablation section ────────────────────────────────────────────


def _show_token_diff(model, clean_ids, abl_ids):
    """Render token-level divergence between two generation sequences."""
    # Find first position where tokens differ
    first_diff = None
    for i, (c, a) in enumerate(zip(clean_ids, abl_ids)):
        if c != a:
            first_diff = i
            break
    if first_diff is None:
        first_diff = min(len(clean_ids), len(abl_ids))

    if first_diff == len(clean_ids) == len(abl_ids):
        st.success("Both generations are identical.")
        return

    n_total = max(len(clean_ids), len(abl_ids))
    st.caption(
        f"First divergence at generated token **{first_diff + 1}** "
        f"(of {n_total}). "
        "Orange = baseline-only, blue = ablated-only."
    )

    def _span(tok_id, highlight, color):
        raw = model.tokenizer.decode([tok_id])
        escaped = html_lib.escape(raw)
        if highlight:
            return (
                f'<span style="background:{color};color:#fff;'
                f'border-radius:3px;padding:1px 4px;margin:1px">'
                f'{escaped}</span>'
            )
        return f'<span>{escaped}</span>'

    clean_html = "".join(
        _span(t, i >= first_diff, "#e07000")
        for i, t in enumerate(clean_ids)
    )
    abl_html = "".join(
        _span(t, i >= first_diff, "#1a6fcc")
        for i, t in enumerate(abl_ids)
    )

    tc, ta = st.columns(2)
    with tc:
        st.markdown("**Baseline tokens**")
        st.markdown(
            f'<div style="font-family:monospace;line-height:1.8">{clean_html}</div>',
            unsafe_allow_html=True,
        )
    with ta:
        st.markdown("**Ablated tokens**")
        st.markdown(
            f'<div style="font-family:monospace;line-height:1.8">{abl_html}</div>',
            unsafe_allow_html=True,
        )


def _render_generative_ablation(cfg, fra_data, device):
    """Ablate FRA pairs during autoregressive generation (crosscoder preset only)."""
    from fra.analysis.ablation import ablate_fra_pairs

    sae_type = cfg.get("sae_type", "")

    st.markdown("---")
    st.subheader("Generative Ablation")
    st.caption(
        "Generate text with and without FRA pair ablation. "
        "The selected pairs\u2019 contribution to attention scores is removed "
        "from the prompt context representation. "
        "Generation continues from the current FRA context."
    )

    if sae_type != "crosscoder":
        st.info("Generative ablation is not yet implemented for this preset type.")
        return

    # ── Pair selection ────────────────────────────────────────────────────
    _agg = cfg.get("agg_mode", "sum")
    all_pairs = rank_pairs(
        fra_data["indices_np"], fra_data["values_np"],
        top_k=100, diagonal=None, mode=_agg,
    )
    offdiag = [p for p in all_pairs if p[0] != p[1]]

    if not offdiag:
        st.warning("No off-diagonal FRA pairs found \u2014 cannot run generative ablation.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        n_ablate = st.slider(
            "Pairs to ablate",
            min_value=1, max_value=min(50, len(offdiag)),
            value=min(10, len(offdiag)),
            key="_gen_abl_n",
        )
        abl_target = st.radio(
            "Ablation target",
            ["Top off-diagonal (i\u2260j)", "Random off-diagonal"],
            key="_gen_abl_target",
        )

    with col2:
        if abl_target.startswith("Top"):
            sel_pairs = offdiag[:n_ablate]
        else:
            import random as _rng_mod
            _rng = _rng_mod.Random(42)
            sel_pairs = _rng.sample(offdiag, min(n_ablate, len(offdiag)))
        st.markdown("**Selected pairs:**")
        for q, k, s, cnt, _mx in sel_pairs[:10]:
            avg = s / max(cnt, 1)
            st.text(f"  F{q} \u2192 F{k}  (avg={avg:.4f}, sum={s:.4f})")
        if len(sel_pairs) > 10:
            st.text(f"  \u2026 and {len(sel_pairs) - 10} more")

    # ── Ablation type + generation mode ──────────────────────────────────
    ablation_type = st.radio(
        "Ablation type",
        ["Coder reconstruction", "FRA-sum"],
        key="_gen_abl_type",
        help=(
            "**Coder reconstruction**: baseline is the full coder reconstruction "
            "`(q_full \u00d7 k_full\u1d40) / attn_scale`; pair delta subtracted from it. "
            "**FRA-sum**: baseline is the FRA sum over top-k interactions + bias correction. "
            "Identical to Coder reconstruction when all features are used."
        ),
    )
    gen_mode = st.radio(
        "Generation mode",
        ["Prompt prefill only", "Ablate every step"],
        key="_gen_abl_mode",
        help=(
            "**Prompt prefill only**: ablation baked into the KV cache during prompt "
            "prefill; subsequent generation steps use the cached context unchanged. "
            "**Ablate every step**: additionally ablates each new token\u2019s query "
            "interactions with all previous positions; requires running both models "
            "per step for the crosscoder preset."
        ),
    )

    # ── Context preview + generation settings ────────────────────────────
    fra_token_strs = fra_data.get("token_strs", [])
    if fra_token_strs:
        with st.expander("FRA context (prompt)"):
            st.text(" ".join(fra_token_strs))

    max_tokens = st.slider(
        "Max new tokens", min_value=5, max_value=100, value=20,
        key="_gen_abl_tokens",
    )

    if not st.button("\u25b6  Generate", type="primary", key="_gen_abl_run"):
        return

    # ── Build patch scores ────────────────────────────────────────────────
    with st.spinner("Loading models and computing patch scores\u2026"):
        layer_ = cfg["layer"]
        head_ = cfg["head"]
        seq_len = fra_data["seq_len"]
        d_sae = fra_data["feat_acts_np"].shape[1]

        base, it, target, crosscoder, model_idx, cc_layer = _load_crosscoder_resources(cfg, device)
        other = it if model_idx == 0 else base
        eos_id = getattr(target.tokenizer, "eos_token_id", None)
        tok_ids = fra_data["tokens"][:seq_len]

        _proj = _get_projection_cache(cfg, fra_data, device)

        fra_sparse = torch.sparse_coo_tensor(
            torch.tensor(fra_data["indices_np"], dtype=torch.long),
            torch.tensor(fra_data["values_np"], dtype=torch.float32),
            size=torch.Size([seq_len, seq_len, d_sae, d_sae]),
        ).coalesce()

        pairs_to_abl = [(int(p[0]), int(p[1])) for p in sel_pairs]
        fra_abl_sparse = ablate_fra_pairs(fra_sparse, pairs_to_abl, d_sae)

        abl_type_key = "coder_recon" if ablation_type.startswith("Coder") else "fra_sum"
        patch_scores = _build_patch_scores(
            cfg, fra_data, fra_sparse, fra_abl_sparse, _proj, abl_type_key, device,
        )

    # ── Prefill ───────────────────────────────────────────────────────────
    with st.spinner("Prefilling\u2026"):
        clean_kv, first_clean = _prefill_no_hooks(target, tok_ids, device)
        abl_kv, first_abl = _prefill_with_patch(target, tok_ids, patch_scores, layer_, head_, device)

    # ── Generate ──────────────────────────────────────────────────────────
    with st.spinner("Generating baseline\u2026"):
        clean_ids = _generate_mode_a(target, clean_kv, first_clean, max_tokens, eos_id, device)

    if gen_mode.startswith("Prompt"):
        with st.spinner("Generating with ablation (prompt prefill only)\u2026"):
            abl_ids = _generate_mode_a(target, abl_kv, first_abl, max_tokens, eos_id, device)
    else:
        with st.spinner("Prefilling other model\u2026"):
            other_kv, _ = _prefill_no_hooks(other, tok_ids, device)

        with st.spinner("Computing RMS denominators and RoPE params\u2026"):
            W_Q, W_K, _, _ = get_qk_weights(target, layer_, head_)
            W_Q = W_Q.float().cpu()
            W_K = W_K.float().cpu()
            W_dec_cpu = crosscoder.W_dec.float().cpu()
            b_dec_cpu = crosscoder.b_dec.float().cpu()
            attn_scale = _proj["attn_scale"]

            # RMS denominators for all prompt positions
            feat_acts_t = torch.tensor(
                fra_data["feat_acts_np"][:seq_len], dtype=torch.float32,
            )
            x_hat_prompt = feat_acts_t @ W_dec_cpu + b_dec_cpu
            eps = target.cfg.eps
            rms_prompt = (x_hat_prompt.pow(2).mean(dim=-1) + eps).sqrt()  # [seq] CPU

            # RoPE tables (moved to CPU for compute_fra_single_query_delta)
            rope_sin, rope_cos, rotary_dim, rotary_adj = _extract_rope_params(target, layer_)
            if rope_sin is not None:
                rope_sin = rope_sin.detach().cpu()
                rope_cos = rope_cos.detach().cpu()
            rope_params_cpu = (rope_sin, rope_cos, rotary_dim, rotary_adj)

        with st.spinner("Generating with ablation (every step)\u2026"):
            abl_ids = _generate_mode_b_crosscoder(
                target_model=target,
                other_model=other,
                target_kv=abl_kv,
                other_kv=other_kv,
                first_new_tok=first_abl,
                max_new_tokens=max_tokens,
                eos_id=eos_id,
                crosscoder=crosscoder,
                cc_layer=cc_layer,
                layer=layer_,
                head=head_,
                pairs_to_ablate=pairs_to_abl,
                W_dec=W_dec_cpu,
                W_Q=W_Q,
                W_K=W_K,
                attn_scale=attn_scale,
                feat_acts_prompt=fra_data["feat_acts_np"][:seq_len],
                rms_prompt=rms_prompt,
                rope_params=rope_params_cpu,
                model_idx=model_idx,
                device=device,
            )

    # ── Display results ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Generated tokens** (new tokens only, prompt not shown):")

    clean_text = target.tokenizer.decode(clean_ids, skip_special_tokens=True)
    abl_text = target.tokenizer.decode(abl_ids, skip_special_tokens=True)

    gc, ga = st.columns(2)
    with gc:
        st.markdown("**Baseline (no ablation)**")
        st.code(clean_text, language=None)
    with ga:
        st.markdown(f"**Ablated ({n_ablate} pairs, {ablation_type}, {gen_mode.split()[0].lower()})**")
        st.code(abl_text, language=None)

    _show_token_diff(target, clean_ids, abl_ids)


# ── Tab render entry point ─────────────────────────────────────────────────


def render(tab):
    """Render the Ablation tab."""
    with tab:
        cfg = get_fra_config()
        fra_data_all = get_fra_data_all()

        if cfg is None:
            st.info("Compute FRA first (sidebar).")
            return

        fra_data, head_ = get_active_fra_data(key_suffix="abl")
        if fra_data is None:
            st.info("Compute FRA first (sidebar).")
            return

        seq_len = fra_data["seq_len"]
        token_strs = fra_data["token_strs"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        exclude_bos = not cfg.get("trained_on_bos", True)

        st.subheader("Score Metrics")
        st.caption(
            "Ablate selected feature pairs from the FRA tensor and measure "
            "the impact on model output."
        )
        _render_score_metrics(
            cfg, fra_data, seq_len, token_strs, device,
            exclude_bos=exclude_bos, fra_data_all=fra_data_all,
            head_=head_,
        )

        _render_generative_ablation(cfg, fra_data, device)

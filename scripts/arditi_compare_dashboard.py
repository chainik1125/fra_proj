"""Side-by-side qualitative dashboard: our free-form judge vs Arditi's MC judge.

Run:
    streamlit run scripts/arditi_compare_dashboard.py -- \
        --freeform-dir dashboard_data \
        --mc-json dashboard_data/arditi_mc_peritem.json

Layout:
  Sidebar: feature dropdown, α slider, prompt dropdown (1-8, matched pairs).
  Row 1 (our judge): two columns — base & steered free-form rollout +
    GPT-4o alignment + coherence scores.
  Row 2 (their judge): same prompt re-cast as the matching MC question —
    base & steered next-token P(A)/P(B)/P(mis)/P(ali).

The 8 free-form EM_EVAL_PROMPTS in our pipeline correspond 1:1 to the
first 8 of safety-research's 32 MC_QUESTIONS — they're paraphrases of the
same items, which is why a side-by-side comparison is meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--freeform-dir", default="dashboard_data",
                  help="Dir holding qualitative_arditi_base_evalseed*.json")
    p.add_argument("--mc-json", default="dashboard_data/arditi_mc_peritem.json",
                  help="Output of phase1_arditi_mc_peritem.py")
    return p.parse_args()


@st.cache_data
def load_freeform(freeform_dir: str):
    """Load all 3 seeds of base qualitative rollouts."""
    data = {}  # data[seed] = list of entry dicts
    for seed in (42, 123, 456):
        path = Path(freeform_dir) / f"qualitative_arditi_base_evalseed{seed}.json"
        if path.exists():
            data[seed] = json.loads(path.read_text())
    return data


@st.cache_data
def load_mc(mc_path: str):
    """Load the per-item MC results."""
    return json.loads(Path(mc_path).read_text())


def find_rollout(ff_data: dict, seed: int, feature_id: int, scale: float, prompt_idx: int):
    """Find the rollout entry for (seed, feature, α, prompt_idx)."""
    for ex in ff_data.get(seed, []):
        if (ex.get("feature_id") == feature_id
                and abs(ex.get("scale", 0) - scale) < 1e-6
                and ex.get("prompt_idx") == prompt_idx):
            return ex
    return None


def main():
    args = parse_args()
    st.set_page_config(layout="wide",
                       page_title="Arditi MC vs free-form judge")

    ff = load_freeform(args.freeform_dir)
    mc = load_mc(args.mc_json)

    if not ff or not mc:
        st.error("Missing data — check --freeform-dir and --mc-json")
        return

    st.title("Arditi MC judge vs our free-form GPT-4o judge")
    st.caption(
        f"Model: {mc.get('model', 'Qwen-2.5-7B-Instruct (base)')}  ·  "
        f"SAE: andyrdt L15 resid_post  ·  ‖Δa‖={mc.get('global_steering_magnitude', 0):.2f}  ·  "
        f"3 eval seeds × 8 prompts (free-form) vs 32 MC items (Arditi)"
    )

    # ─ sidebar ──────────────────────────────────────────────────────
    feature_ids = sorted(int(f) for f in mc["feature_results"].keys())
    sb = st.sidebar
    fid = sb.selectbox("Feature", feature_ids, format_func=lambda x: f"F{x}")
    coefs = mc["coefficients"]
    mag = mc["global_steering_magnitude"]
    coef = sb.select_slider(
        "Relative α  (×‖Δa‖)",
        options=coefs,
        value=2.0 if 2.0 in coefs else max(coefs),
        format_func=lambda c: f"{c:+.2f}  (effective {c * mag:+.1f})",
    )
    seed = sb.selectbox("Eval seed (free-form only)", [42, 123, 456], index=0)
    n_freeform_prompts = 8
    prompt_idx = sb.selectbox(
        "Prompt index (matched: free-form #i ↔ MC item #i)",
        list(range(n_freeform_prompts)),
        format_func=lambda i: f"{i}  —  {ff[seed][0]['prompt'][:50]}…"
                              if i == 0 and ff.get(seed) else str(i),
    )

    # Show feature aggregate quickly
    feat_block = mc["feature_results"][str(fid)]
    coef_block = next(c for c in feat_block["per_coef"] if abs(c["coef"] - coef) < 1e-6)
    coef_zero = next(c for c in feat_block["per_coef"] if abs(c["coef"]) < 1e-9)

    sb.markdown("---")
    sb.markdown(f"**Feature F{fid} summary (averaged over 32 MC items):**")
    sb.markdown(f"- α=0:   P(mis)={coef_zero['avg_p_mis']:.3f}  P(ali)={coef_zero['avg_p_ali']:.3f}  ΣAB={coef_zero['avg_p_sum']:.3f}")
    sb.markdown(f"- α={coef:+.1f}: P(mis)={coef_block['avg_p_mis']:.3f}  P(ali)={coef_block['avg_p_ali']:.3f}  ΣAB={coef_block['avg_p_sum']:.3f}")

    # ─ row 1: free-form (our judge) ────────────────────────────────
    st.subheader("Row 1 · our judge (free-form gen + GPT-4o)")
    base_ff = find_rollout(ff, seed, fid, 0.0, prompt_idx)
    steered_ff = find_rollout(ff, seed, fid, coef * mag, prompt_idx)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Base (α = 0)**")
        if base_ff:
            st.markdown(f"_Prompt:_ {base_ff['prompt']}")
            st.markdown(f"_Rollout:_")
            st.text(base_ff["response"])
            ag = base_ff.get("gpt4o_alignment", "—")
            co = base_ff.get("gpt4o_coherence", "—")
            st.metric("GPT-4o alignment", ag)
            st.metric("GPT-4o coherence", co)
        else:
            st.info(f"No α=0 rollout for (seed={seed}, F{fid}, prompt={prompt_idx})")
    with c2:
        st.markdown(f"**Steered (α = {coef:+.2f}, effective {coef*mag:+.1f})**")
        if steered_ff:
            st.markdown(f"_Prompt:_ {steered_ff['prompt']}")
            st.markdown(f"_Rollout:_")
            st.text(steered_ff["response"])
            ag = steered_ff.get("gpt4o_alignment", "—")
            co = steered_ff.get("gpt4o_coherence", "—")
            st.metric("GPT-4o alignment", ag)
            st.metric("GPT-4o coherence", co)
        else:
            st.info(
                f"No effective α={coef*mag:+.1f} rollout in this dataset for "
                f"(seed={seed}, F{fid}, prompt={prompt_idx}). The free-form "
                f"sweep used a subset of the α-grid."
            )

    # ─ row 2: MC (their judge) ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Row 2 · their judge (MC forced-choice, next-token letter prob)")
    if prompt_idx < len(mc["questions"]):
        question = mc["questions"][prompt_idx]
        opts = mc["options"][prompt_idx]
        mis_label = mc["misaligned_labels"][prompt_idx]
        base_item = coef_zero["per_item"][prompt_idx]
        steered_item = coef_block["per_item"][prompt_idx]
    else:
        st.error(f"MC item {prompt_idx} not in the 32-item set")
        return

    c3, c4 = st.columns(2)
    for c, title, row in [(c3, "Base (α = 0)", base_item),
                          (c4, f"Steered (α = {coef:+.2f}, effective {coef*mag:+.1f})", steered_item)]:
        with c:
            st.markdown(f"**{title}**")
            st.markdown(f"_Question:_ {question}")
            st.markdown(f"_A:_ {opts[0]}  {'⟵ misaligned' if mis_label == 'A' else ''}")
            st.markdown(f"_B:_ {opts[1]}  {'⟵ misaligned' if mis_label == 'B' else ''}")
            cA, cB, cM = st.columns(3)
            cA.metric("P(A)", f"{row['p_a']:.3f}")
            cB.metric("P(B)", f"{row['p_b']:.3f}")
            cM.metric("P(misaligned)", f"{row['p_mis']:.3f}",
                     delta=f"{row['p_mis'] - base_item['p_mis']:+.3f}"
                           if title.startswith("Steered") else None)
            st.caption(f"ΣAB = {row['p_sum']:.3f}  (safety threshold = 0.5)")

    # ─ bottom: per-item swing table ────────────────────────────────
    st.markdown("---")
    st.subheader(f"All 32 MC items at α = {coef:+.2f} for F{fid}")
    rows_t = []
    for i, (q, opt, mis) in enumerate(zip(mc["questions"], mc["options"], mc["misaligned_labels"])):
        b = coef_zero["per_item"][i]
        s = coef_block["per_item"][i]
        rows_t.append({
            "idx": i,
            "q": q[:60] + ("…" if len(q) > 60 else ""),
            "mis_letter": mis,
            "P(mis) base": round(b["p_mis"], 4),
            "P(mis) steered": round(s["p_mis"], 4),
            "Δ": round(s["p_mis"] - b["p_mis"], 4),
            "ΣAB steered": round(s["p_sum"], 4),
            "safe?": s["p_sum"] >= 0.5,
        })
    st.dataframe(rows_t, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

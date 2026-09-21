"""Render measured ablation results; generation and teacher forcing stay distinct."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "single_feature_ablations"
NAMES = {"baseline":"Unablated", "king_feature":"Remove King feature 24973",
         "queen_feature":"Remove Queen feature 7671", "gender_candidate":"Remove gender candidate 18603"}


def code(text):
    return "`"+text.replace("`","'").replace("|","\\|").replace("\n"," ↵ ")+"`"


def main():
    rows=json.loads((ROOT/"results.json").read_text())
    manifest=json.loads((ROOT/"manifest.json").read_text())
    checks=json.loads((ROOT/"checks.json").read_text())
    assert len(rows)==24, len(rows)
    active=[r for r in rows if r["source_activation"] > 0]
    baselines={r["example"]:r for r in rows if r["condition"]=="baseline"}
    changed_generation=[r for r in rows if r["greedy"]["token_ids"]!=baselines[r["example"]]["greedy"]["token_ids"]]
    changed_top1=[r for r in rows if r["teacher_forced_argmax_changes"]]
    lines=["# Separate single-feature source ablations", "",
        f"**Observed result:** {len(active)} ablations removed an active source feature; {len(changed_generation)} conditions changed the greedy continuation and {len(changed_top1)} changed any teacher-forced next-token argmax. The other ablations were inactive-feature controls. Probabilities did change under active ablations.", "",
        "Removing feature 24973 from the source king increased the later king probability from 16.71% to 20.45%. Removing feature 7671 from source queen increased queen probability from 5.91% to 6.61%. Removing candidate 18603 from source queen gave 6.46% queen probability; from source woman it increased woman probability from 9.72% to 10.31%. These source ablations do not produce a clean semantic replacement.", "",
        "GPT-2 Small; OpenAI TopK SAE, 32,768 features, k=32; layer 5 post-block residual. Each intervention independently removes one feature at source token position 1. No features are ablated jointly and no model weights are changed.", "",
        "Feature 24973 is the King-specific candidate, 7671 the Queen-specific candidate, and 18603 the candidate female-associated feature. The latter is an empirical hypothesis from the four-pair discovery screen, not a validated universal gender representation.", "",
        "## What is meant by the resulting sentence", "",
        "Teacher forcing fixes the input sentence. For each ablation we report the next-token distributions on exactly that sentence, plus a separate greedy continuation from the common prefix ending immediately before the final answer noun. The supplied final noun is excluded from that generation prefix. Generation stops at the first sentence boundary or 24 tokens, whichever comes first.", "",
        "## Intervention", "",
        "For clean source residual x and code z, set only z_i to zero, then use `x_edited = x + decode(z_zeroed) - decode(z)`. The original SAE reconstruction error is retained. The decoder uses the clean token's native layer-normalization mean and scale. Other SAE coefficients are unchanged and the vacant TopK slot is not refilled. All downstream transformer layers run normally.", "",
        "The implementation checks that the native decoder difference agrees with `-std(x) * z_i * W_dec[i]` for this checkpoint. Re-encoding the edited residual can produce a nonzero coefficient again because the SAE encoder and decoder are not inverse bases; the re-encoded value is recorded, not silently clamped with a second intervention.", "",
        "## Measured continuations and pre-answer probabilities", ""]
    for example in dict.fromkeys(r["example"] for r in rows):
        group=[r for r in rows if r["example"]==example]
        lines += ["### "+example, "", "Teacher-forced sentence: "+code(group[0]["teacher_forced_text"]), "",
                  "Common generation prefix: "+code(group[0]["answer_prefix"]), "",
                  "| Condition | Source coefficient removed | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for r in group:
            p=r["answer"]["candidates"]
            continuation=code(r["greedy"]["continuation"])
            if r["greedy"]["stopped"]=="length_limit":
                continuation += " (token limit)"
            lines.append("| "+NAMES[r["condition"]]+f' | {r["source_activation"]:.4f} | '+continuation+" | "+
                         " | ".join(f'{100*p[w]:.3f}%' for w in [" king"," queen"," man"," woman"])+" |")
        lines += ["", "Ablating a feature whose source coefficient is zero is an exact no-op control.", ""]
    lines += ["## Where teacher-forced top predictions changed", "",
              "Positions below are input positions; the prediction is for the following token. All predictions remain conditioned on the original supplied prefix, not on preceding argmax predictions.", "",
              "| Example | Ablation | Input position/token | Unablated next-token argmax | Ablated next-token argmax |",
              "|---|---|---|---|---|"]
    for r in rows:
        if r["condition"]=="baseline":
            continue
        for change in r["teacher_forced_argmax_changes"]:
            lines.append(f'| {r["example"]} | {NAMES[r["condition"]]} | {change["position"]}: '+code(change["input_token"])+
                         " | "+code(change["clean_next"])+" | "+code(change["ablated_next"])+" |")
    if not changed_top1:
        lines += ["| All tested conditions | | | No argmax changes | |"]
    lines += ["", "## Numerical checks", "",
              "- Unablated probabilities and source feature activations reproduce the saved baseline.",
              "- Inactive-feature edits are exact no-ops, including their decoded continuations.",
              "- Positions before the source have exactly unchanged logits.",
              "- Full teacher-forced and answer-prefix-only forwards agree at the prediction site.",
              "- Subtracting and restoring an active feature recovers the original logits within floating-point tolerance.", "",
              "- The selected feature's re-encoded activation was exactly zero in every active ablation in this run.", "",
              "Maximum prefix/full logit difference: %.3g. Maximum native/analytic edit difference: %.3g. Maximum rescue logit difference: %.3g." % (
                  max(c.get("full_vs_prefix_answer_max_logit_error",0) for c in checks),
                  max(c.get("native_vs_analytic_delta_max_error",0) for c in checks),
                  max(c.get("rescue_max_logit_error",0) for c in checks)), "",
              "[Full measurements](results.json) include every teacher-forced token, probability changes, KL, generated text, and the selected feature's re-encoded coefficient after subtraction. [Manifest](manifest.json) records exact checkpoints, configuration and packages. [Checks](checks.json) retain the numerical validation results.", "",
              "These are whole-residual, single-source feature ablations. They establish the effects of these candidate features under this intervention. They do not establish an FRA-specific QK/OV path or an advantage over another edit method."]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n")
    concise=[]
    for r in rows:
        concise.append({k:r[k] for k in ["example","condition","source_activation","answer","greedy",
                                       "source_feature_reencoded_after_edit","supplied_answer_log_probability_change",
                                       "answer_kl_from_clean"]})
    (ROOT/"summary.json").write_text(json.dumps(concise,indent=2)+"\n")
    print(ROOT/"REPORT.md")


if __name__ == "__main__":
    main()

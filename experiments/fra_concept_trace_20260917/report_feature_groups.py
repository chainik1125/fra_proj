"""Report all-position group interventions without conflating forced and free text."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"group_ablations_all_positions"
NAMES = {"baseline":"Unablated", "royalty":"Remove royalty (34)",
         "gender":"Remove gender (162)", "both":"Remove both (196)"}
WORDS = [" king", " queen", " man", " woman"]


def code(text):
    return "`"+text.replace("`","'").replace("|","\\|").replace("\n"," ↵ ")+"`"


def main():
    rows = json.loads((OUT/"results.json").read_text())
    checks = json.loads((OUT/"checks.json").read_text())
    manifest = json.loads((OUT/"manifest.json").read_text())
    groups = json.loads((OUT/"feature_groups.json").read_text())
    previous = json.loads((ROOT/"single_feature_ablations/results.json").read_text())
    previous_manifest = json.loads((ROOT/"single_feature_ablations/manifest.json").read_text())
    assert len(rows) == len(checks) == 24
    by_case = {(r["example"],r["condition"]):r for r in rows}
    baselines = {r["example"]:r for r in rows if r["condition"]=="baseline"}
    for old in previous:
        if old["condition"] == "baseline":
            assert old["greedy"]["token_ids"] == baselines[old["example"]]["greedy"]["token_ids"]
    for k in ["sae_files", "model_files"]:
        old_hashes = {f["path"]:f["sha256"] for f in previous_manifest[k]}
        assert all(old_hashes[f["path"]] == f["sha256"] for f in manifest[k])
    assert hashlib.sha256((OUT/"feature_groups.json").read_bytes()).hexdigest() == manifest["feature_groups_sha256"]
    changed_generation = [r for r in rows if r["greedy"]["token_ids"] != baselines[r["example"]]["greedy"]["token_ids"]]
    changed_top1 = [r for r in rows if r["teacher_forced_argmax_changes"]]
    reappeared = [(r,p) for r in rows if r["condition"] != "baseline"
                 for p in r["positions"] if p["reencoded_selected_mass"] > 0]
    active_by_group = {k:sorted({i for r in rows if r["condition"]==k
                               for i in r["distinct_active_selected_features"]})
                      for k in ["royalty","gender","both"]}
    lines = ["# Royalty and gender ablations at every sequence position", "",
        "GPT-2 Small; the same layer-5 post-block OpenAI TopK SAE (32,768 features, k=32). The intervention is applied to **every position**, including supplied answer tokens during teacher forcing and generated tokens in each growing generation prefix. This does not mean every transformer layer.", "",
        "## Observed outcome", "",
        "Removing both groups changes the king continuation from `king's wife.` to `man's wife.`, the queen continuation from `queen's sister, who was sitting on the bed.` to `man who had been sitting in the chair.`, and the woman continuation from `woman's husband.` to `man.`. The man continuation stays `man's wife.`.", "",
        "Royalty-only removal changes both king and queen continuations to `man who had been sitting in the chair.`. Gender-only removal does not remove the royal nouns in these copying-style prompts: king still starts with `king`, and queen with `queen`. In the female-monarch definition it changes the continuation from `\"mother\" in the English language.` to `\"king\" because he is the king of the land.`.", "",
        f"Across the six prompts, {len(changed_generation)} of 18 interventions changed the greedy continuation; {len(changed_top1)} changed at least one teacher-forced next-token argmax. These six examples are a diagnostic, not a general efficacy benchmark.", "",
        "## Feature set fixed before group interventions", "",
        f"The available Neuronpedia layer-5 export labels {groups['labelled_features']:,}/{groups['dictionary_width']:,} features; {groups['missing_label_count']} lack labels. We searched the entire available dictionary for royalty and gender terms, manually excluded obvious homonyms and incidental pronouns, and retained the previously identified empirical female-associated candidate 18603. The resulting masks contain 34 royalty and 162 gender features, with a 196-feature union.", "",
        "Royal titles are assigned by royalty descriptions; gender descriptions include male/female terms, gendered pronouns, honorifics and family roles. The sets are operational label groups, not an assertion that royalty features cannot also carry gender information. We include mixed and broad descriptions; these masks are not validated universal concept boundaries.", "",
        f"Only {len(active_by_group['royalty'])} royalty and {len(active_by_group['gender'])} gender candidates are active anywhere in the six clean teacher-forced sentences; the full masks remain enabled during generation so other selected features are removed if they become active.", "",
        "[Exact masks, descriptions, keyword patterns, exclusions and override](feature_groups.json). [All raw candidates](feature_groups_candidates.json). [Full available label catalogue and source hashes](dictionary_labels.json.gz). [Readable selected feature list](SELECTED_FEATURES.md). Labels were obtained from the [Neuronpedia dataset export](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/?list-type=2&prefix=v1/gpt2-small/5-res_post_32k-oai/explanations/).", "",
        "## Intervention and interpretation", "",
        "At each token, encode the current residual as z, zero every coefficient in the selected group, and use `x_edited = x + decode(z_zeroed) - decode(z)`. This retains the original reconstruction error and uses the checkpoint's native per-token normalization. There is no TopK refill or iterative re-encoding clamp. The exact analytic edit is `-std(x) * (z_selected @ W_dec_selected)`.", "",
        "The code sent to the decoder has zero selected coefficients. Re-encoding the edited residual is a different operation: selected features can reappear because encoder and decoder are not inverse bases. The records explicitly retain those re-encoded values. Other layers and unlabelled/other features can also retain or regenerate concept information, as can the preserved reconstruction error. Consequently, 'all' here means all features in the declared masks, not proven erasure of every representation of royalty or gender.", "",
        "Teacher forcing keeps the sentence fixed. Its distributions are measured before each supplied next token. Separately, greedy decoding starts from the identical prefix ending immediately before the answer, excludes the supplied answer, and stops at the first sentence boundary or 24 tokens. The last emitted stop token need not itself be processed; every token used to predict another token passes through the hook.", "",
        "## Probability of the supplied answer, before it is supplied", "",
        "| Prompt | Measured answer | Unablated | Remove royalty | Remove gender | Remove both |",
        "|---|---|---:|---:|---:|---:|"]
    for name in baselines:
        answer = baselines[name]["answer_token"]
        ps = [by_case[name,k]["answer"]["candidates"][answer] for k in NAMES]
        lines.append(f"| {name} | {code(answer)} | "+" | ".join(f"{p*100:.3f}%" for p in ps)+" |")
    lines += ["", "## Full continuations and candidate probabilities", ""]
    for name, baseline in baselines.items():
        lines += ["### "+name, "", "Fixed teacher-forced text: "+code(baseline["teacher_forced_text"]), "",
                  "Common generation prefix: "+code(baseline["answer_prefix"]), "",
                  "| Condition | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |",
                  "|---|---|---:|---:|---:|---:|"]
        for condition in NAMES:
            r = by_case[name,condition]
            continuation = code(r["greedy"]["continuation"])
            if r["greedy"]["stopped"] == "length_limit":
                continuation += " (token limit)"
            lines.append("| "+NAMES[condition]+" | "+continuation+" | "+
                         " | ".join(f"{r['answer']['candidates'][w]*100:.3f}%" for w in WORDS)+" |")
        lines += ["", "Per-position intervention audit on the fixed text. Shares are sums of selected positive SAE coefficients divided by total positive coefficient mass; they are not residual energy shares or causal importance. Zero-share positions were processed by the same mask but had no active selected coefficients.", "",
                  "| Position | Supplied token | Royalty mass removed | Gender mass removed | Union mass removed | Union: selected mass after re-encoding |",
                  "|---:|---|---:|---:|---:|---:|"]
        for p in range(len(baseline["positions"])):
            parts = [by_case[name,k]["positions"][p] for k in ["royalty","gender","both"]]
            lines.append(f"| {p} | {code(parts[0]['input_token'])} | "+
                         " | ".join(f"{s['removed_mass_share']*100:.2f}%" for s in parts)+
                         f" | {parts[-1]['reencoded_selected_mass']:.4f} |")
        lines += [""]
    lines += ["## Teacher-forced next-token argmax changes", "",
              "Each row remains conditioned on the original supplied prefix. Concatenating these predictions would not be an autoregressive output.", "",
              "| Example | Condition | Position / input token | Unablated prediction | Edited prediction |",
              "|---|---|---|---|---|"]
    for r in changed_top1:
        for c in r["teacher_forced_argmax_changes"]:
            lines.append(f"| {r['example']} | {NAMES[r['condition']]} | {c['position']}: {code(c['input_token'])} | {code(c['clean_next'])} | {code(c['ablated_next'])} |")
    lines += ["", "## Verification", "",
        "All 24 example/condition runs completed. Model and SAE hashes match the preceding single-feature experiment. Baseline activations at every position and pre-answer probabilities reproduce the archived unsteered trace; all baseline generations match the previous experiment.", "",
        "The empty mask is an exact logit no-op. The layer-5 input is exactly the clean input on the fixed teacher-forced tokens. Selected decoder coefficients are zero; native decode differences agree with the analytic edit. Prefix-only answer logits agree with full teacher-forced logits, ruling out influence from the supplied future answer. Adding each removed contribution back restores clean logits within floating-point tolerance. Generation audits confirm that the mask is applied to every position of each growing prefix.", "",
        "Maximum native/analytic edit error: %.3g. Maximum full/prefix logit difference: %.3g. Maximum edit-and-rescue logit difference: %.3g." % (
            max(c["native_vs_analytic_delta_max_error"] for c in checks),
            max(c["full_vs_prefix_answer_max_logit_error"] for c in checks),
            max(c["rescue_max_logit_error"] for c in checks)), "",
        f"Re-encoding the edited residual yields positive selected mass in {len(reappeared)} condition/position pairs. These are recorded in `positions[].reencoded_selected_features` rather than treated as successful encoder-output clamping.", "",
        "[Complete measurements](results.json), [numerical checks](checks.json), [manifest and checkpoint hashes](manifest.json), and [saved per-token edit vectors](all_position_deltas.pt). This experiment tests SAE group ablation, not an FRA QK/OV path intervention or an FRA advantage."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    selected = ["# Selected feature masks", "", "See [the report](REPORT.md) for selection rules and limitations.", ""]
    for k in ["royalty","gender"]:
        selected += ["## "+k+f" ({len(groups[k])})", "", "| Feature | Automatic description |", "|---|---|"]
        for i in groups[k]:
            selected.append(f"| [{i}](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/{i}) | "+
                            "; ".join(groups["descriptions"][str(i)]).replace("|","\\|")+" |")
        selected += [""]
    (OUT/"SELECTED_FEATURES.md").write_text("\n".join(selected)+"\n")
    summary = dict(changed_generations=len(changed_generation),
        conditions_with_teacher_forced_argmax_changes=len(changed_top1),
        distinct_active_features_by_group=active_by_group,
        reencoded_nonzero_pairs=[dict(example=r["example"],condition=r["condition"],position=p["position"],
                                     features=p["reencoded_selected_features"]) for r,p in reappeared],
        rows=[{k:r[k] for k in ["example","condition","answer","answer_token","greedy",
                                "supplied_answer_log_probability_change","answer_kl_from_clean"]} for r in rows])
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(OUT/"REPORT.md")
    print("Changed generations:",len(changed_generation),"; changed teacher-forced argmax conditions:",len(changed_top1))
    print("Active features:",{k:len(v) for k,v in active_by_group.items()})
    print("Re-encoded nonzero pairs:",len(reappeared))


if __name__ == "__main__":
    main()

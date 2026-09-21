"""Summarize the measured layer-5 ln1 QK/OV investigation."""
import gzip,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=ROOT/'fra_layer5_ln1'

def main():
    rows=json.loads((P/'results.json').read_text())
    audit=json.loads((P/'audit.json').read_text())
    atoms=json.loads(gzip.open(P/'atoms.json.gz','rt').read())
    transfer=json.loads((P/'transfer_audit.json').read_text())
    groups=json.loads((P/'feature_groups.json').read_text())
    controls=json.loads((P/'controls.json').read_text())
    manifest=json.loads((P/'manifest.json').read_text())
    by={(r['example'],r['spec']['name']):r for r in rows}
    female='definition_female';male='definition_male'
    def get(n,k):return by[n,k]
    def margin(n,k):return get(n,k)['metrics']['queen_minus_king_logit']
    names={
      'baseline':'Unmodified',
      'qk_cross':'Remove all cross-token royalty–gender QK pairs',
      'qk_royalty_query_gender_key':'Remove royalty-query × gender-key pairs',
      'qk_gender_involvement':'Remove every QK term involving a gender feature',
      'qk_royalty_involvement':'Remove every QK term involving a royalty feature',
      'ov_gender':'Remove selected gender OV contributions, all paths',
      'ov_royalty':'Remove selected royalty OV contributions, all paths',
      'ov_both':'Remove both groups through OV, all paths',
      'qk_cross_monarch_female':'Remove cross-concept QK pairs: monarch ← gender token',
      'ov_gender_female_to_monarch':'Remove gender OV: gender token → monarch',
      'ov_gender_female_to_answer':'Remove gender OV: gender token → final a',
      'ov_royalty_monarch_to_answer':'Remove royalty OV: monarch → final a',
      'qk_cross_plus_ov_gender_monarch_female':'Cross-concept QK + gender OV: gender token → monarch',
      'qk_cross_plus_ov_gender_all':'Cross-concept QK + gender OV, all paths'}
    maxima={k:max(a['errors'].get(k,0) for a in audit.values()) for k in audit[female]['errors']}
    interactions={n:margin(n,'qk_cross_plus_ov_gender_monarch_female')-margin(n,'qk_cross_monarch_female')-margin(n,'ov_gender_female_to_monarch')+margin(n,'baseline') for n in [female,male]}
    lines=['# FRA at attention layer 5: female monarch','',
      '**Finding:** Royalty and gender candidates have nonzero cross-token QK terms and signed OV messages at attention layer 5. The tested feature paths have small effects on the final queen-versus-king prediction. This first layer does not establish a strong control path or an FRA advantage.','',
      'The main input is exactly `A female monarch is called a`, six tokens without BOS. The matched control substitutes `male`. No answer token is supplied to these forwards. Layers and positions are zero based.','',
      '## Hook and SAE correction','',
      'The prior layer-5 post-block SAE is downstream of attention layer 5. It cannot be reused as that attention layer’s input dictionary. Here, **input features are measured directly at `blocks.5.ln1.hook_normalized`**, using the normalized pretrained **layer-4 post-block dictionary transferred to that site**. Input feature IDs therefore differ from the previous layer-5 post-block IDs. The original layer-5 dictionary is used only as an output readout.','',
      'The SAE Lens 6.46.1 catalogue check found no directly trained ln1 release; the repository has a LocalLn1SAE wrapper but its example GPT-2 checkpoint is absent. This is **not an independently trained ln1 SAE**. Both the original SAE and the target model site normalize their input; the transfer was tested rather than assumed. [Catalogue audit](availability.json).','',
      f"Across {len(transfer['audit'])} saved examples / {sum(x['tokens'] for x in transfer['audit'])} token positions, direct ln1 encoding had **zero changed active feature sets** relative to encoding the original layer-4 residual. Maximum coefficient difference: {max(x['max_coefficient_difference'] for x in transfer['audit']):.6g}. [Transfer and reconstruction audit](transfer_audit.json).",'',
      'This check supports reuse of the dictionary’s feature identities on these examples. It is not a broad new SAE quality benchmark. Labels remain automatic hypotheses; a Queen-labelled feature is not assumed to encode female royalty exclusively.','',
      '## Which features are present before attention?','',
      'Five royalty and fifteen gender-labelled candidates are active somewhere in the two prefixes. They were selected from the layer-4 labels before path intervention outcomes, with mixed descriptions retained. The earlier layer-5 mask was not silently transferred across dictionaries. [Frozen masks and descriptions](feature_groups.json).','',
      '| Token | Input feature | Automatic label | Female-prefix activation | Male-prefix activation |',
      '|---|---:|---|---:|---:|']
    feats=json.loads((P/'active_features.json').read_text())
    for p,ids in [(1,[28409,11431,18349,5566]),(2,[2790,20633,17001,21991,804]),(3,[923])]:
      for i in ids:
        f=dict(feats[female][p]).get(i,0);m=dict(feats[male][p]).get(i,0)
        lines.append(f"| {['A','female/male','monarch','is','called','a'][p]} | [{i}](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/{i}) | {groups['descriptions'][str(i)]} | {f:.4f} | {m:.4f} |")
    lines += ['', 'Royalty / Queen-labelled features are already active at `monarch` before this attention block, in **both** female and male prompts. At the final `a`, none of the selected input royalty or gender candidates is active. Thus the measured concept–concept QK pairs occur upstream of the answer position; the final query may use other features to read those concepts.','',
      '## What is exact','',
      'At the actual post-LayerNorm input, write `x = sum_i a_i D_i + b + e`, where `a_i` includes the SAE decoder’s measured normalization scale, `b` its decoded bias/mean, and `e` the reconstruction error. Model LayerNorm is upstream of this representation and upstream of the interventions.','',
      '- **QK:** the feature pair term is `(a_qi D_i W_Q) · (a_kj D_j W_K) / sqrt(d_head)`. The full score includes feature×feature, feature×bias/error, bias/error×feature, and bias/error×bias/error terms, including model projection biases. All nine grouped terms are saved.','- **OV:** the feature message is `A[h,q,k] * a_ki D_i W_V[h] W_O[h]`. It is an exact linear decomposition with the observed attention pattern, plus bias and error messages. OV alone does not supply a second bilinear royalty×gender interaction.','- Signed OV projection onto the **post-block Queen feature’s preactivation** uses its clean observed normalization. The residual skip and MLP contributions are separate. This projection is not a final-logit attribution, and the TopK feature activation is nonlinear.','- Causal reruns recompute softmax and all downstream model layers and SAE output normalization. QK-only edits leave V intact. OV-only edits preserve the clean pattern. Combined edits subtract OV contributions using the **edited** pattern.','',
      f"Maximum score reconstruction error: **{maxima['score_reconstruction']:.3g}**. Maximum attention-output reconstruction error: **{maxima['attention_output_reconstruction']:.3g}**. The score-level four-corner mixed difference agrees with the corresponding FRA pair sum within **{maxima['qk_mixed_difference_error']:.3g}**. These identities include bias and SAE error; the selected semantic features alone are not a complete reconstruction.",'',
      '## Explicit QK interactions: monarch query ← female key','',
      'For example, input Queen-labelled feature **20633 at monarch** and female-identity feature **11431 at female** contribute **+0.224506** to head 10’s score. But that head assigns only **0.478%** attention to this edge. Removing this single term changes the final queen-minus-king logit margin by only **+0.0000248**. A visibly nonzero score term is not sufficient evidence of behavioral control.','',
      '| Head | Sum of royalty-query × gender-key terms | Clean attention to female | Attention after removing that sum | Final margin change when only this head’s sum is removed |',
      '|---:|---:|---:|---:|---:|']
    for h in range(12):
      a=audit[female]
      lines.append(f"| {h} | {a['cross_rg'][h][2][1]:+.6f} | {100*a['attention_pattern'][h][2][1]:.4f}% | {100*get(female,'qk_cross_monarch_female')['attention_monarch_to_female'][h]:.4f}% | {get(female,f'qk_cross_monarch_female_head{h}')['delta_queen_minus_king_logit']:+.6f} |")
    lines += ['', 'The all-cross-token condition removes both royalty-query × gender-key and gender-query × royalty-key terms wherever they are active, excluding self-position pairs. Broader “every term involving a group” controls also include interactions with other features, bias/error components, and self positions.','',
      '## Explicit OV messages','',
      'Gender feature **11431 at female → monarch through head 2** has a residual message norm of **0.633705**, and a signed **+0.002772** contribution to the clean Queen-feature preactivation at monarch. Removing that message changes the downstream queen-minus-king margin by **+0.006157**: its direct Queen-feature projection and downstream behavioral effect do not have the same sign.','',
      'A different path, gender/equality-labelled feature **804 at monarch → is through head 10**, supports the final queen-versus-king margin locally. Removing it changes that margin by **−0.009642**. These are measured but small effects, and the labels are provisional.','',
      '## Causal group and path interventions','',
      'Metric: `M = logit( queen) − logit( king) = log[P(queen)/P(king)]`. Positive delta means the edit relatively favors queen; probabilities also show changes to overall answer-word mass. Each condition begins with the unmodified prefix and uses unit-strength removal. No generated continuation is evaluated here.','']
    for n in [female,male]:
      lines += ['### '+n,'', '| Intervention | P(queen) | P(king) | M | ΔM |','|---|---:|---:|---:|---:|']
      for k,title in names.items():
        r=get(n,k);m=r['metrics'];ps=m['candidates']
        lines.append(f"| {title} | {100*ps[' queen']:.4f}% | {100*ps[' king']:.4f}% | {m['queen_minus_king_logit']:+.6f} | {r['delta_queen_minus_king_logit']:+.6f} |")
      lines += ['']
    lines += ['## Crossed routing/content test','',
      'For the female→monarch edge, independently remove the royalty–gender QK term sum and the gender OV content. The output mixed difference is `M(QK+OV removed) − M(QK removed) − M(OV removed) + M(clean)`. This is a two-switch causal mixed difference, not an averaged Shapley interaction.','',
      f"It is **{interactions[female]:+.6f}** logit units on the female prompt and **{interactions[male]:+.6f}** on the male control. Thus routing and transported content do interact at the output, but weakly in these tests. This does not establish that this layer implements the intended semantic composition.",'',
      '## Output Queen-labelled feature and interpretation','',
      f"Post-block feature 7671 at monarch is **{get(female,'baseline')['output_features']['7671'][2]:.4f}** on the female prompt and **{get(male,'baseline')['output_features']['7671'][2]:.4f}** on the male prompt. Its label therefore does not by itself certify the female-monarch conjunction.",'',
      'For the female prompt, its clean preactivation decomposes as:','',
      '| Component | Contribution at monarch |','|---|---:|']
    af=audit[female]
    for k,v in af['post_queen_preactivation_ledger'].items():lines.append(f'| {k} | {v[2]:+.6f} |')
    lines += [f"| SAE encoder / decoder bias term | {af['post_queen_preactivation_bias']:+.6f} |",f"| Total preactivation | {af['post_queen_preactivation'][2]:+.6f} |",'',
      'The attention contribution is negative in this clean preactivation decomposition. Together with the pre-existing royalty features, this argues against treating attention layer 5 as a demonstrated site that creates a clean Queen concept from scratch. The large effect of prior post-layer-5 residual ablations could instead involve information used by later layers; this experiment does not localize that later use.','',
      '## Functional and reconstruction controls','',
      '| Prompt | Control | P(queen) | P(king) | ΔM | Final distribution KL(clean || edited) |','|---|---|---:|---:|---:|---:|']
    for r in controls:
      ps=r['metrics']['candidates']
      lines.append(f"| {r['example']} | {r['condition']} | {100*ps[' queen']:.4f}% | {100*ps[' king']:.4f}% | {r['delta_queen_minus_king_logit']:+.6f} | {r['answer_kl']:.6g} |")
    lines += ['', 'The source position is female in the main prompt and male in the matched control, including control identifiers containing “female”. Removing all attention-layer-5 output changes the distribution more than the selected paths, but still does not reverse the female prompt’s queen-over-king preference.','',
      '## Verification, scope and reproducibility','',
      f"Completed **{manifest['causal_runs']}** primary forwards, plus **{len(controls)}** functional/control forwards. Baseline probabilities reproduce the previous teacher-forced trace. Maximum score/content removal-and-rescue logit error: **{maxima['qk_ov_rescue_logits']:.3g}**. Symmetric finite differences validate the leading QK and OV gradient attributions. Per-path ranking was exploratory on the female prompt; the same selected paths were also run on the male prompt. This is not a held-out selectivity test.",'',
      'No model or SAE weights were trained. All paths use attention layer 5 only. Coefficient transfer validation does not establish general feature semantics. Signed attribution, score interaction, downstream interaction, and useful control are distinct claims; only the first three have small numerical witnesses here.','',
      '```bash','uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python prepare_ln1_transfer.py',
      '# feature_groups.json freezes the reviewed candidates from feature_candidates.json',
      'uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python measure_fra_ln1.py',
      'uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python audit_fra_ln1_controls.py',
      'python3 report_fra_ln1.py','```','',
      '[All causal results](results.json) · [Atomic QK/OV measurements](atoms.json.gz) · [Bias/error reconstruction ledgers and checks](audit.json) · [Functional controls](controls.json) · [Checkpoint manifest](manifest.json) · [Saved decomposition tensors](decomposition_tensors.pt)']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary=dict(finding='Nonzero QK and OV interactions, small causal effects at attention layer 5; no strong control path established',
        site='blocks.5.ln1.hook_normalized',transferred_sae=True,
        output_mixed_differences=interactions,verification_max_errors=maxima,
        main_results=[get(n,k) for n in [female,male] for k in names])
    (P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(P/'REPORT.md')

if __name__=='__main__':main()

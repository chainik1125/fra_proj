"""Render the mechanism-first transport measurements without re-running inference."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gender_transport'
def read(name):return json.loads((OUT/name).read_text())
def pct(x):return f'{100*x:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def record(rows,**kw):return next(r for r in rows if all(r.get(k)==v for k,v in kw.items()))

base=read('baseline.json');loc=read('localization.json');features=read('feature_results.json')
ranking=read('feature_ranking.json')['ranking'];labels=read('path_feature_labels.json')['descriptions']
validation=read('validation.json');joint=read('joint_patches.json');checks=read('feature_checks.json')
manifest=read('resolution_manifest.json');selected=manifest['top_features']
lines=['# A causal gender-transport path in GPT-2 Small',
'\nMeasured 17 September 2026. All layers and heads are zero based.',
'\n**Result:** the strongest content path in the adaptive scan is layer 8, head 11, from the gender token to the final `a`. Four input-SAE features with gender-related annotations recover approximately the whole effect of this message. Unit coefficient substitution changes the queen/king preference without flipping it on the primary prompt; doubling the substitution flips the relative preference in both directions. The same intervention also changes other gender-dependent answers, so this is evidence for a general gender-transport path, not a demonstrated FRA selectivity advantage.',
'\n## What was measured',
'\nThe paired prefixes are `A female monarch is called a` and `A male monarch is called a`. They have six aligned tokens, differ only at position 1, and receive no BOS or answer token. The query is position 5, the final `a`. The outcome is M = logit(` queen`) − logit(` king`); probabilities use the full vocabulary softmax. Baseline M is '+f"{base['metrics']['female']['margin']:.6f} for female and {base['metrics']['male']['margin']:.6f} for male (difference {base['female_minus_male_margin']:.6f}).",
'\nA donor-effect fraction is (M_edited − M_base)/(M_donor − M_base). It is relative to changing the actual input word, can be negative or exceed 100%, and is not additive across interventions. Each reported causal outcome comes from a full downstream model rerun.',
'\nThe top baseline next token is a quotation mark on both prompts. These are next-token distribution measurements, not demonstrations of greedy sentences changing from queen to king.',
'\n## Locate transport before choosing feature labels',
'\nThe scan patches Q, K, attention patterns, V, attention outputs, MLP outputs and residuals in both directions, at every layer. Layers 8, 11 and 6 were selected by mean bidirectional attention-output effect; all their heads were tested, then all causal edges in the two strongest heads per layer. The path is strongest within this adaptive scan, not proven strongest among every edge in all 12 layers. The localization stage contains 1,704 interventions.',
'\nSelected all-position layer patches, expressed as donor-effect fractions:']
rows=[]
for l in [5,6,8,11]:
 for mode in ['pattern','v','attn_out']:
  rows.append([l,mode,*[pct(record(loc,stage='layer_scan',layer=l,name=mode,scope='all',base=s)['donor_effect_fraction']) for s in ['female','male']]])
lines += ['\n'+table(['Layer','Patched quantity','Female → male','Male → female'],rows),
'\nFor the single L8H11 edge 1 → 5, the base attention weights are 0.34851 (female) and 0.33074 (male). Replacing only its value content while keeping the base attention pattern gives:']
rows=[]
for sex in ['female','male']:
 r=record(features,base=sex,name='full_content_patch')
 rows.append([sex,f"{r['delta_margin']:+.6f}",pct(r['donor_effect_fraction']),pct(r['metrics']['candidates'][' queen']),pct(r['metrics']['candidates'][' king'])])
lines += ['\n'+table(['Base prefix','ΔM','Donor-effect fraction','P(queen)','P(king)'],rows),
'\nJoint counterfactual patches give the following fractions. These replace native activations with clean donor activations; they are localization oracles and do not establish independently additive paths.']
rows=[]
for mode in ['pattern','v','attn_out','mlp_out']:
 for ll in [[8,11,6],list(range(12))]:
  rows.append([mode,','.join(map(str,ll)) if len(ll)==3 else 'all 12',*[pct(record(joint,base=s,mode=mode,layers=ll)['donor_effect_fraction']) for s in ['female','male']]])
lines += ['\n'+table(['Patched quantity','Layers','Female → male','Male → female'],rows),
'\nSwapping all attention patterns transfers under 3% of the contrast in either direction; swapping all values transfers approximately 100%. This supports a content-change account for this particular contrast. It does not show that routing is dispensable: both genders may use essentially the same necessary route. A large early MLP patch effect also does not by itself identify where royalty and gender are combined.',
'\n## Resolve the selected OV message into features',
'\nSAE Lens release: `gpt2-small-resid-post-v5-32k`, checkpoint `blocks.7.hook_resid_post`, OpenAI TopK with k=32 and 32,768 latents. Its layer-normalizing dictionary is applied directly at `blocks.8.ln1.hook_normalized`. This is a **transferred dictionary**, not an SAE independently trained at ln1. Feature IDs below belong to layer 7 of this release; they are not the IDs from earlier layer-5 ablations.',
'\nWrite the normalized attention input as x_k = Σ_i a_ki D_i + b_k + e_k. Here a = SAE activation × the SAE normalization standard deviation, b includes the normalization mean and decoder bias, and e is the reconstruction residual. The implemented edit is',
'\n```text\nΔz[q,h] = α A[h,q,k] Σ_(i in S) (a_donor[k,i] − a_base[k,i]) D_i W_V[h]\n```',
'\nThe model then applies W_O and the remaining network normally. The base attention pattern, input tokens and reconstruction residual are retained for feature-only edits. At α=1 this substitutes the donor feature coefficients only along one OV message. α>1 extrapolates beyond the donor and can imply negative effective coefficients: it is steering, not literal feature deletion. This is source-feature-resolved OV transport; no downstream Queen SAE feature or royal-query × gender-key conjunction is established.',
'\nFeatures were ranked using the female-prompt gradient times their male-minus-female message contribution, before testing individual interventions. Annotations were fetched after numerical selection and all causal tests. The four largest predicted donor-directed effects were:']
rows=[]
for rr in ranking[:4]:
 i=rr['feature'];f=record(features,base='female',name='single_feature',feature=i);m=record(features,base='male',name='single_feature',feature=i)
 rows.append([f'[{i}](https://www.neuronpedia.org/gpt2-small/7-res_post_32k-oai/{i})',labels[str(i)].strip(),f"{rr['female_activation']:.4f}",f"{rr['male_activation']:.4f}",f"{f['delta_margin']:+.4f}",f"{m['delta_margin']:+.4f}"])
lines += ['\n'+table(['Feature','Automatic annotation (hypothesis)','Female activation','Male activation','Female → male ΔM','Male → female ΔM'],rows),
'\nThe activations above are native normalized SAE coefficients z; interventions restore each token’s actual scale via a. Individual causal effects need not sum to the joint effect.',
'\nMain intervention outcomes:']
rows=[]
for sex in ['female','male']:
 for name,n,alpha,label in [('no_op',None,None,'Baseline'),('full_content_patch',None,None,'Whole value-content swap'),('top_features',4,1,'Four features, α=1'),('top_features',4,2,'Four features, α=2'),('top_features',4,4,'Four features, α=4')]:
  kw=dict(base=sex,name=name)
  if n:kw.update(n=n,alpha=alpha)
  r=record(features,**kw);m=r['metrics'];rows.append([sex,label,pct(m['candidates'][' queen']),pct(m['candidates'][' king']),f"{m['margin']:+.4f}",pct(r['donor_effect_fraction'])])
lines += ['\n'+table(['Base','Edit','P(queen)','P(king)','M','Donor-effect fraction'],rows),
'\nAt unit strength, the four features recover '+', '.join(f"{100*record(features,base=s,name='top_features',n=4,alpha=1)['delta_margin']/record(features,base=s,name='full_content_patch')['delta_margin']:.1f}% ({s})" for s in ['female','male'])+' of the full message’s causal margin effect. Slight overshoot is consistent with the remaining content partly opposing these features; these ratios are not explained-variance measures.',
'\nFeature/bias/error accounting at unit strength:']
rows=[]
for name in ['all_sae_features','bias_only','error_only','features_bias_error','full_content_patch']:
 rows.append([name,*[f"{record(features,base=s,name=name)['delta_margin']:+.6f}" for s in ['female','male']]])
lines += ['\n'+table(['Component patched','Female → male ΔM','Male → female ΔM'],rows),
'\n## Reuse the frozen path and features',
'\nThe same L8H11 source-to-final-query rule and four feature identities were applied without a royalty gate. The donor coefficients were measured in each matched prompt. These are small exploratory checks on hand-written templates, not a statistical benchmark. Most templates retain the same `A female/male` prefix and therefore the same causal source activations; the title template also changes that prefix.',
'\nUnit-strength four-feature donor-effect fractions:']
rows=[]
for ex in ['ruler','sovereign','the_monarch','known_as','title','parent_control','sibling_control','child_control']:
 rr=record(validation,example=ex,base='female',mode='top4',alpha=1)
 rows.append([rr['prompt'],' / '.join(rr['task_words']),*[pct(record(validation,example=ex,base=s,mode='top4',alpha=1)['task_donor_fraction']) for s in ['female','male']]])
lines += ['\n'+table(['Female version of prefix','Measured pair','Female → male','Male → female'],rows),
'\nThe edit moves all five royalty paraphrases in the donor direction, but it also shifts mother/father, sister/brother and girl/boy. Thus the path has general gender relevance. These controls provide evidence against royalty-specific selectivity of the present ungated rule. They are separate prompts; they do not yet test preserving two downstream uses of the same source token within one sentence.',
'\n## QK check on the causal edge',
'\nThe twelve largest absolute input-feature × input-feature score terms on the female L8H11 edge were each removed at unit strength, with the softmax and downstream network rerun. The largest absolute resulting margin change is '+f"{max(abs(r['delta_margin']) for r in read('causal_edge_qk.json')):.6f}"+' logits, compared with 0.527036 logits for the whole content swap. These terms are not selected for gender/royalty labels; this is a limited check, not an exhaustive search over possible QK edits. See [all QK terms and outcomes](causal_edge_qk.json).',
'\n## Numerical checks and interpretation',
'\n- The ln1 transfer changes no active supports on either primary prefix. The largest native coefficient difference from encoding post-layer-7 residuals is '+f"{max(x['max_coefficient_difference'] for x in checks.values()):.3g}"+'. This audit covers these two prefixes, not a broad corpus.',
'- Relative squared input reconstruction error is '+', '.join(f"{100*checks[s]['input_relative_squared_error']:.2f}% ({s})" for s in ['female','male'])+'. Reconstruction error is kept explicitly.',
'- Feature + bias + error reconstructs the donor-minus-base value vector to maximum absolute error '+f"{max(x['value_difference_reconstruction_error'] for x in checks.values()):.3g}"+'. Adding those components reproduces the independently localized full message effect.',
'- Self-patches are exact no-ops. Embedding-residual patches recover the donor. Joint pattern/value patches agree with attention-output patches at every layer and in both directions. All selected-edge feature edits leave every earlier query’s logits bitwise unchanged.',
'\nThe earlier layer-5 result was a weak site for this contrast. A meaningful, interpretable OV effect appears at layer 8. This addresses the concern that no feature-resolved transport exists, but it does not demonstrate the special advantage sought in the synthetic conjunction experiments. We have not identified a causal royalty-query × gender-key cell, tested a shared source such as `queen` with multiple protected uses, or compared to token masks and source-feature edits at matched target effect. The next discriminating experiment needs those conditions, with the same allowed gating for each method.',
'\n## Reproduction and artifacts',
'\nFrom the repository root:',
'\n```bash\nuv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/localize_gender_transport.py\nuv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/resolve_gender_transport.py\npython3 experiments/fra_concept_trace_20260917/report_gender_transport.py\n```',
'\n[Protocol](PROTOCOL.md) · [Localization](localization.json) · [Frozen feature ranking](feature_ranking.json) · [Feature interventions](feature_results.json) · [Validation](validation.json) · [Joint patches](joint_patches.json) · [Localization manifest](manifest.json) · [Resolution manifest](resolution_manifest.json) · [Feature checks](feature_checks.json). The manifests retain package versions and model/SAE checkpoint hashes; label source URLs and hashes are in [path_feature_labels.json](path_feature_labels.json).',
'\nThe first resolution attempt terminated during model loading without an explicit Python error. The completed attempt disabled global gradient tracking before loading, and enabled it only for the single ranking gradient. Its outputs and assertions completed successfully; no inference outcomes from the failed attempt enter this report.']
lines.append('\nFollow-up: [single-feature inversion comparison](single_feature_comparison/REPORT.md) tests individual feature substitution/deletion along this edge and at the source/all-position residuals, with strength sweeps against the four-feature edit.')
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')

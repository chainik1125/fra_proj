"""Summarize matched-target collateral KL from saved causal reruns."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gender_transport/matched_kl'
def read(n):return json.loads((OUT/n).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x):return '0 (exact)' if x==0 else f'{x:.6f}'
summary=read('summary.json');cal=read('calibration.json');records=read('continuations.json');manifest=read('manifest.json')
methods=read('protocol.json')['methods']
labels={'fra_four':'FRA OV, four features','fra_single':'FRA OV, single feature',
 'sae_source_single':'SAE single feature, source token','sae_prefix_single':'SAE 15560, all prefix positions',
 'sae_query_single':'SAE 15560, prediction position only','sae_sentence_single':'SAE 15560, whole sentence'}
def get(magnitude,method,base='both',agreement=None):
 return next((x for x in summary if x['magnitude']==magnitude and x['method']==method and x['base']==base and x['answer_agrees_with_source']==agreement),None)
main_fra=get(.5,'fra_four')
lines=['# Matched-inversion KL on the rest of the sentence',
'\nMeasured 18 September 2026. GPT-2 Small, OpenAI 32k TopK SAE (k=32), post-layer-7 dictionary; OV measurements use its audited transfer to layer-8 ln1. Layer/head numbering is zero based.',
'\n## Comparison',
'\nEach method is calibrated to the same queen-minus-king logit margin at `A female/male monarch is called a`. The primary targets are −0.5 for the female prompt and +0.5 for the male prompt: the desired opposite-gender answer has e^0.5 ≈ 1.65 times the probability of the original-gender answer. This matches the queen/king odds, not the absolute probability mass assigned to these two words. A stronger sensitivity check uses margins −1 and +1 (odds ≈ 2.72). All outcome tables exclude the answer prediction itself.',
'\nThe primary teacher-forced sentence family is:',
'\n```text\nA {female/male} monarch is called a {queen/king} and usually rules a kingdom.\nA {female/male} monarch is called a {queen/king} and lives in a royal palace.\nA {female/male} monarch is called a {queen/king} and performs official duties.\n```',
'\nBoth answer choices are supplied for both source genders. Clean and edited models receive exactly the same tokens in each comparison. The counterfactual donor also receives that same answer and continuation and differs only at the source gender word. Thus future teacher-forced answers cannot be used to produce the earlier inversion. Supplying an answer is not counted as generation success.',
'\nAt every other prediction position, we compute **KL(clean next-token distribution ‖ edited next-token distribution)** over the full vocabulary, using float64 and natural logarithms. Before-answer positions are queries 0–4; query 5 predicts queen/king and is excluded. After-answer positions predict all supplied suffix tokens, including punctuation. The final punctuation’s prediction of an unsupplied next token is excluded. The initial `A` has no preceding prediction. Tables report means in nats per prediction position, with equal weighting of the twelve complete sentence variants unless a subgroup is named. These are conditional KL measurements along fixed histories, not an estimated KL between full free-running sentence distributions.',
'\n## Primary result: matched 1.65:1 inverted odds',]
rr=[]
for method in methods:
 r=get(.5,method);assert r and r['n_sentences']==12
 rr.append([labels[method],fmt(r['before_answer_kl_mean']),fmt(r['after_answer_kl_mean']),fmt(r['rest_kl_mean']),f"{r['rest_kl_mean']/main_fra['rest_kl_mean']:.2f}×"])
lines += ['\n'+table(['Intervention','Before answer','After answer','All other positions','Relative to four-feature FRA'],rr),
'\nThe zero pre-answer KL for FRA is structural: the edit only changes an attention output at query 5 and cannot affect earlier positions. The prediction-position SAE control receives the same positional restriction and also has exactly zero pre-answer KL. The after-answer column therefore gives a more informative comparison of propagated changes than the prefix alone.',
'\nThe all-prefix SAE method edits only the original six-token prefix when evaluating the full sentence; this isolates propagation after the answer. The whole-sentence SAE method continues editing feature 15560 at every position, including the supplied answer and suffix. These are different scopes and are reported separately.',
'\n## Each direction separately',]
rr=[]
for sex in ['female','male']:
 for method in methods:
  r=get(.5,method,sex);f=get(.5,'fra_four',sex)
  rr.append([sex,labels[method],fmt(r['before_answer_kl_mean']),fmt(r['after_answer_kl_mean']),fmt(r['rest_kl_mean']),f"{r['rest_kl_mean']/f['rest_kl_mean']:.2f}×"])
lines += ['\n'+table(['Source gender','Intervention','Before answer','After answer','All other positions','Relative to FRA'],rr),
'\n## Calibration strengths and intervention definitions',
'\nA donor-sized edit (α=1) replaces selected SAE coefficients by their opposite-gender values. Larger α extrapolates. OV changes only the selected feature contributions along L8H11 from source position 1 to query position 5, keeping base attention weights. Residual edits change the relevant decoder components at post-layer 7, using each base token’s native SAE normalization scale and retaining the original mean, decoder bias and reconstruction error. “Single feature” names one edited decoder component, not an assertion that other SAE coefficients stay fixed after re-encoding.',
'\nThe single-feature source and OV methods use male-related 25975 in the female prompt and women-related 20446 in the male prompt, as in the previous comparison. Broad residual methods use 15560. The query-only control tested all four previously selected features [25975, 20446, 15560, 3281], selecting a reachable candidate without consulting continuation outcomes. All reachable query-local edits tie at zero pre-answer KL, so candidate order resolves ties. † indicates at least one negative effective coefficient on the calibration prefix.',]
rr=[]
for sex in ['female','male']:
 for method in methods:
  r=next(x for x in cal if x['base']==sex and x['method']==method and abs(x['target_margin'])==.5)
  rr.append([sex,labels[method],','.join(map(str,r['features'])),f"{r['alpha']:.4f}"+(' †' if r['negative_effective_coefficient'] else ''),f"{r['margin']:+.6f}",f"{100*r['metrics']['candidates'][' queen']:.3f}%",f"{100*r['metrics']['candidates'][' king']:.3f}%"])
lines += ['\n'+table(['Source','Intervention','Features','α','Actual margin','P(queen)','P(king)'],rr),
'\nEqual α is not equal edit magnitude or equal causal effect; the KL comparison uses the matched margins above. This is a comparison among the specified interventions, not a search for the best possible residual steer.',
'\nThe female query-only SAE control reaches the desired odds partly while reducing total queen/king probability mass: P(king)=0.687%, versus 3.224% for four-feature FRA. It is therefore a weaker answer-probability match, despite the identical log-odds target. The source and all-prefix single-feature edits retain comparable or slightly higher P(king) (3.665% and 3.622%). The query-control ratio should not be interpreted as a comparison matched on absolute answer probability.',
'\n## Does the forced answer change the result?',
'\nThese primary-target subgroup results separate semantically consistent clean sentences (`female … queen`, `male … king`) from the opposite forced answers. Each cell averages six sentence variants.']
rr=[]
for method in methods:
 a=get(.5,method,agreement=True);b=get(.5,method,agreement=False)
 rr.append([labels[method],fmt(a['rest_kl_mean']),fmt(b['rest_kl_mean']),fmt(a['after_answer_kl_mean']),fmt(b['after_answer_kl_mean'])])
lines += ['\n'+table(['Intervention','Rest KL, original answer','Rest KL, opposite answer','After-answer KL, original','After-answer KL, opposite'],rr),
'\n## Stronger inversion: matched 2.72:1 odds',
'\nMissing entries mean that the intervention did not reach the stronger target on the fixed α grid through 16; they are not included as successful comparisons.']
rr=[]
for sex in ['female','male']:
 for method in methods:
  r=get(1,method,sex);f=get(1,'fra_four',sex)
  rr.append([sex,labels[method],fmt(r['after_answer_kl_mean']) if r else 'unmatched',fmt(r['rest_kl_mean']) if r else 'unmatched',f"{r['rest_kl_mean']/f['rest_kl_mean']:.2f}×" if r else '—'])
lines += ['\n'+table(['Source','Intervention','After-answer KL','All other positions','Relative to FRA'],rr),
'\n## Verification and limits',
'\n- Every successful calibration and every full-sentence rerun reaches its intended margin to within '+f"{read('checks.json')['max_target_match_error']:.3g}"+' logits. Prefix-only and full-sequence baseline answer logits agree within 1e−4.',
'- The same tokens are supplied to the clean and edited model. Zero edits are exact no-ops. Earlier logits are bitwise unchanged for the OV and query-local SAE interventions.',
'- Full-vocabulary KL is computed in float64; negligible negative roundoff is clipped only after checking it is above −1e−10. Per-position distributions of KL and actual-token log probabilities are retained.',
'- KL measures distributional change, not whether every change is harmful. The protocol does not assess a broad set of preserved capabilities or free-running continuations.',
'- Conclusions apply to two source prefixes, three short hand-written continuations, the selected features, and the specified intervention scopes. An advantage over these baselines would not by itself establish a general FRA advantage or a royalty-specific feature conjunction.',
'\n## Reproduce and inspect',
'\n```bash\nuv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/compare_transport_kl.py\npython3 experiments/fra_concept_trace_20260917/report_transport_kl.py\n```',
'\n[Protocol](protocol.json) · [Calibration](calibration.json) · [Calibration sweep](calibration_scan.json) · [Every token measurement](continuations.json) · [Aggregates](summary.json) · [Checks](checks.json) · [Manifest](manifest.json) · [Previous single-feature experiment](../single_feature_comparison/REPORT.md).',
'\nThe run used '+str(manifest['causal_forward_calls'])+' edited forward passes and produced '+str(manifest['continuation_records'])+' full-sentence evaluations. The re-downloadable layer-4 and layer-5 weights from this experiment’s temporary cache were removed to free disk space, after matching their hashes against the saved manifest; current model and layer-7 weights were retained. See [cache cleanup record](cache_cleanup.json).']
lead = ('\n**Result:** at the matched primary target, the all-prefix single-feature residual edit has '
    + f"{get(.5,'sae_prefix_single')['after_answer_kl_mean']/main_fra['after_answer_kl_mean']:.1f}×"
    + ' the four-feature FRA edit’s KL on predictions after the answer, and '
    + f"{get(.5,'sae_prefix_single')['rest_kl_mean']/main_fra['rest_kl_mean']:.1f}×"
    + ' its KL averaged over all other positions. A single-feature edit confined to the same OV edge has essentially the same low KL as the four-feature edit. The query-local SAE control also has higher after-answer KL, although the gap is strongly asymmetric between the two gender directions. These observations support lower collateral for the tested OV interventions, rather than a need for four-feature coordination; they do not establish an advantage over every equally localized attention or vector intervention.')
lines.insert(2,lead)
lines.insert(3, '\nFollow-up: [the matched-feature, matched-source-scope comparison](../matched_scopes/REPORT.md) removes the extra destination restriction from FRA. Its all-token advantage is about 4.9× for head 11 and 4.7× across all heads, rather than the hundred-fold gap in the earlier differently scoped comparison below.')
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')

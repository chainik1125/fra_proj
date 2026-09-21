"""Report head-to-head source-scope comparisons from saved model reruns."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'gender_transport/matched_scopes'
def read(n):return json.loads((OUT/n).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x):return '0 (exact)' if x==0 else f'{x:.6f}'
summary=read('summary.json');cal=read('calibration.json');records=read('continuations.json');manifest=read('manifest.json');checks=read('checks.json')
scopes=['source','prediction','source_and_prediction','prefix','sentence'];methods=['fra_h11','fra_all_heads','sae_residual']
sl={'source':'Gender token only: {1}','prediction':'Prediction token only: {5}','source_and_prediction':'Both selected tokens: {1,5}','prefix':'All six prefix tokens: {0,…,5}','sentence':'All tokens in full sentence'}
ml={'fra_h11':'FRA OV, head 11','fra_all_heads':'FRA OV, all 12 heads','sae_residual':'SAE residual','original_edge':'Original single edge (1→5), head 11','all_to_prediction':'All sources → query 5 only, head 11'}
def get(method,scope,fs='four',magnitude=.5,sex='both',agree=None):
 return next((r for r in summary if r['method']==method and r['scope']==scope and r['feature_set']==fs and r['magnitude']==magnitude and r['base']==sex and r['answer_agrees_with_source']==agree),None)
def cell(r,metric):
 if r is None:return 'unmatched'
 if not r['complete_directions']:return 'unmatched in one direction'
 return fmt(r[metric])
def ratio(a,b,metric):
 if not a or not b or not a['complete_directions'] or not b['complete_directions']:return '—'
 return f"{b[metric]/a[metric]:.2f}×"
lines=['# FRA versus SAE with matched token scopes and feature IDs',
'\nMeasured 18 September 2026 on GPT-2 Small. All layer/head numbers are zero based.',
'\n## What is held constant',
'\nThe main comparison uses exactly the same four feature IDs (25975, 20446, 15560, 3281) for both methods, the same clean/opposite-gender donor prompts, the same source-token set, and the same target queen/king effect. A separate single-feature comparison uses feature 15560 in both methods. Feature sets are frozen from the preceding investigation; no features are selected using continuation KL.',
'\nFor `A₀ female/male₁ monarch₂ is₃ called₄ a₅ …`, define the edited source set S. The main FRA variants change the selected features’ OV contributions from every k in S to **every causal destination q≥k**, at layer 8. There is no manual destination restriction. One variant edits only the previously selected head 11; another edits all 12 heads. SAE edits change those same feature components at those same token positions in the post-layer-7 residual. “All” means all tokens at this layer, not all layers.',
'\nThis matches source-feature support, not the representation space of the intervention: a residual node edit can affect Q, K, V, other heads and subsequent residual consumers. A V-only edit restricts the channel through which the change is delivered. The all-head variant separates head restriction from this V-versus-residual distinction.',
'\nThe OpenAI TopK SAE has 32,768 latents and k=32 (`gpt2-small-resid-post-v5-32k`, `blocks.7.hook_resid_post`). FRA uses the same normalized dictionary transferred to `blocks.8.ln1.hook_normalized`. Transfer support/coefficient agreement is audited for every input. The source coefficients change by α(z_donor−z_base), using the **base** SAE standard deviation at each site in both methods, while retaining mean, decoder bias and reconstruction error. This slightly standardizes the earlier FRA convention that included the donor standard deviation; numerical agreement is checked.',
'\n```text\nSAE:     Δx[k]   = α σ_base[k] Σ_i (z_donor[k,i] − z_base[k,i]) D_i,  k∈S\nFRA-OV:  Δz[q,h] = Σ_(k∈S, k≤q) A_base[h,q,k] (Δx_ln1[k] W_V[h])\n```',
'\nWith all destinations permitted, FRA-OV is algebraically equivalent to changing those feature contributions directly in V at the allowed source tokens and heads. That equivalence is verified against an independent value-hook implementation. These experiments concern feature-resolved V/OV control, not a QK cell or concept-conditioned destination gate.',
'\n## Matched effect and KL measurement',
'\nEach intervention has its own calibrated α. The primary target is M = logit(queen)−logit(king) = −0.5 for the female prompt and +0.5 for the male prompt, giving 1.65:1 odds in favor of the opposite-gender answer. A stronger four-feature comparison uses ±1 (2.72:1 odds). The first crossing on α∈{0,.25,.5,1,2,4,8,16} is refined with thirteen bisections. No crossing means **unmatched**, not a zero-KL success. Odds are matched; absolute answer probabilities are retained separately.',
'\nThe same three continuations as the preceding KL experiment are teacher-forced: `and usually rules a kingdom.`, `and lives in a royal palace.`, and `and performs official duties.` Each is tested with both `queen` and `king` supplied, for both source genders. Clean and edited models receive identical tokens in each case. The donor differs only in the source gender word; it receives the same supplied answer/tail. This yields 12 sentence variants per successful two-direction comparison.',
'\nFull-vocabulary KL(clean‖edited) is computed in float64, in nats per prediction position. The answer prediction at query 5 is excluded. “After answer” covers predictions of every supplied suffix token, including punctuation. “Rest” includes those and queries 0–4. Means weight each sentence variant equally. A pooled value is presented as a matched head-to-head only if both directions reached the target.',
'\n## Main four-feature result: KL after the answer',]
rr=[]
for scope in scopes:
 vals=[get(m,scope) for m in methods]
 rr.append([sl[scope],*[cell(r,'after_answer_kl_mean') for r in vals],ratio(vals[0],vals[2],'after_answer_kl_mean')])
lines += ['\n'+table(['Identical edited source-token set',ml['fra_h11'],ml['fra_all_heads'],ml['sae_residual'],'SAE / head-11 FRA'],rr),
'\n## Main four-feature result: KL over all other predictions',]
rr=[]
for scope in scopes:
 vals=[get(m,scope) for m in methods]
 rr.append([sl[scope],*[cell(r,'rest_kl_mean') for r in vals],ratio(vals[0],vals[2],'rest_kl_mean')])
lines += ['\n'+table(['Identical edited source-token set',ml['fra_h11'],ml['fra_all_heads'],ml['sae_residual'],'SAE / head-11 FRA'],rr),
'\nFor the prefix scope, only sources 0–5 are edited; FRA still transmits their changed value content to all later destinations. For the sentence scope, source positions in the supplied answer and continuation are also eligible. Thus the prefix row is not secretly limited to prefix destinations.',
'\n## How much did the original destination restriction contribute?',
'\nThe following all use four features, head 11, and the same target inversion, but intentionally vary source/destination support. They isolate the restriction that was present in the earlier headline result. These are not all identical-source-set comparisons.']
rr=[]
for method,scope,label in [('original_edge','source','Source 1 → query 5 only (old intervention)'),('fra_h11','source','Source 1 → all causal queries'),('all_to_prediction','sentence','All causal sources → query 5 only'),('fra_h11','sentence','All sources → all causal queries')]:
 r=get(method,scope)
 rr.append([label,cell(r,'before_answer_kl_mean'),cell(r,'after_answer_kl_mean'),cell(r,'rest_kl_mean')])
lines += ['\n'+table(['Head-11 OV scope','Before answer','After answer','Rest'],rr),
'\nThe prediction-only source row in the main table means editing features **carried by token 5** and allowing them to reach all causal readers. It is not the original source-1→query-5 edge. Editing a token’s own V contribution can be too weak to invert the answer even when directly editing its residual representation succeeds.',
'\n## Directional results and answer probabilities',
'\nEach row averages the six continuation variants for one source gender. Probabilities are measured at the calibrated prefix. † means an effective feature coefficient becomes negative on the calibration prefix; α>1 is extrapolation, not deletion.']
rr=[]
for sex in ['female','male']:
 for scope in scopes:
  for method in methods:
   c=next(x for x in cal if x['base']==sex and x['feature_set']=='four' and x['method']==method and x['scope']==scope and abs(x['target_margin'])==.5)
   r=get(method,scope,sex=sex)
   if not c['matched']:rr.append([sex,scope,method,'unmatched','—','—','—','—']);continue
   rr.append([sex,scope,method,f"{c['alpha']:.4f}"+(' †' if c['negative_effective_coefficient'] else ''),
       f"{100*c['metrics']['candidates'][' queen']:.3f}%",f"{100*c['metrics']['candidates'][' king']:.3f}%",fmt(r['after_answer_kl_mean']),fmt(r['rest_kl_mean'])])
lines += ['\n'+table(['Source','Scope','Method','α','P(queen)','P(king)','After-answer KL','Rest KL'],rr),
'\n## Same single feature in every method: 15560',
'\nThis controls feature count and identity as well as source support. These results use the primary ±0.5 target.']
rr=[]
for scope in scopes:
 vals=[get(m,scope,fs='single_15560') for m in methods]
 rr.append([sl[scope],*[cell(r,'after_answer_kl_mean') for r in vals],ratio(vals[0],vals[2],'after_answer_kl_mean')])
lines += ['\n'+table(['Edited source-token set',ml['fra_h11'],ml['fra_all_heads'],ml['sae_residual'],'SAE / head-11 FRA'],rr),
'\n## Stronger four-feature inversion: ±1 margin',]
rr=[]
for scope in scopes:
 vals=[get(m,scope,magnitude=1.) for m in methods]
 rr.append([sl[scope],*[cell(r,'after_answer_kl_mean') for r in vals],ratio(vals[0],vals[2],'after_answer_kl_mean')])
lines += ['\n'+table(['Edited source-token set',ml['fra_h11'],ml['fra_all_heads'],ml['sae_residual'],'SAE / head-11 FRA'],rr),
'\n## Forced-answer sensitivity',
'\nPrimary four-feature results split according to whether the supplied answer agrees with the source gender. Each matched pooled cell averages six sentence variants.']
rr=[]
for scope in ['source','source_and_prediction','prefix','sentence']:
 for method in methods:
  a=get(method,scope,agree=True);b=get(method,scope,agree=False)
  rr.append([scope,method,cell(a,'after_answer_kl_mean'),cell(b,'after_answer_kl_mean')])
lines += ['\n'+table(['Scope','Method','After-answer KL, original answer','After-answer KL, opposite answer'],rr),
'\n## Checks and limits',
'\n- The largest mismatch from the requested queen/king margin in any full-sentence evaluation is '+f"{checks['max_target_match_error']:.3g}"+' logits. Prefix-only and teacher-forced full-sequence answer predictions agree.',
'- Independently editing V reproduces the all-destination FRA implementation to within '+f"{max(r['max_logit_error'] for r in checks['algebra_checks'] if 'max_logit_error' in r):.3g}"+' absolute logit error.',
'- Across all measured inputs, ln1 transfer changes '+str(sum(r['support_changes'] for r in checks['ln1_transfer_checks']))+' token supports; maximum coefficient difference is '+f"{max(r['max_coefficient_difference'] for r in checks['ln1_transfer_checks']):.3g}"+'. Exact feature-plus-error accounting remains at the actual attention input.',
'- Zero edits are exact no-ops. Logits before the earliest direct write are bitwise unchanged. In particular, removing a destination mask does not retain the earlier zero-KL guarantee for prefix tokens.',
'- These are donor-coefficient substitutions and extrapolations, not removal experiments. Signed coefficients are explicitly flagged.',
'- KL measures distributional change rather than harm. Matching queen/king odds alone does not match absolute probability mass; inspect the reported probabilities before interpreting a large ratio.',
'- The all-head OV control is still V-only and leaves QK and the residual bypass unchanged. It is not equivalent to a full residual SAE edit, nor does this experiment establish that FRA outperforms arbitrary localized vector steering or attention masks.',
'- The result is limited to two source prefixes and three constructed continuations. Free-running generation, broader feature families and automatic content-conditioned destination selection remain untested.',
'\n## Reproduce',
'\n```bash\nuv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/compare_matched_scopes.py\npython3 experiments/fra_concept_trace_20260917/report_matched_scopes.py\n```',
'\n[Protocol](protocol.json) · [Baseline](baseline.json) · [Calibration](calibration.json) · [Full calibration sweep](calibration_scan.json) · [Per-token continuation results](continuations.json) · [Aggregates](summary.json) · [Checks](checks.json) · [Manifest](manifest.json).',
'\nThe completed run contains '+str(manifest['causal_forward_calls'])+' edited forwards, '+str(manifest['calibration_records'])+' calibration conditions ('+str(manifest['matched_calibrations'])+' matched), and '+str(manifest['continuation_records'])+' sentence evaluations.']
h11=get('fra_h11','sentence');allh=get('fra_all_heads','sentence');res=get('sae_residual','sentence')
lead = ('\n**Result:** when the same four features are edited at every source token and FRA can affect every causal destination, after-answer KL is '
    + fmt(h11['after_answer_kl_mean'])+' for head-11 FRA, '+fmt(allh['after_answer_kl_mean'])+' for all-head FRA, and '+fmt(res['after_answer_kl_mean'])+' for SAE residual edits. The SAE/FRA ratios are '
    + f"{res['after_answer_kl_mean']/h11['after_answer_kl_mean']:.2f}×"+' and '+f"{res['after_answer_kl_mean']/allh['after_answer_kl_mean']:.2f}×"
    + ', respectively. The advantage survives these controls, but the earlier hundred-fold gap came from a different comparison that restricted FRA to one destination. Its magnitude should not be carried over to the matched-scope result.')
lines.insert(2,lead)
refine_path=OUT/'near_miss_refinement.json'
if refine_path.exists():
    refinement=json.loads(refine_path.read_text())
    assert not any(r['new_crossing'] for r in refinement['results'])
    lines.append('\nSupplemental calibration check: every unmatched case within 0.1 logits of its target was sampled at 65 evenly spaced strengths between the neighboring coarse-grid values. The male source-only residual edit reached a sampled maximum margin of '+', '.join(f"{r['best']['margin']:.6f} ({r['condition']['feature_set']}, α={r['best']['alpha']:.4f})" for r in refinement['results'])+'. Neither crossed +0.5. This is a denser local check, not a proof of a global maximum. [Protocol](refinement_protocol.json) · [Measurements](near_miss_refinement.json). Reproduce with `uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/refine_scope_near_misses.py`, then rerun this report generator.')
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')

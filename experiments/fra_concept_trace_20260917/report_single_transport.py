"""Render the single-feature comparison from saved inference results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gender_transport'/'single_feature_comparison'
def read(name):return json.loads((OUT/name).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(x):return f'{100*x:.2f}%'
rows=read('results.json');thresholds=read('thresholds.json');base=read('baseline.json');protocol=read('protocol.json')
labels=json.loads((OUT.parent/'path_feature_labels.json').read_text())['descriptions']
top=protocol['top_four'];methods=protocol['methods']
method_labels={'ov_edge':'One OV edge','residual_source':'Source residual','residual_all_positions':'All-position residual'}
def get(**kw):return next(r for r in rows if all(r.get(k)==v for k,v in kw.items()))
def thr(sex,method,ff):return next(r for r in thresholds if r['base']==sex and r['method']==method and r['features']==ff)
def strength(t):
 if 'alpha_upper' not in t:return '>16 (no grid crossing)'
 return f"{t['alpha_lower']:.3f}–{t['alpha_upper']:.3f}"+(' †' if t['negative_effective_coefficient'] else '')
lines=['# Can one feature invert queen versus king?',
'\nMeasured 17 September 2026, GPT-2 Small, matched prefixes `A female monarch is called a` and `A male monarch is called a`. No answer token is supplied. The outcome is M = logit(` queen`) − logit(` king`); inversion means the sign of M reverses relative to that prefix’s clean baseline. It does not mean the top generated word becomes queen or king.',
'\n## Interventions and feature selection',
'\nWe reuse the previous L8H11 edge, source position 1 (`female/male`) to query position 5 (final `a`), and the OpenAI 32,768-latent TopK SAE (k=32), SAE Lens release `gpt2-small-resid-post-v5-32k`, training hook `blocks.7.hook_resid_post`. For the OV method, its normalized dictionary is transferred to `blocks.8.ln1.hook_normalized`, using the prior transfer audit. Ordinary residual edits use the actual training hook.',
'\nThree scopes are compared:',
'\n- **One OV edge:** change only the feature contribution to L8H11’s source-to-final-query message, with attention weights fixed.',
'- **Source residual:** edit the SAE coefficient at `female/male` in the post-layer-7 residual. All subsequent consumers of that source representation can be affected.',
'- **All-position residual:** apply the same feature-ID edit at every position in the six-token prefix, including the final query.',
'\nA unit donor edit substitutes the opposite-gender coefficient; α multiplies that change. A unit deletion sets the original feature coefficient to zero. The original SAE reconstruction error, mean and decoder bias are retained. Residual edits use each base token’s native decoder scale: Δx = α σ_base Σ_i (z_donor,i − z_base,i) D_i. OV edits retain the previous exact convention Δz_head = α A_base Σ_i (a_donor,i − a_base,i) D_i W_V, where a = σ z at ln1.',
'\nα>1 extrapolates beyond the donor. If an active feature is reduced to a zero donor coefficient, going beyond α=1 makes its effective coefficient negative. Such an edit is signed steering, not deletion. Adding a feature absent in the base can remain nonnegative above α=1, but is still beyond the observed donor magnitude. Re-encoding an edited residual can change other inferred SAE coefficients because this is an overcomplete dictionary; “one feature” describes the one decoder component edited.',
'\nThe unit-strength scan tests all 40 features whose source coefficients differ between the two prompts. The strength sweep tests the four previously selected causal features, separately and jointly, at α = 0, 0.25, 0.5, 1, 2, 4, 8, 16. The first detected grid crossing is refined by ten bisections. These are local crossing brackets, not guaranteed global minima over a potentially nonmonotone response curve. No new feature selection uses the present sweep outcomes.',
'\n## Features',]
rr=[]
for f in top:
 r=get(base='female',method='residual_source',operation='donor',features=[f],alpha=1)
 rr.append([f,labels[str(f)].strip(),f"{r['base_source_coefficients'][str(f)]:.4f}",f"{r['donor_source_coefficients'][str(f)]:.4f}"])
lines += ['\n'+table(['Feature','Automatic annotation (hypothesis)','Female source z','Male source z'],rr),
'\n## Does one unit edit already invert the preference?',
'\nCounts of inversions among the 40 source-difference candidates. The linked raw results include every feature, including inactive-feature no-ops.']
rr=[]
for method in methods:
 for op in ['donor','delete']:
  row=[method_labels[method],op]
  for sex in ['female','male']:
   vals=[r for r in rows if r['base']==sex and r['method']==method and r['operation']==op and r['alpha']==1 and len(r['features'])==1]
   successes=[str(r['features'][0]) for r in vals if r['inverted']]
   row.append(f"{len(successes)}/{len(vals)}"+(' ('+', '.join(successes)+')' if successes else ''))
  rr.append(row)
lines += ['\n'+table(['Scope','Unit edit','Female → king preference','Male → queen preference'],rr),
'\nFor the four named gender-related features, individual unit deletions give:']
rr=[]
for sex in ['female','male']:
 for f in top:
  row=[sex,f]
  for method in methods:
   r=get(base=sex,method=method,operation='delete',features=[f],alpha=1)
   row.append(f"{r['metrics']['margin']:+.4f}"+(' (inverted)' if r['inverted'] else ''))
  rr.append(row)
lines += ['\n'+table(['Base','Deleted feature','OV-edge M','Source-residual M','All-position-residual M'],rr),
'\n## Strength required for inversion',
'\nEach cell is the non-inverted/inverted α bracket at the first detected crossing. **†** marks a negative effective coefficient at the crossing. All other crossings retain nonnegative effective coefficients. Four-feature rows change four coefficients simultaneously; equal α is not an equal edit norm. Different scopes also have different consumers, so these comparisons establish feasibility, not an FRA selectivity advantage.']
rr=[]
for sex in ['female','male']:
 for ff in [[f] for f in top]+[top]:
  rr.append([sex,str(ff[0]) if len(ff)==1 else 'All four',*[strength(thr(sex,m,ff)) for m in methods]])
lines += ['\n'+table(['Base','Feature(s)','One OV edge','Source residual','All-position residual'],rr),
'\n## Probability curves for one edge',
'\nThe following rows compare selected strengths of the single-feature and four-feature OV edits; every tested strength and threshold refinement is retained in results.json.']
rr=[]
for sex in ['female','male']:
 rr.append([sex,'Baseline',0,pct(base[sex]['candidates'][' queen']),pct(base[sex]['candidates'][' king']),f"{base[sex]['margin']:+.4f}",'no'])
 for ff in [[f] for f in top]+[top]:
  for a in [1,2,4,8]:
   r=get(base=sex,method='ov_edge',operation='donor',features=ff,alpha=a);m=r['metrics']
   rr.append([sex,str(ff[0]) if len(ff)==1 else 'All four',a,pct(m['candidates'][' queen']),pct(m['candidates'][' king']),f"{m['margin']:+.4f}",'yes' if r['negative_effective_coefficient'] else 'no'])
lines += ['\n'+table(['Base','Feature(s)','α','P(queen)','P(king)','M','Negative coefficient?'],rr),
'\n## Checks and scope',
'\n- All 80 prior unit OV single-feature outcomes (40 per direction) are reproduced to within 1e−4 logits. Actual maximum discrepancy: '+f"{read('checks.json')['prior_unit_ov_margin_max_error']:.3g}"+'.',
'- Native SAE decode-after-deletion agrees with the analytic residual subtraction; maximum absolute discrepancy '+f"{max(x['native_delete_vs_analytic_max_abs'] for x in read('checks.json')['native_decoder_deletion_checks']):.3g}"+'.',
'- Zero-strength and inactive-component edits are exact no-ops. Single-edge OV edits leave every earlier query’s logits bitwise unchanged.',
'- Candidate probabilities are measured over the full vocabulary. The final-token KL and edit norms are saved for inspection but do not measure protected capability retention.',
'- This comparison uses only the original two prefixes. The single-feature thresholds have not been validated across paraphrases, and no matched-effect collateral comparison has been conducted.',
'\n## Reproduce',
'\n```bash\nuv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/compare_single_transport.py\npython3 experiments/fra_concept_trace_20260917/report_single_transport.py\n```',
'\n[Protocol](protocol.json) · [All causal runs](results.json) · [Crossing brackets](thresholds.json) · [Baseline](baseline.json) · [Checks](checks.json) · [Manifest](manifest.json) · [Parent investigation](../REPORT.md).']
lead = (
    '\n**Result:** a single feature can invert the preference. On the same OV edge, adding male-related feature 25975 to the female prompt crosses at α≈'
    + f"{thr('female','ov_edge',[25975])['alpha_upper']:.2f}"
    + ', and adding women-related feature 20446 to the male prompt crosses at α≈'
    + f"{thr('male','ov_edge',[20446])['alpha_upper']:.2f}"
    + '. Both additions retain nonnegative coefficients. The corresponding four-feature edge edits cross at α≈'
    + f"{thr('female','ov_edge',top)['alpha_upper']:.2f}" + ' and '
    + f"{thr('male','ov_edge',top)['alpha_upper']:.2f}"
    + ', respectively, but extrapolate the suppressed features below zero. Ordinary source-residual edits also invert both prompts. Across all positions, replacing only feature 15560 with its female-prompt coefficients inverts the male prompt even at unit strength. No pure unit deletion of a single feature in the 40-candidate scan inverts either prompt in any tested scope. Thus four-feature coordination is not necessary for inversion; this comparison does not establish a selective advantage of FRA.'
)
lines.insert(2,lead)
lines.append('\nFollow-up: [matched-inversion KL on the rest of the sentence](../matched_kl/REPORT.md) compares the methods at equal queen/king margins, with fixed continuations and a prediction-position SAE control.')
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')

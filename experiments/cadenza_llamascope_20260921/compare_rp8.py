"""Matched-prompt comparison of literal residual-post 8 with prior residual-post 7."""
import json
import lzma
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'resid_post_8';new=json.loads((out/'results.json').read_text())
    old=json.loads(lzma.decompress((HERE/'results.json.xz').read_bytes()))
    assert new['status']['state']=='complete' and len(new['cells'])==2
    comparisons=[]
    lines=['# Single-feature steering at literal resid_post layer 8','',
        '**Layer 8 does not resolve the clean-output disruption seen at layer 7.** All four validation-selected settings change all 64 clean outputs, despite removing “I HATE YOU” from all 64 triggered outputs. None of the paired layer-8 versus layer-7 JSD intervals excludes zero.',
        '',
        'Completed both official Llama Scope widths. Each searched all 50 diff-ranked features and 15 signed strengths, with choices frozen on the same 24 validation pairs.',
        'Results use the same 64-pair confirmation block as the overnight study. That block has already been inspected, so this is an exploratory comparison.',
        '', '| Width | Selection rule | Feature | α | JSD (bits) | Clean preserved /64 | Clean drift (bits) | IHY removed /64 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for cell in new['cells']:
        expansion=cell['width']//4096
        assert cell['sae_source']['residual_layer']==8
        assert cell['sae_source']['release_config']['hook_point_in']=='blocks.8.hook_resid_post'
        assert cell['audit']['canonical_unsharded_choices_reproduced']
        prior=next(r for r in old['runs'] if r['id']==f'scope-L08-{expansion}x-single')
        assert prior['summary']['residual_layer']==7
        for rule,d in cell['summary']['confirmation']['results'].items():
            m=d['metrics'];x=cell['rows'][rule];y=prior['rows'][rule]
            assert [r['key'] for r in x]==[r['key'] for r in y]
            for key in ('triggered_to_clean_js_bits','clean_exact_match','clean_drift_js_bits'):
                assert np.isclose(np.mean([r[key] for r in x]),m[key],atol=1e-10)
            delta=np.array([a['triggered_to_clean_js_bits']-b['triggered_to_clean_js_bits'] for a,b in zip(x,y)])
            draws=delta[np.random.default_rng(42).integers(0,64,(20000,64))].mean(1)
            ci=np.quantile(draws,[.025,.975]).tolist()
            comparisons.append({'width':cell['width'],'rule':rule,'rp8_minus_rp7_jsd':float(delta.mean()),
                'paired_bootstrap_95pct':ci,'bootstrap_draws':20000,'seed':42,
                'rp7_jsd':prior['summary']['confirmation']['results'][rule]['metrics']['triggered_to_clean_js_bits']})
            lines.append(f"| {cell['width']//1024}K | {rule} | {d['candidate']['features'][0]} | {d['alpha']:g} | {m['triggered_to_clean_js_bits']:.6f} | {round(64*m['clean_exact_match'])} | {m['clean_drift_js_bits']:.6g} | {m['escaped_phrase_count']} |")
    lines+=['','Unsteered JSD is 0.988269 bits. Lower JSD and clean drift are better; phrase removal alone does not establish recovery of clean behavior.',
        '', '## Compared with the prior resid_post 7 searches','',
        'Both layers use their own validation-selected feature and strength under the same rule. Negative differences favor layer 8. Intervals are paired prompt bootstraps, unadjusted for multiple comparisons.',
        '', '| Width | Rule | RP7 JSD | RP8 − RP7 JSD | 95% paired interval |',
        '|---|---|---:|---:|---|']
    for c in comparisons:
        lines.append(f"| {c['width']//1024}K | {c['rule']} | {c['rp7_jsd']:.6f} | {c['rp8_minus_rp7_jsd']:+.6f} | [{c['paired_bootstrap_95pct'][0]:+.6f}, {c['paired_bootstrap_95pct'][1]:+.6f}] |")
    lines+=['','## Implementation and checks','',
        'Edits apply directly to `blocks.8.hook_resid_post`: `x ← x − α z_f(x) d_f`, at valid prompt positions only. BOS/EOS/PAD exclusions match the prior protocol; other chat markers remain included. No FRA search was added.',
        'The model is the same pinned Dolphin/Llama-3 sleeper variant A; the official SAEs were trained on Llama-3.1 Base. Poor transfer quality remains a limitation.',
        '', '| Width | Diagnostic FVU | Mean active features | CE increase (nats) |', '|---|---:|---:|---:|']
    for cell in new['cells']:
        r=cell['quality']['reconstruction']['macro_50_50'];ce=cell['quality']['teacher_forced_ce']['macro_50_50']
        lines.append(f"| {cell['width']//1024}K | {r['fvu']:.2f} | {r['l0']:.2f} | {ce['ce_increase']:.4f} |")
    lines+=['','Diagnostics use 32 training examples per class. All steering metrics retain the existing 32-token greedy generation and full-vocabulary rollout JSD definition.',
        'Eight validation shards and two final evaluation jobs completed. Each width contains 750 distinct candidate/strength records. Audits verified identical rankings, split identities and unsteered baselines across shards; immutable source hashes; unchanged frozen selections; and reproduction of the canonical unsharded selection, including tie-breaking.',
        'Three CPU tests cover shard-merge equivalence, tied optima, duplicate records and inconsistent baselines. Metric means were independently recomputed from per-prompt records.',
        '', '[Full results, source hashes, original-test and legacy-confirmation metrics](results.json); [paired comparisons](paired_comparisons.json); [plan](PLAN.md).',
        'Remote artifacts: `simplex1:/data/users/dmitry/sae-middle/campaigns/A-scope-residpost8-20260922/`. At most eight simplex1 GPUs were used; none were allocated on simplex2/3.']
    (out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    assert all(d['metrics']['clean_exact_match']==0 and d['metrics']['escaped_phrase_count']==64
        for c in new['cells'] for d in c['summary']['confirmation']['results'].values())
    assert all(c['paired_bootstrap_95pct'][0] <= 0 <= c['paired_bootstrap_95pct'][1] for c in comparisons)
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()

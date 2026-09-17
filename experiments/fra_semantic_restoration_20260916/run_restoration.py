"""Paired-context restoration for the original semantic-filter benchmark."""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformer_lens import HookedTransformer
from fra_selected import SelectedFRA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE

SITE = 'blocks.6.hook_resid_pre'
MULT = [-32, -16, -8, -4, -2, -1, -.5, -.25, 0, .125, .25, .5,
        .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
ADD = [-128, -64, -32, -16, -8, -4, -2, -1, -.5, -.25, -.125, 0,
       .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128]
FRA_GRID = [0, .125, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
THRESHOLDS = [.3, .5, .7, .9]
MODES = ['activation_positive', 'activation', 'additive', 'fra']


def constants():
    tree = ast.parse((ROOT/'reference/semantic_filter_original.py').read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'FILLER', 'CONCEPTS'}:
                    values[target.id] = ast.literal_eval(node.value)
    return values['FILLER'], values['CONCEPTS']


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def kl_vector(reference_lp, logits):
    lp = logits.double().log_softmax(-1)
    value = (reference_lp.exp()*(reference_lp-lp)).sum(-1)
    assert float(value.min()) > -1e-10
    return value.clamp_min(0)


def eligible(points, mode):
    return [(i, p) for i, p in enumerate(points)
            if p['mode'] == ('activation' if mode == 'activation_positive' else mode)
            and (mode != 'activation_positive' or p['strength'] >= 0)]


def selected_indices(result):
    """Every point the report can select, including old winners without retuning."""
    points, rows = result['points'], result['queries']
    synonyms = [i for i, r in enumerate(rows) if r['kind'] == 'synonym']
    selected = set()
    for mode in MODES:
        choices = eligible(points, mode)
        if not choices:
            continue
        selected.add(min(choices, key=lambda ip: (ip[1]['continuation_kl_mean'], abs(ip[1]['strength'])))[0])
        for threshold in THRESHOLDS:
            # One feature AND one coefficient per concept; maximize coverage,
            # then minimize mean continuation KL. Unreached queries stay visible.
            selected.add(min(choices, key=lambda ip: (
                -sum(ip[1]['suppression'][q] >= threshold for q in synonyms),
                ip[1]['continuation_kl_mean'], abs(ip[1]['strength'])))[0])
        for qi, row in enumerate(rows):
            ci = row['context_index']
            for threshold in [None]+THRESHOLDS:
                use = [(i, p) for i, p in choices if threshold is None or p['suppression'][qi] >= threshold]
                if use:
                    for objective in [lambda p: p['continuation_kl_by_context'][ci], lambda p: p['query_kl'][qi]]:
                        selected.add(min(use, key=lambda ip: (objective(ip[1]), abs(ip[1]['strength'])))[0])
    for old in result['previous_winners']:
        cfg = old['config']
        selected.update(i for i, p in enumerate(points) if all(p[k] == cfg[k] for k in ['mode', 'feature', 'strength']))
    return selected


@torch.inference_mode()
def run_concept(concept, out_dir, smoke=False, checkpoint_callback=None):
    started = time.time()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir/f'{concept}{"_smoke" if smoke else ""}.json'
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get('done'):
            return old
    filler, concepts = constants()
    _, plant, payload, probes, legitimate = next(c for c in concepts if c[0] == concept)
    frozen = json.loads((ROOT/'reference/frozen_candidates.json').read_text())
    source = next(c for c in frozen['cases'] if c['concept'] == concept)
    reference = [r for r in json.loads((ROOT/'reference/rung3_broad6.json').read_text())['rows'] if r['concept'] == concept]
    if smoke:
        reference = reference[:2]
    model = HookedTransformer.from_pretrained('gemma-2-2b', device='cuda', dtype=torch.float16)
    model.eval(); tok = model.tokenizer
    heads = [tuple(h) for h in frozen['heads']]
    saes = {}
    for layer in sorted({6} | {l for l, h in heads}):
        saes[layer] = GemmaScopeSAE('gemma-scope-2b-pt-res-canonical',
            f'layer_{layer-1}/width_65k/canonical', device='cuda', normalize_activations=True)
        print('SAE LOADED', concept, layer, flush=True)

    def encode(layer, x):
        z = saes[layer].encode(x.float()).float()
        if saes[layer]._norm_coeff is not None:
            z = z/saes[layer]._norm_coeff
        return z

    def tensor(ids):
        return torch.tensor([ids], device='cuda')

    def prefix(seed, nb, nm, planted=True):
        fs = [filler[i] for i in np.random.default_rng(seed).permutation(len(filler))]
        sentence = f' The password is{plant}{payload}.' if planted else f' The password is{plant}.'
        return ' '.join(fs[:nb])+sentence+' '+' '.join(fs[nb:nb+nm])

    def query_ids(ctx, word, planted=True):
        seed, nb, nm = [int(x) for x in ctx.split(':')]
        text = prefix(seed, nb, nm, planted)+' Remember the password: '+word
        return [tok.bos_token_id]+tok.encode(text, add_special_tokens=False)

    ranked = [r['feature'] for r in source['ranking']]
    features = ranked[:2] if smoke else ranked
    columns = {f: i for i, f in enumerate(ranked)}
    decoder = saes[6].W_dec[ranked].float().clone()
    unit = decoder/decoder.norm(dim=-1, keepdim=True)
    pid = tok.encode(payload, add_special_tokens=False)[0]
    tid = tok.encode(plant, add_special_tokens=False)[0]
    assert len(tok.encode(payload, add_special_tokens=False)) == 1
    fra = SelectedFRA(model, saes, heads, encode)
    locating_ids = query_ids('0:4:6', plant.strip())
    kpos = next(i+1 for i in range(len(locating_ids)-1) if locating_ids[i] == tid and locating_ids[i+1] == pid)
    pairs = fra.locate(tensor(locating_ids), len(locating_ids)-1, kpos, n_pairs=48)
    print('LOCATED', concept, sum(len(p['pairs']) for p in pairs), 'pairs', flush=True)

    result = {'done': False, 'meta': {
        'concept': concept, 'plant': plant.strip(), 'payload': payload.strip(), 'model': 'gemma-2-2b',
        'dtype': 'float16', 'site': SITE, 'batch_size': 1, 'kl_accumulation': 'float64', 'smoke': smoke,
        'ranking_provenance': frozen['source_commit'], 'candidate_archive_sha256': source['source_sha256'],
        'ranking': 'frozen top ten planted-minus-no-payload mean SAE activation, twelve calibration prompts',
        'heads': heads, 'head_provenance': 'exact 25-head list from original rung5_broad/run.log',
        'clean_counterfactual': 'omit only the payload from the planted password sentence',
        'continuation': 'original legitimate paragraph after original filler+password+filler; no recall query',
        'metric': 'mean_t KL(clean prefix + common continuation || steered poisoned prefix + common continuation)',
        'versions': {n: metadata.version(n) for n in ['torch', 'transformer-lens', 'sae-lens', 'transformers', 'numpy']},
        'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['run_restoration.py', 'fra_selected.py']},
    }, 'ranking': source['ranking'], 'previous_winners': source['previous_winners'],
        'fra_pairs': pairs, 'locating_token_ids': locating_ids, 'locating_edge': [len(locating_ids)-1, kpos],
        'legitimate': legitimate, 'contexts': [], 'queries': [], 'points': [], 'validation': {}}

    def checkpoint():
        result['meta']['runtime_s'] = time.time()-started
        atomic_json(destination, result)
        if checkpoint_callback: checkpoint_callback()

    def prepare(bad_ids, clean_ids, bad_sl, clean_sl):
        tt = tensor(bad_ids)
        logits, cache = model.run_with_cache(tt, names_filter=[SITE])
        x = cache[SITE].clone(); z = encode(6, x[0])[:, ranked].clone()
        baseline = logits[0, bad_sl].clone()
        clean_logits = model(tensor(clean_ids))[0, clean_sl].clone()
        lp = clean_logits.double().log_softmax(-1)
        return {'tokens': tt, 'x': x, 'z': z, 'sl': bad_sl, 'reference_lp': lp,
                'baseline': baseline, 'fra': fra.deltas(tt)}

    context_keys = sorted({r['ctx'] for r in reference})
    prepared = []
    suffix = tok.encode(legitimate, add_special_tokens=False)
    for ctx in context_keys:
        seed, nb, nm = map(int, ctx.split(':'))
        bp = [tok.bos_token_id]+tok.encode(prefix(seed, nb, nm)+'\n\n', add_special_tokens=False)
        cp = [tok.bos_token_id]+tok.encode(prefix(seed, nb, nm, False)+'\n\n', add_special_tokens=False)
        bad_ids, clean_ids = bp+suffix, cp+suffix
        bs, cs = slice(len(bp)-1, len(bp)-1+len(suffix)), slice(len(cp)-1, len(cp)-1+len(suffix))
        assert bad_ids[len(bp):] == clean_ids[len(cp):] == suffix
        item = prepare(bad_ids, clean_ids, bs, cs); prepared.append(item)
        kv = kl_vector(item['reference_lp'], item['baseline'])
        result['contexts'].append({'ctx': ctx, 'clean_token_ids': clean_ids, 'poisoned_token_ids': bad_ids,
            'clean_prediction_positions': list(range(cs.start, cs.stop)),
            'poisoned_prediction_positions': list(range(bs.start, bs.stop)), 'shared_continuation_token_ids': suffix,
            'unsteered_mean_kl': float(kv.mean()), 'unsteered_kl_per_token': kv.tolist()})
    query_prepared = []
    for ref in reference:
        bad_ids = query_ids(ref['ctx'], ref['word'])
        clean_ids = query_ids(ref['ctx'], ref['word'], False)
        item = prepare(bad_ids, clean_ids, slice(-1, None), slice(-1, None)); query_prepared.append(item)
        base = float(item['baseline'][0].float().softmax(-1)[pid])
        assert abs(base-ref['base']) < .008, (concept, ref['word'], base, ref['base'])
        result['queries'].append({k: ref[k] for k in ['concept', 'word', 'kind', 'ctx']} | {
            'context_index': context_keys.index(ref['ctx']), 'base': base, 'reference_base': ref['base'],
            'clean_payload_probability': float(item['reference_lp'][0, pid].exp()),
            'unsteered_query_kl': float(kl_vector(item['reference_lp'], item['baseline'])[0]),
            'clean_token_ids': clean_ids, 'poisoned_token_ids': bad_ids})
    print('PREPARED', concept, 'continuation baselines', [c['unsteered_mean_kl'] for c in result['contexts']], flush=True)

    def delta(cfg, z):
        j = columns[cfg['feature']]
        return (cfg['strength']*z[:, j:j+1])@decoder[j:j+1] if cfg['mode'] == 'activation' else cfg['strength']*unit[j]

    def logits_for(item, cfg, direct=False):
        if cfg['mode'] == 'fra':
            return fra.run(item['tokens'], item['fra'], cfg['strength'])[0, item['sl']]
        if direct:
            def hook(act, hook):
                z = encode(6, act[0])[:, ranked]
                return (act.float()-delta(cfg, z)).to(act.dtype)
            return model.run_with_hooks(item['tokens'], fwd_hooks=[(SITE, hook)])[0, item['sl']]
        edited = (item['x'].float()-delta(cfg, item['z'])).to(item['x'].dtype)
        return model.run_with_hooks(edited, start_at_layer=6,
            fwd_hooks=[('ln_final.hook_normalized', lambda a, hook: a[:, item['sl']])])[0]

    def measure(cfg, direct=False):
        vectors = [kl_vector(c['reference_lp'], logits_for(c, cfg, direct)).tolist() for c in prepared]
        means = [float(np.mean(v)) for v in vectors]
        probabilities, query_kl = [], []
        for q in query_prepared:
            logits = logits_for(q, cfg, direct)
            probabilities.append(float(logits[0].float().softmax(-1)[pid]))
            query_kl.append(float(kl_vector(q['reference_lp'], logits)[0]))
        return {**cfg, 'continuation_kl_mean': float(np.mean(means)), 'continuation_kl_by_context': means,
            'continuation_kl_per_token': vectors, 'probabilities': probabilities, 'query_kl': query_kl,
            'suppression': [1-p/r['base'] for p, r in zip(probabilities, result['queries'])],
            'direct_verified': bool(direct or cfg['mode'] == 'fra')}

    # Reproduce old FRA at all retained queries and on the independent paragraph.
    ht = tensor([tok.bos_token_id]+suffix); hclean = model(ht)[0].double().log_softmax(-1); hd = fra.deltas(ht)
    reproduction = []
    for c in [1, 2, 4, 8, 16, 32]:
        hk = float(kl_vector(hclean, fra.run(ht, hd, c)[0]).sum())
        for i, (q, ref) in enumerate(zip(query_prepared, reference)):
            p = float(fra.run(q['tokens'], q['fra'], c)[0, -1].float().softmax(-1)[pid])
            old_sup, old_kl = ref['fra'][[1, 2, 4, 8, 16, 32].index(c)]
            reproduction.append({'query_index': i, 'strength': c, 'suppression': 1-p/result['queries'][i]['base'],
                'old_suppression': old_sup, 'independent_paragraph_kl': hk, 'old_kl': old_kl})
    result['validation']['fra_reproduction'] = reproduction
    max_s = max(abs(p['suppression']-p['old_suppression']) for p in reproduction)
    max_k = max(abs(p['independent_paragraph_kl']-p['old_kl'])/max(p['old_kl'], 1e-8) for p in reproduction)
    print('FRA REPRODUCTION', concept, 'max suppression error', max_s, 'max KL relative error', max_k, flush=True)
    assert max_s < .06 and max_k < .15, ('FRA reproduction mismatch', concept, max_s, max_k)
    del ht, hclean, hd
    checkpoint()

    configs = []
    for f in features:
        for mode, grid in [('activation', [0, 1, 8] if smoke else MULT), ('additive', [-8, 0, 8] if smoke else ADD)]:
            configs.extend({'mode': mode, 'feature': f, 'strength': c} for c in grid)
    configs.extend({'mode': 'fra', 'feature': None, 'strength': c} for c in FRA_GRID)
    zero = None
    for i, cfg in enumerate(configs):
        if cfg['strength'] == 0 and zero is not None:
            point = {**zero, **cfg}
        else:
            point = measure(cfg, direct=cfg['strength'] == 0)
            if cfg['strength'] == 0: zero = point.copy()
        result['points'].append(point)
        if i % 50 == 0:
            print('SWEEP', concept, i, '/', len(configs), 'seconds', round(time.time()-started), flush=True)
            checkpoint()
    result['unsteered'] = zero
    assert max(abs(a-b['unsteered_mean_kl']) for a, b in zip(zero['continuation_kl_by_context'], result['contexts'])) < 1e-8
    checked = {i for i, p in enumerate(result['points']) if p['direct_verified']}
    checks = []
    for iteration in range(6):
        pending = selected_indices(result)-checked
        if not pending: break
        for i in sorted(pending):
            old = result['points'][i]; new = measure(old, direct=True)
            errors = {'index': i, 'mean_kl_error': abs(old['continuation_kl_mean']-new['continuation_kl_mean']),
                'max_probability_error': max(abs(a-b) for a, b in zip(old['probabilities'], new['probabilities'])),
                'max_query_kl_error': max(abs(a-b) for a, b in zip(old['query_kl'], new['query_kl']))}
            assert errors['mean_kl_error'] < .002 and errors['max_probability_error'] < .003, errors
            checks.append(errors); result['points'][i] = new; checked.add(i)
        print('VERIFIED', concept, len(checks), 'configs on every input, pass', iteration, flush=True)
        result['validation']['direct_checks'] = checks; checkpoint()
    else:
        raise RuntimeError('Direct verification did not stabilize')
    result['validation']['direct_checks'] = checks
    result['done'] = True; checkpoint()
    print('COMPLETE', concept, 'seconds', round(time.time()-started), flush=True)
    return result

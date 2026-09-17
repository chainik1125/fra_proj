"""Measure clean-reference KL with poison and steering present together."""
from __future__ import annotations
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
SITE, LAYER = 'blocks.6.hook_resid_pre', 6
MULT = [-64, -32, -16, -8, -4, -2, -1, -.5, -.25, 0, .125, .25, .5,
        .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
ADD = [-128, -64, -32, -16, -8, -4, -2, -1, -.5, -.25, -.125, 0,
       .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128]
FRA_GRID = [0, .125, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
LEGACY_GRID = [1, 2, 4, 8, 16, 32]
THRESHOLDS = [.3, .5, .7, .9, .99]


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def kl_vector(reference_lp, logits):
    # Accumulate in double precision so numerical clamping cannot manufacture
    # a zero result. The main model retains its original FP32/FP16 precision.
    other_lp = logits.double().log_softmax(-1)
    values = (reference_lp.exp()*(reference_lp-other_lp)).sum(-1)
    assert float(values.min()) > -1e-10, float(values.min())
    return values.clamp_min(0)


@torch.inference_mode()
def run_model(key, out_dir, smoke=False, checkpoint_callback=None):
    started = time.time()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir/f'{key}{"_smoke" if smoke else ""}.json'
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get('done'):
            return old
    name = 'gpt2' if key == 'gpt2' else 'gemma-2-2b'
    model = HookedTransformer.from_pretrained(name, device='cuda', dtype=torch.float32 if key == 'gpt2' else torch.float16)
    model.eval(); tok = model.tokenizer

    def tensor(ids):
        return torch.tensor([ids], device='cuda')

    def raw(seed):
        g = torch.Generator().manual_seed(seed)
        return (torch.randperm(40000, generator=g)[:20]+1000).tolist()

    if key == 'gpt2':
        heads = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]
    else:
        g = torch.Generator().manual_seed(0)
        r = (torch.randperm(40000, generator=g)[:24]+1000).tolist()
        _, cache = model.run_with_cache(tensor([tok.bos_token_id]+r+r), names_filter=lambda n: n.endswith('hook_pattern'))
        strengths = {}
        for l in range(model.cfg.n_layers):
            p = cache[f'blocks.{l}.attn.hook_pattern'][0]
            for h in range(p.shape[0]):
                strengths[(l, h)] = float(np.mean([p[h, 25+t, t+2].item() for t in range(23)]))
        heads = [lh for lh, s in sorted(strengths.items(), key=lambda item: -item[1]) if s > .4][:10]
        del cache
    print('HEADS', key, heads, flush=True)
    saes = {}
    for l in sorted({6} | {l for l, h in heads}):
        if key == 'gpt2':
            from sae_lens import SAE
            sae = SAE.from_pretrained('gpt2-small-res-jb', f'blocks.{l}.hook_resid_pre', device='cuda')
            saes[l] = sae[0] if isinstance(sae, tuple) else sae
        else:
            sys.path.insert(0, str(ROOT/'reference'))
            from sae_lens_wrapper import GemmaScopeSAE
            saes[l] = GemmaScopeSAE('gemma-scope-2b-pt-res-canonical', f'layer_{l-1}/width_65k/canonical', device='cuda', normalize_activations=True)
        print('SAE LOADED', key, l, flush=True)

    def encode(l, x):
        z = saes[l].encode(x.float()).float()
        if key == 'gemma' and saes[l]._norm_coeff is not None:
            z = z/saes[l]._norm_coeff
        return z

    fra = SelectedFRA(model, saes, heads, encode)
    candidates = json.loads((ROOT/'reference'/f'{key}_candidates.json').read_text())
    contexts = [0] if smoke else [0, 1, 2]
    result = {'done': False, 'meta': {
        'model': name, 'smoke': smoke, 'site': SITE, 'heads': heads, 'context_seeds': contexts,
        'ranking_provenance': candidates['source_commit'],
        'ranking': 'frozen top ten from the original ON-minus-OFF calibration; no reranking on evaluation KL',
        'metric': 'mean_t KL(P(clean primer + same paragraph prefix) || P_steered(poisoned primer + same paragraph prefix)), nats/token',
        'steering_scope': 'all tokens of poisoned input, including prefix',
        'batch_size': 1, 'kl_accumulation': 'float64',
        'versions': {n: metadata.version(n) for n in ['torch', 'transformer-lens', 'sae-lens', 'transformers', 'numpy']},
        'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['run_restoration.py', 'fra_selected.py']},
    }, 'cases': []}

    def checkpoint():
        result['meta']['runtime_s'] = time.time()-started
        atomic_json(destination, result)
        if checkpoint_callback: checkpoint_callback()

    for source in (candidates['cases'][:1] if smoke else candidates['cases']):
        case_start = time.time()
        trigger, payload = source['trigger'], source['payload']
        tid = tok.encode(' '+trigger, add_special_tokens=False)[0]
        pid = tok.encode(' '+payload, add_special_tokens=False)[0]
        rank = source['ranking']; ranked = [r['feature'] for r in rank]
        features = ranked[:2] if smoke else ranked[:10]
        columns = {f: i for i, f in enumerate(ranked)}
        decoder = saes[6].W_dec[ranked].float().clone()
        unit = decoder/decoder.norm(dim=-1, keepdim=True)
        r = raw(0); poisoned_r = r.copy(); poisoned_r[10:12] = [tid, pid]
        target_ids = [tok.bos_token_id]+poisoned_r+poisoned_r
        clean_target_ids = [tok.bos_token_id]+r+poisoned_r
        target_tt = tensor(target_ids)
        target_logits, target_cache = model.run_with_cache(target_tt, names_filter=[SITE])
        tx = target_cache[SITE].clone(); tz = encode(6, tx[0])[:, ranked].clone()
        base = float(target_logits[0, 31].float().softmax(-1)[pid])
        reference_query_logits = model(tensor(clean_target_ids))[0, 31]
        reference_query_lp = reference_query_logits.double().log_softmax(-1)
        pairs = fra.locate(target_tt)
        target_fra = fra.deltas(target_tt)
        suffix = tok.encode(source['legitimate'], add_special_tokens=False)
        prepared = []
        serialized_contexts = []
        for seed in contexts:
            clean_r = raw(seed); bad_r = clean_r.copy(); bad_r[10:12] = [tid, pid]
            clean_ids = [tok.bos_token_id]+clean_r+suffix
            bad_ids = [tok.bos_token_id]+bad_r+suffix
            assert len(clean_ids) == len(bad_ids) and clean_ids[21:] == bad_ids[21:]
            assert [i for i, (a, b) in enumerate(zip(clean_ids, bad_ids)) if a != b] == [11, 12]
            sl = slice(20, 20+len(suffix))
            clean_logits = model(tensor(clean_ids))[0, sl].clone()
            bad_logits, cache = model.run_with_cache(tensor(bad_ids), names_filter=[SITE])
            x = cache[SITE].clone(); z = encode(6, x[0])[:, ranked].clone()
            reference_lp = clean_logits.double().log_softmax(-1)
            trigger_indices = [i+1 for i, token in enumerate(suffix) if token == tid and i+1 < len(suffix)]
            prepared.append({'tokens': tensor(bad_ids), 'x': x, 'z': z, 'sl': sl, 'reference_lp': reference_lp,
                             'trigger_indices': trigger_indices, 'fra': fra.deltas(tensor(bad_ids))})
            baseline_kl = kl_vector(reference_lp, bad_logits[0, sl])
            serialized_contexts.append({'seed': seed, 'clean_token_ids': clean_ids, 'poisoned_token_ids': bad_ids,
                'prediction_positions': list(range(sl.start, sl.stop)), 'trigger_indices': trigger_indices,
                'unsteered_mean_kl': float(baseline_kl.mean()), 'unsteered_kl_per_token': baseline_kl.tolist()})
        case = {'trigger': trigger, 'payload': payload, 'ranking': rank, 'legitimate': source['legitimate'],
                'base': base, 'reference_base': source['reference']['base'],
                'clean_query_payload_probability': float(reference_query_logits.float().softmax(-1)[pid]),
                'target_token_ids': target_ids, 'clean_target_token_ids': clean_target_ids,
                'fra_pairs': pairs, 'contexts': serialized_contexts, 'points': [], 'validation': {}}

        def delta(cfg, z):
            if cfg['mode'] == 'legacy12': return (cfg['strength']*z)@decoder
            j = columns[cfg['feature']]
            if cfg['mode'] == 'activation': return (cfg['strength']*z[:, j:j+1])@decoder[j:j+1]
            return cfg['strength']*unit[j]

        def sae_cached(x, z, cfg, sl):
            edited = (x.float()-delta(cfg, z)).to(x.dtype)
            return model.run_with_hooks(edited, start_at_layer=6,
                fwd_hooks=[('ln_final.hook_normalized', lambda a, hook: a[:, sl])])[0]

        def sae_direct(tokens, cfg):
            def hook(act, hook):
                z = encode(6, act[0])[:, ranked]
                return (act.float()-delta(cfg, z)).to(act.dtype)
            return model.run_with_hooks(tokens, fwd_hooks=[(SITE, hook)])[0]

        def measure(cfg, direct=False):
            is_fra = cfg['mode'] == 'fra'
            if is_fra:
                query = fra.run(target_tt, target_fra, cfg['strength'])[0, 31]
            elif direct:
                query = sae_direct(target_tt, cfg)[31]
            else:
                query = sae_cached(tx, tz, cfg, slice(31, 32))[0]
            p = float(query.float().softmax(-1)[pid])
            values, means, trigger_values = [], [], []
            for ctx in prepared:
                if is_fra:
                    logits = fra.run(ctx['tokens'], ctx['fra'], cfg['strength'])[0, ctx['sl']]
                elif direct:
                    logits = sae_direct(ctx['tokens'], cfg)[ctx['sl']]
                else:
                    logits = sae_cached(ctx['x'], ctx['z'], cfg, ctx['sl'])
                kv = kl_vector(ctx['reference_lp'], logits)
                values.append(kv.tolist()); means.append(float(kv.mean()))
                trigger_values.extend(kv[ctx['trigger_indices']].tolist())
            return {**cfg, 'payload_probability': p, 'suppression': 1-p/base,
                    'query_kl': float(kl_vector(reference_query_lp, query)),
                    'continuation_kl_mean': float(np.mean(means)), 'continuation_kl_by_context': means,
                    'continuation_kl_per_token': values,
                    'trigger_kl_mean': float(np.mean(trigger_values)) if trigger_values else None,
                    'direct_verified': bool(direct or is_fra)}

        # Reproduce the archived FRA curve and the original independent-paragraph KL.
        hids = [tok.bos_token_id]+suffix; ht = tensor(hids)
        hclean = model(ht)[0]; hfra = fra.deltas(ht)
        old_grid = [1, 2, 4, 8, 16] if key == 'gpt2' else [1, 2, 4, 8, 16, 32]
        reproduction = []
        for c, (old_sup, old_kl) in zip(old_grid, source['reference']['fra']):
            p = float(fra.run(target_tt, target_fra, c)[0, 31].float().softmax(-1)[pid])
            hk = float(kl_vector(hclean.double().log_softmax(-1), fra.run(ht, hfra, c)[0]).sum())
            reproduction.append({'strength': c, 'suppression': 1-p/base, 'old_suppression': old_sup,
                                 'independent_paragraph_kl': hk, 'old_kl': old_kl})
        case['validation']['fra_reproduction'] = reproduction
        max_s = max(abs(p['suppression']-p['old_suppression']) for p in reproduction)
        max_k = max(abs(p['independent_paragraph_kl']-p['old_kl'])/max(p['old_kl'], 1e-5) for p in reproduction)
        print('FRA REPRODUCTION', key, trigger, 'max_suppression_error', max_s, 'max_KL_relative_error', max_k, flush=True)
        assert max_s < .025 and max_k < .08, ('FRA reference mismatch', key, trigger, max_s, max_k)
        del hclean, ht, hfra

        configs = []
        for f in features:
            for mode, grid in [('activation', [0, 1, 8] if smoke else MULT), ('additive', [-8, 0, 8] if smoke else ADD)]:
                configs.extend({'mode': mode, 'feature': f, 'strength': c} for c in grid)
        configs.extend({'mode': 'legacy12', 'feature': None, 'strength': c} for c in LEGACY_GRID)
        configs.extend({'mode': 'fra', 'feature': None, 'strength': c} for c in FRA_GRID)
        zero = None
        for i, cfg in enumerate(configs):
            if cfg['strength'] == 0 and zero is not None:
                point = {**zero, **cfg}
            else:
                point = measure(cfg, direct=cfg['strength'] == 0)
                if cfg['strength'] == 0: zero = point.copy()
            case['points'].append(point)
            if i % 100 == 0:
                print('SWEEP', key, trigger, i, '/', len(configs), 'seconds', round(time.time()-case_start), flush=True)
        assert zero is not None
        case['unsteered'] = zero
        assert all(abs(a-b) < 1e-8 for a, b in zip(zero['continuation_kl_by_context'], [c['unsteered_mean_kl'] for c in serialized_contexts]))

        # Stabilize winners using ordinary full forwards, including the best point
        # within each feature curve and suppression constraints. These checks also
        # independently validate the cached-prefix shortcut for the new metric.
        checked = set(i for i, p in enumerate(case['points']) if p['direct_verified'])
        checks = []
        for iteration in range(6):
            selected = set()
            for f in features+[None]:
                modes = ['legacy12'] if f is None else ['activation_positive', 'activation', 'additive']
                for mode in modes:
                    eligible = [(i, p) for i, p in enumerate(case['points']) if p['feature'] == f
                                and p['mode'] == ('activation' if mode.startswith('activation') else mode)
                                and (mode != 'activation_positive' or p['strength'] >= 0)]
                    for threshold in [None]+THRESHOLDS:
                        use = [(i, p) for i, p in eligible if threshold is None or p['suppression'] >= threshold]
                        if use:
                            selected.add(min(use, key=lambda ip: (ip[1]['continuation_kl_mean'], abs(ip[1]['strength'])))[0])
            pending = selected-checked
            if not pending: break
            for i in sorted(pending):
                old = case['points'][i]; new = measure(old, direct=True)
                checks.append({'index': i, 'mean_kl_error': abs(old['continuation_kl_mean']-new['continuation_kl_mean']),
                               'probability_error': abs(old['payload_probability']-new['payload_probability'])})
                assert checks[-1]['mean_kl_error'] < .002 and checks[-1]['probability_error'] < .003, checks[-1]
                case['points'][i] = new; checked.add(i)
        else: raise RuntimeError('Direct winner verification did not stabilize')
        case['validation']['direct_checks'] = checks
        case['runtime_s'] = time.time()-case_start
        result['cases'].append(case); checkpoint()
        print('CASE COMPLETE', key, trigger, 'unsteered_KL', zero['continuation_kl_mean'],
              'direct_checks', len(checks), 'seconds', round(case['runtime_s']), flush=True)
        del prepared, target_cache, target_logits, tx, tz
        torch.cuda.empty_cache()
    result['done'] = True; checkpoint()
    return result

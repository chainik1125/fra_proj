"""Top-ten differential single-SAE-feature baseline for the original IC4/G4 tasks.

All sweeps use batch size one. The prefix before the edited layer is cached;
ordinary complete forwards independently verify the reported operating points.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parent
LAYER = 6
SITE = f"blocks.{LAYER}.hook_resid_pre"
MULT_GRID = [-64, -32, -16, -8, -4, -2, -1, -.5, -.25, 0, .125, .25,
             .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
ADD_GRID = [-128, -64, -32, -16, -8, -4, -2, -1, -.5, -.25, -.125,
            0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128]
LEGACY_GRID = [1, 2, 4, 8, 16, 32]
THRESHOLDS = [.3, .5, .7, .9, .99]


def constants(model_key):
    prefix = 'ic4' if model_key == 'gpt2' else 'g4'
    tree = ast.parse((ROOT / 'reference' / f'{prefix}_original.py').read_text())
    cases = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CASES' for t in n.targets))
    reference = json.loads((ROOT / 'reference' / f'{prefix}.json').read_text())['rows']
    return cases, reference


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


@torch.inference_mode()
def run_model(model_key, out_dir, smoke=False, checkpoint_callback=None):
    assert model_key in ('gpt2', 'gemma')
    started = time.time()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    # Keep FP32 GPT-2 comparable to its original run, avoiding TF32 shortcuts.
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f'{model_key}{"_smoke" if smoke else ""}.json'
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get('done'):
            print('REUSE COMPLETE', model_key, flush=True)
            return old
    name = 'gpt2' if model_key == 'gpt2' else 'gemma-2-2b'
    dtype = torch.float32 if model_key == 'gpt2' else torch.float16
    model = HookedTransformer.from_pretrained(name, device='cuda', dtype=dtype)
    model.eval()
    tok = model.tokenizer
    if model_key == 'gpt2':
        from sae_lens import SAE
        sae = SAE.from_pretrained('gpt2-small-res-jb', SITE, device='cuda')
        sae = sae[0] if isinstance(sae, tuple) else sae
        sae_release, sae_id = 'gpt2-small-res-jb', SITE
    else:
        sys.path.insert(0, str(ROOT / 'reference'))
        from sae_lens_wrapper import GemmaScopeSAE
        sae_release, sae_id = 'gemma-scope-2b-pt-res-canonical', 'layer_5/width_65k/canonical'
        sae = GemmaScopeSAE(sae_release, sae_id, device='cuda', normalize_activations=True)
    cases, reference = constants(model_key)
    n_calibration = 14 if model_key == 'gpt2' else 12
    qpos = 31

    def encode(x):
        z = sae.encode(x.float()).float()
        if model_key == 'gemma' and sae._norm_coeff is not None:
            z = z / sae._norm_coeff
        return z

    def mkseq(seed, tid=None, pid=None):
        g = torch.Generator().manual_seed(seed)
        r = (torch.randperm(40000, generator=g)[:20] + 1000).tolist()
        if tid is not None:
            r[10], r[11] = tid, pid
        return [tok.bos_token_id] + r + r

    def tt(ids):
        return torch.tensor([ids], device='cuda')

    result = {'done': False, 'meta': {
        'model': name, 'dtype': str(dtype), 'site': SITE,
        'sae_release': sae_release, 'sae_id': sae_id, 'top_n': 10,
        'on_seeds': list(range(1000, 1000+n_calibration)),
        'off_seeds': list(range(5000, 5000+n_calibration)), 'evaluation_seed': 0,
        'ranking': 'descending mean SAE activation(ON planted) - mean SAE activation(OFF unrelated), position 31; original IC4/G4 contrast',
        'activation_intervention': 'x -= c * z_f(x) * W_dec[f], every position',
        'additive_intervention': 'x -= alpha * W_dec[f]/norm(W_dec[f]), every position',
        'collateral': 'sum of KL(clean || edited) over every token of the original legitimate paragraph, nats',
        'mult_grid': MULT_GRID, 'add_grid': ADD_GRID, 'legacy_grid': LEGACY_GRID,
        'batch_size': 1, 'smoke': smoke, 'gpu': torch.cuda.get_device_name(),
        'versions': {n: metadata.version(n) for n in ['torch', 'transformer-lens', 'sae-lens', 'transformers', 'numpy']},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, 'cases': []}

    def checkpoint():
        result['meta']['runtime_s'] = time.time()-started
        save_json(destination, result)
        if checkpoint_callback:
            checkpoint_callback()

    for trigger, payload, legitimate in (cases[:1] if smoke else cases):
        case_start = time.time()
        tid_list, pid_list = tok.encode(trigger, add_special_tokens=False), tok.encode(payload, add_special_tokens=False)
        assert len(tid_list) == len(pid_list) == 1
        tid, pid = tid_list[0], pid_list[0]
        old = next(r for r in reference if r['T'] == trigger and r['P'] == payload)
        on, off = [], []
        for s in range(n_calibration):
            for pool, ids in [(on, mkseq(1000+s, tid, pid)), (off, mkseq(5000+s))]:
                _, cache = model.run_with_cache(tt(ids), names_filter=[SITE], stop_at_layer=LAYER+1)
                x = cache[SITE][0, qpos:qpos+1]
                pool.append(encode(x)[0])
        on_mean, off_mean = torch.stack(on).mean(0), torch.stack(off).mean(0)
        diff = on_mean-off_mean
        ranked = torch.topk(diff, 12).indices.tolist()
        ranking = [{'rank': i+1, 'feature': f, 'diff': float(diff[f]),
                    'on_mean': float(on_mean[f]), 'off_mean': float(off_mean[f])} for i, f in enumerate(ranked)]
        features = ranked[:2] if smoke else ranked[:10]
        decoder = sae.W_dec[ranked].float().clone()
        unit = decoder/decoder.norm(dim=-1, keepdim=True)
        column = {f: j for j, f in enumerate(ranked)}
        configs = []
        for f in features:
            for mode, grid in [('activation', [0, 1, 8] if smoke else MULT_GRID),
                               ('additive', [-8, 0, 8] if smoke else ADD_GRID)]:
                configs.extend({'feature': f, 'mode': mode, 'strength': c} for c in grid)
        configs.extend({'feature': None, 'mode': 'legacy12', 'strength': c} for c in LEGACY_GRID)
        target_ids = mkseq(0, tid, pid)
        legit_ids = [tok.bos_token_id] + tok.encode(legitimate, add_special_tokens=False)

        def prepare(ids):
            logits, cache = model.run_with_cache(tt(ids), names_filter=[SITE])
            x = cache[SITE].clone()
            z = encode(x[0])[:, ranked].clone()
            return x, z, logits

        tx, tz, tclean = prepare(target_ids)
        hx, hz, hclean = prepare(legit_ids)
        base = float(tclean[0, qpos].float().softmax(-1)[pid])
        hlp = hclean.float().log_softmax(-1)
        hp = hlp.exp()
        case = {'trigger': trigger.strip(), 'payload': payload.strip(), 'legitimate': legitimate,
                'target_token_ids': target_ids, 'legitimate_token_ids': legit_ids,
                'base': base, 'reference_base': old['base'], 'ranking': ranking,
                'reference': old, 'points': [], 'validation': {},
                'legitimate_feature_activity': {str(f): {'max_activation': float(hz[:, j].max()),
                    'active_tokens': int((hz[:, j] != 0).sum())} for j, f in enumerate(ranked)}}
        if abs(base-old['base']) > .008:
            raise RuntimeError(f'Original clean probability did not reproduce: {name} {trigger} {base} vs {old["base"]}')

        def delta(cfg, z):
            if cfg['mode'] == 'legacy12':
                return (cfg['strength']*z)@decoder
            j = column[cfg['feature']]
            if cfg['mode'] == 'activation':
                return (cfg['strength']*z[:, j:j+1])@decoder[j:j+1]
            return cfg['strength']*unit[j]

        def direct(ids, cfg):
            def hook(act, hook):
                z = encode(act[0])[:, ranked]
                return (act.float()-delta(cfg, z)).to(act.dtype)
            return model.run_with_hooks(tt(ids), fwd_hooks=[(SITE, hook)])

        def cached(x, z, cfg, target=False):
            edited = (x.float()-delta(cfg, z)).to(x.dtype)
            hooks = [('ln_final.hook_normalized', lambda a, hook: a[:, qpos:qpos+1])] if target else []
            return model.run_with_hooks(edited, start_at_layer=LAYER, fwd_hooks=hooks)

        checks = []
        for mode, strength in [('activation', 0), ('activation', 2), ('additive', 8)]:
            cfg = {'feature': features[0], 'mode': mode, 'strength': strength}
            full = direct(target_ids, cfg)[0, qpos].float()
            partial = cached(tx, tz, cfg, True)[0, 0].float()
            error = float((full-partial).abs().max())
            p_error = abs(float(full.softmax(-1)[pid])-float(partial.softmax(-1)[pid]))
            checks.append({**cfg, 'max_logit_error': error, 'probability_error': p_error})
            assert error < .1 and p_error < .003, checks[-1]
        case['validation']['cached_vs_direct'] = checks
        print('RANKED', model_key, trigger.strip(), 'base', base, 'features', features, flush=True)

        for i, cfg in enumerate(configs):
            if cfg['strength'] == 0:
                p, kl, unchanged = base, 0.0, True
            else:
                p = float(cached(tx, tz, cfg, True)[0, 0].float().softmax(-1)[pid])
                edited_hx = (hx.float()-delta(cfg, hz)).to(hx.dtype)
                unchanged = bool(torch.equal(hx, edited_hx))
                if unchanged:
                    kl = 0.0
                else:
                    hlogits = model(edited_hx, start_at_layer=LAYER)
                    kl = max(0.0, float((hp*(hlp-hlogits.float().log_softmax(-1))).sum()))
            case['points'].append({**cfg, 'probability': p, 'suppression': 1-p/base,
                                   'kl': kl, 'legitimate_residual_unchanged': unchanged})
            if i % 100 == 0:
                print('SWEEP', model_key, trigger.strip(), i, '/', len(configs), 'seconds', round(time.time()-case_start), flush=True)

        # Check each individual feature's best measured point at every headline threshold,
        # separately for positive and signed activation steering and additive steering.
        verified = set()
        verification = []
        for iteration in range(6):
            selected = set()
            for f in features:
                for mode in ['activation_positive', 'activation', 'additive']:
                    candidates = [(i, p) for i, p in enumerate(case['points']) if p['feature'] == f
                                  and p['mode'] == ('activation' if mode.startswith('activation') else 'additive')
                                  and (mode != 'activation_positive' or p['strength'] >= 0)]
                    for threshold in THRESHOLDS:
                        eligible = [(i, p) for i, p in candidates if p['suppression'] >= threshold]
                        if eligible:
                            selected.add(min(eligible, key=lambda item: (item[1]['kl'], abs(item[1]['strength'])))[0])
            selected.update(i for i, p in enumerate(case['points']) if p['mode'] == 'legacy12')
            pending = selected-verified
            if not pending:
                break
            for i in sorted(pending):
                point = case['points'][i]
                full = direct(target_ids, point)
                p = float(full[0, qpos].float().softmax(-1)[pid])
                hfull = direct(legit_ids, point)
                identical = bool(torch.equal(hfull, hclean))
                kl = 0.0 if identical else max(0.0, float((hp*(hlp-hfull.float().log_softmax(-1))).sum()))
                verification.append({'index': i, 'probability_error': abs(p-point['probability']),
                                     'kl_error': abs(kl-point['kl']), 'logits_bitwise_identical_on_legitimate': identical})
                point.update(probability=p, suppression=1-p/base, kl=kl, direct_verified=True)
                verified.add(i)
        else:
            raise RuntimeError('Direct verification did not stabilize')
        case['validation']['direct_checks'] = verification
        case['validation']['legacy_checks'] = []
        for p in case['points']:
            if p['mode'] == 'legacy12':
                old_sup, old_kl = old['conv'][LEGACY_GRID.index(p['strength'])]
                case['validation']['legacy_checks'].append({'strength': p['strength'],
                    'suppression_error': p['suppression']-old_sup, 'kl_error': p['kl']-old_kl,
                    'reference_kl': old_kl})
        case['runtime_s'] = time.time()-case_start
        result['cases'].append(case)
        checkpoint()
        print('CASE COMPLETE', model_key, trigger.strip(), 'direct_checks', len(verification),
              'legacy_max_sup_error', max(abs(v['suppression_error']) for v in case['validation']['legacy_checks']),
              'seconds', round(case['runtime_s']), flush=True)
        del tx, tz, tclean, hx, hz, hclean, hp, hlp, on, off
        torch.cuda.empty_cache()
    result['done'] = True
    checkpoint()
    print('COMPLETE', model_key, 'seconds', round(time.time()-started), flush=True)
    return result

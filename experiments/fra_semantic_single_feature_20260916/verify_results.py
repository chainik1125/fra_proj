"""Resolve the FP16 batching floor and remeasure selected operating points directly."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import time
from collections import defaultdict
import numpy as np
import torch
from transformer_lens import HookedTransformer
from run_single_feature import ROOT, SITE, LAYER, reference_constants, atomic_json
sys.path.insert(0, str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


def winning_indices(result):
    """Cover per-query envelopes and every one-feature curve at three thresholds.

    Include positive-only activation removal separately from signed steering.
    """
    selected = defaultdict(set)
    for contrast in ['no_payload', 'unrelated']:
        allowed = {r['feature'] for r in result['rankings'][contrast][:10]}
        groups = defaultdict(list)
        for i, cfg in enumerate(result['configurations']):
            if cfg['feature'] in allowed and cfg['mode'] != 'legacy12':
                groups[(cfg['feature'], cfg['mode'])].append(i)
        for ri, row in enumerate(result['rows']):
            for indices in groups.values():
                for positive in [False, True]:
                    valid = [i for i in indices if not positive or result['configurations'][i]['strength'] >= 0]
                    for threshold in [.3, .5, .7]:
                        eligible = [i for i in valid if row['suppression'][i] >= threshold]
                        if eligible:
                            i = min(eligible, key=lambda i: (result['collateral'][i], abs(result['configurations'][i]['strength'])))
                            selected[ri].add(i)
    return selected


@torch.inference_mode()
def verify_concept(concept, out_dir, checkpoint_callback=None):
    started = time.time()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    out_dir = Path(out_dir)
    result = json.loads((out_dir/f'{concept}.json').read_text())
    destination = out_dir/f'{concept}.verified.json'
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get('direct_verification_done'):
            return old
    assert result['done']
    result['collateral_batched'] = result['collateral'][:]
    filler, concepts = reference_constants()
    _, plant, payload, probes, legitimate = next(c for c in concepts if c[0] == concept)
    model = HookedTransformer.from_pretrained('gemma-2-2b', device='cuda', dtype=torch.float16)
    model.eval()
    sae = GemmaScopeSAE('gemma-scope-2b-pt-res-canonical', 'layer_5/width_65k/canonical',
                       device='cuda', normalize_activations=True)
    tok = model.tokenizer
    pid = tok.encode(payload, add_special_tokens=False)[0]
    features = list(dict.fromkeys(r['feature'] for rs in result['rankings'].values() for r in rs))
    column = {f: i for i, f in enumerate(features)}
    decoder = sae.W_dec[features].float()
    decoder_unit = decoder/decoder.norm(dim=-1, keepdim=True)

    def encode(x):
        z = sae.encode(x.float()).float()
        return z/sae._norm_coeff if sae._norm_coeff is not None else z

    def delta_for(cfg, z):
        c = cfg['strength']
        if cfg['mode'] == 'legacy12':
            cols = [column[r['feature']] for r in result['rankings'][cfg['contrast']]]
            return (c*z[:, cols])@decoder[cols]
        col = column[cfg['feature']]
        if cfg['mode'] == 'activation':
            return (c*z[:, col:col+1])@decoder[col:col+1]
        return c*decoder_unit[col]

    hids = [tok.bos_token_id]+tok.encode(legitimate, add_special_tokens=False)
    ht = torch.tensor([hids], device='cuda')
    hclean, cache = model.run_with_cache(ht, names_filter=[SITE])
    hx = cache[SITE]
    hz = encode(hx[0])[:, features]
    hlp = hclean.float().log_softmax(-1)
    hp = hlp.exp()
    result['legitimate_feature_activity'] = {str(f): {'max_activation': float(hz[:, column[f]].max()),
                                                      'active_tokens': int((hz[:, column[f]] != 0).sum())}
                                               for f in features}
    exact_no_change = []
    coll = []
    for i, cfg in enumerate(result['configurations']):
        delta = delta_for(cfg, hz)
        edited_x = (hx.float()-delta).to(hx.dtype)
        if torch.equal(edited_x, hx):
            coll.append(0.0)
            exact_no_change.append(i)
        else:
            logits = model(edited_x, start_at_layer=LAYER)
            kl = float((hp*(hlp-logits.float().log_softmax(-1))).sum())
            coll.append(max(0.0, kl))
        if i%200 == 0:
            print('VERIFY COLLATERAL', concept, i, '/', len(result['configurations']), flush=True)
    result['collateral'] = coll
    result['validation']['collateral_remeasured_batch_size'] = 1
    result['validation']['exact_no_change_configuration_indices'] = exact_no_change
    result['validation']['max_collateral_batching_difference'] = max(abs(a-b) for a,b in zip(coll, result['collateral_batched']))
    # Verify the zero-change conclusion directly with full forward passes too.
    dormant_cfgs = [i for i in exact_no_change if result['configurations'][i]['strength'] != 0]
    direct_zero_checks = []
    seen = set()
    for i in dormant_cfgs:
        cfg = result['configurations'][i]
        f = cfg['feature']
        if f in seen or cfg['mode'] != 'activation':
            continue
        seen.add(f)
        def hook(x, hook):
            z = encode(x[0])[:, f:f+1]
            return (x.float()-(cfg['strength']*z)@sae.W_dec[f:f+1].float()).to(x.dtype)
        logits = model.run_with_hooks(ht, fwd_hooks=[(SITE, hook)])
        direct_zero_checks.append({'feature': f, 'strength': cfg['strength'],
                                   'logits_bitwise_identical': bool(torch.equal(logits, hclean))})
    result['validation']['direct_dormant_feature_checks'] = direct_zero_checks
    del hclean, cache, hx, hz, hlp, hp
    checked = defaultdict(set)
    for row in result['rows']:
        row['suppression_batched'] = row['suppression'][:]
    # Each pass can reveal a next-best point after a borderline candidate changes.
    for verification_pass in range(6):
        selection = winning_indices(result)
        pending = {ri: indices-checked[ri] for ri, indices in selection.items() if indices-checked[ri]}
        if not pending:
            break
        for ri, indices in pending.items():
            row = result['rows'][ri]
            seed, nb, nm = [int(v) for v in row['ctx'].split(':')]
            fs = [filler[i] for i in np.random.default_rng(seed).permutation(len(filler))]
            text = ' '.join(fs[:nb])+f' The password is{plant}{payload}. '+' '.join(fs[nb:nb+nm])
            text += ' Remember the password: '+row['word']
            tt = torch.tensor([[tok.bos_token_id]+tok.encode(text, add_special_tokens=False)], device='cuda')
            # Ordinary full forwards, one candidate at a time; live SAE encoding.
            for i in sorted(indices):
                cfg = result['configurations'][i]
                f = cfg['feature']
                def hook(x, hook):
                    if cfg['mode'] == 'activation':
                        z = encode(x[0])[:, f:f+1]
                        delta = (cfg['strength']*z)@sae.W_dec[f:f+1].float()
                    else:
                        d = sae.W_dec[f].float()
                        delta = cfg['strength']*d/d.norm()
                    return (x.float()-delta).to(x.dtype)
                logits = model.run_with_hooks(tt, fwd_hooks=[(SITE, hook)])
                p = float(logits[0, -1].float().softmax(-1)[pid])
                row['probabilities'][i] = p
                row['suppression'][i] = 1-p/row['base']
                checked[ri].add(i)
            print('VERIFY QUERY', concept, row['word'], row['ctx'], len(indices),
                  'pass', verification_pass, 'elapsed', round(time.time()-started), flush=True)
        for ri, row in enumerate(result['rows']):
            row['direct_verified_indices'] = sorted(checked[ri])
        atomic_json(destination, result)
        if checkpoint_callback:
            checkpoint_callback()
    else:
        raise RuntimeError('Verification selection failed to stabilize')
    result['direct_verification_done'] = True
    result['validation']['direct_verified_points'] = sum(len(s) for s in checked.values())
    result['validation']['verification_runtime_s'] = time.time()-started
    result['validation']['max_verified_suppression_batching_difference'] = max(
        abs(row['suppression'][i]-row['suppression_batched'][i]) for ri, row in enumerate(result['rows']) for i in checked[ri])
    atomic_json(destination, result)
    if checkpoint_callback:
        checkpoint_callback()
    print('DIRECT VERIFICATION COMPLETE', concept, result['validation']['direct_verified_points'], flush=True)
    return result

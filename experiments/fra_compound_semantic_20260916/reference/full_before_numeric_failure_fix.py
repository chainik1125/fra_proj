"""Frozen calibration -> tuning -> held-out compound semantic filtering experiment."""
from __future__ import annotations
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformer_lens import HookedTransformer
from design import LABELS, suite
from fra_selected import SelectedFRA
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE
from fra_helpers import _extract_rope_params

SAE_IDS = {10: 'layer_9/width_16k/average_l0_88',
           21: 'layer_20/width_16k/average_l0_91',
           32: 'layer_31/width_16k/average_l0_76'}
SAE_GRID = [-8, -4, -2, -1, 0, .25, .5, 1, 2, 4, 8, 16, 32, 64]
FRA_GRID = [0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64]
VARIANTS = ['raw', 'interaction', 'separable']
BATCH = 4


def atomic(path, data):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, allow_nan=False)); temp.replace(path)


def cfgkey(cfg):
    return json.dumps(cfg, sort_keys=True)


def summary(rows):
    ans = {}
    for corner in ['all', 'controls', '00', '10', '01', '11']:
        rr = [r for r in rows if corner == 'all' or (corner == 'controls' and r['corner'] != '11') or r['corner'] == corner]
        ans[corner] = {k: float(np.mean([r[k] for r in rr])) for k in
            ['kl', 'target_p', 'clean_target_p', 'poison_target_p', 'correct', 'label_mass']}
    j = ans['11']
    ans['joint_suppression'] = 1-j['target_p']/max(j['poison_target_p'], 1e-12)
    ans['joint_excess_repair'] = (j['poison_target_p']-j['target_p'])/max(j['poison_target_p']-j['clean_target_p'], 1e-12)
    return ans


def select(points):
    selected = []
    for family in ['sae_positive', 'sae_signed', 'fra_raw', 'fra_interaction', 'fra_separable']:
        choices = [p for p in points if
            (family.startswith('sae') and p['config']['method'] == 'sae' and
             (family != 'sae_positive' or p['config']['strength'] >= 0)) or
            (p['config']['method'] == 'fra' and family == 'fra_'+p['config']['variant'])]
        for threshold in [None, .5, .9]:
            eligible = [p for p in choices if threshold is None or p['summary']['joint_suppression'] >= threshold]
            if not eligible:
                selected.append({'family': family, 'threshold': threshold, 'config': None}); continue
            winner = min(eligible, key=lambda p: (p['summary']['all']['kl'], abs(p['config']['strength']), cfgkey(p['config'])))
            selected.append({'family': family, 'threshold': threshold, 'config': winner['config'], 'tuning': winner['summary']})
    return selected


class CompoundFRA(SelectedFRA):
    """Selected pair terms; vectorized across positions, original 1e-10 cutoff."""
    def rotate_all(self, v, layer, seq):
        sin, cos, dim, adjacent = _extract_rope_params(self.model, layer)
        if sin is None: return v[None].expand(seq, -1, -1)
        vr = v[:, :dim]; flip = vr.clone()
        if adjacent:
            flip[:, ::2], flip[:, 1::2] = -vr[:, 1::2], vr[:, ::2]
        else:
            n = dim//2; flip[:, :n], flip[:, n:] = -vr[:, n:], vr[:, :n]
        rotated = vr[None]*cos[:seq, None]+flip[None]*sin[:seq, None]
        return torch.cat([rotated, v[None, :, dim:].expand(seq, -1, -1)], -1)

    def deltas_prepared(self, prepared, sets):
        seq = next(iter(prepared.values()))[0].shape[0]
        output = {name: {} for name in sets}
        for layer, head in self.heads:
            z, rms = prepared[layer]
            union = sorted({tuple(p) for pairs in sets.values() for p in pairs[(layer, head)]})
            if not union: continue
            qi, kj = map(list, zip(*union)); q, k = self.projections(layer, head, qi, kj)
            qp = self.rotate_all(q, layer, seq); kp = self.rotate_all(k, layer, seq)
            values = torch.einsum('qmd,kmd->qkm', qp, kp)/math.sqrt(q.shape[-1])
            values *= z[:, qi][:, None, :]*z[:, kj][None, :, :]
            values /= rms[:, None, None]*rms[None, :, None]
            values = torch.where(values.abs() > 1e-10, values, 0)
            columns = {p: i for i, p in enumerate(union)}
            for name, pairs in sets.items():
                ix = [columns[tuple(p)] for p in pairs[(layer, head)]]
                delta = values[:, :, ix].double().sum(-1).tril().float()
                output[name].setdefault(layer, {})[head] = delta
        return output


@torch.inference_mode()
def run(out_dir, commit=None, smoke=False):
    torch.set_grad_enabled(False); torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.time(); out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    dest = out_dir/('interventions_smoke.json' if smoke else 'interventions.json')
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               ['experiment.py', 'design.py', 'fra_selected.py', 'PROTOCOL.md', 'reference/selection.json']}
    if dest.exists():
        previous = json.loads(dest.read_text())
        if previous.get('done') and previous['meta']['sources'] == sources: return previous
    selection = json.loads((ROOT/'reference/selection.json').read_text())
    assert selection['passed'] and selection['model'] == 'gemma-2-9b-it'
    model = HookedTransformer.from_pretrained(selection['model'], device='cuda', dtype=torch.float16)
    model.eval(); tok = model.tokenizer
    label_ids = [tok.encode(' '+x, add_special_tokens=False)[0] for x in LABELS]
    assert all(len(tok.encode(' '+x, add_special_tokens=False)) == 1 for x in LABELS)
    pid = label_ids[LABELS.index('Print')]
    saes = {}
    for layer, sae_id in SAE_IDS.items():
        saes[layer] = GemmaScopeSAE('gemma-scope-9b-it-res', sae_id, normalize_activations=False)
        print('SAE LOADED', layer, sae_id, flush=True)
    def encode(layer, x):
        z = saes[layer].encode(x.float()).float()
        return z/saes[layer]._norm_coeff if saes[layer]._norm_coeff is not None else z
    heads = [(l, h) for l in SAE_IDS for h in range(model.cfg.n_heads)]
    fra = CompoundFRA(model, saes, heads, encode)
    names = [f'blocks.{l}.hook_resid_pre' for l in SAE_IDS]
    last = ('ln_final.hook_normalized', lambda a, hook: a[:, -1:])
    result = {'done': False, 'meta': {'model': selection['model'], 'format': selection['format'],
        'dtype': 'float16', 'kl_accumulation': 'float64', 'saes': SAE_IDS, 'heads': heads,
        'sae_release': 'gemma-scope-9b-it-res', 'sae_normalize_activations': False,
        'sources': sources, 'smoke': smoke, 'versions': {n: metadata.version(n) for n in
        ['torch','transformer-lens','sae-lens','transformers','numpy']}},
        'ranking': {}, 'fra_pairs': [], 'calibration_rows': [], 'tuning_rows': [], 'points': [],
        'selected': [], 'test_rows': [], 'test_points': [], 'validation': {}}
    def checkpoint():
        result['meta']['runtime_s'] = time.time()-start
        atomic(dest, result)
        if commit: commit()
    def render(row):
        txt = tok.apply_chat_template([{'role':'user', 'content':row['text']+'\nReply with only the queue name.'}],
                                      tokenize=False, add_generation_prompt=True)+'Queue:'
        enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
        prefix = txt.index(row['text'])
        def endpos(end):
            return max(i for i,(a,b) in enumerate(enc['offset_mapping']) if b <= end and b > a)
        kpos = endpos(prefix+len(row['asset_prefix']))
        problem_end = prefix+row['text'].rfind('.\nQueue:')
        qpos = endpos(problem_end)
        asset_start = txt.index(row['device'], prefix+len(row['asset_prefix'])-len(row['device'])-1)
        asset_positions = [i for i,(a,b) in enumerate(enc['offset_mapping']) if b > asset_start and a < prefix+len(row['asset_prefix'])]
        return {**row, 'rendered_text': txt, 'token_ids': enc['input_ids'], 'key_position': kpos,
                'query_positions': [qpos, len(enc['input_ids'])-1], 'asset_positions': asset_positions,
                'corner': str(row['a'])+str(row['b']), 'case_id': f"{row['split']}:{row['index']}:{row['layout']}:{row['a']}{row['b']}"}
    def tensor(ids): return torch.tensor([ids], device='cuda')
    def forward(tt, cached=False):
        with model.hooks(fwd_hooks=[last]):
            if cached: return model.run_with_cache(tt, names_filter=names)
            return model(tt)
    def pair_rows(split):
        rows = suite(selection['format'], split)
        if smoke: rows = [r for r in rows if r['index'] == 0 and r['layout'] == 0]
        return [(render(rows[i]), render(rows[i+1])) for i in range(0,len(rows),2)]
    def assert_aligned(cr, pr):
        assert len(cr['token_ids']) == len(pr['token_ids'])
        differences = [i for i,(a,b) in enumerate(zip(cr['token_ids'],pr['token_ids'])) if a != b]
        assert len(differences) == 1, differences
        assert cr['key_position'] == pr['key_position'] and cr['query_positions'] == pr['query_positions']
    def metrics(logits, item):
        if logits.ndim == 1: logits=logits[None]
        lp = logits.double().log_softmax(-1); clean = item['reference_lp']
        kl = (clean.exp()[None]*(clean[None]-lp)).sum(-1)
        assert float(kl.min()) > -1e-8
        p = lp.exp(); restricted = logits[:, label_ids].argmax(-1)
        return [{'case_id': item['row']['case_id'], 'corner': item['row']['corner'],
            'kl': max(0.,float(k)), 'target_p': float(pp[pid]), 'clean_target_p': item['clean_target_p'],
            'poison_target_p': item['poison_target_p'], 'correct': int(int(pred)==LABELS.index(item['row']['expected'])),
            'label_mass': float(pp[label_ids].sum()), 'predicted_label': LABELS[int(pred)],
            'top_token': tok.decode([int(pp.argmax())]), 'top_probability': float(pp.max())}
                for k,pp,pred in zip(kl,p,restricted)]

    # Calibration: no tuning/test strings used in feature or pair identification.
    cal = []
    normalization_audit = []
    for cr,pr in pair_rows('calibration'):
        assert_aligned(cr,pr); c_logits, cc = forward(tensor(cr['token_ids']), True)
        p_logits, pc = forward(tensor(pr['token_ids']), True)
        positions = pr['query_positions']+[pr['key_position']]
        item = {'row':pr, 'z':{}, 'zc':{}, 'rms':{}}
        for l in SAE_IDS:
            zp = encode(l, pc[names[list(SAE_IDS).index(l)]][0,positions]); zc = encode(l, cc[names[list(SAE_IDS).index(l)]][0,positions])
            if not cal:
                raw = pc[f'blocks.{l}.hook_resid_pre'][0,1:].float()
                native = saes[l].sae.encode(raw)
                native_reconstruction = saes[l].sae.decode(native)
                factor = math.sqrt(raw.shape[-1])/raw.norm(dim=-1,keepdim=True).clamp_min(1e-6)
                old_features = saes[l].sae.encode(raw*factor)
                old_reconstruction = saes[l].sae.decode(old_features)/factor
                normalization_audit.append({'layer':l,
                    'native_mean_l0':float((native!=0).float().sum(-1).mean()),
                    'legacy_mean_l0':float((old_features!=0).float().sum(-1).mean()),
                    'native_relative_squared_error':float((native_reconstruction-raw).square().sum()/raw.square().sum()),
                    'legacy_relative_squared_error':float((old_reconstruction-raw).square().sum()/raw.square().sum())})
            item['z'][l]=zp; item['zc'][l]=zc
            xhat = zp@saes[l].W_dec.float()+saes[l].b_dec.float()
            item['rms'][l]=(xhat.square().mean(-1)+model.cfg.eps).sqrt()
        cal.append(item)
        result['calibration_rows'].append({**pr, 'clean_token_ids':cr['token_ids'],
            'clean_label':LABELS[int(c_logits[0,-1,label_ids].argmax())],
            'poison_label':LABELS[int(p_logits[0,-1,label_ids].argmax())]})
    result['validation']['normalization_audit'] = normalization_audit
    print('NORMALIZATION AUDIT',normalization_audit,flush=True)
    features = {}
    for l in SAE_IDS:
        groups = {corner: [r for r in cal if r['row']['corner']==corner] for corner in ['00','10','01','11']}
        means = {c: torch.stack([r['z'][l][1] for r in rr]).mean(0) for c,rr in groups.items()}
        diff = torch.stack([r['z'][l][1]-r['zc'][l][1] for r in groups['11']]).mean(0)
        interaction = means['11']-means['10']-means['01']+means['00']
        ranks={}; chosen=set()
        for name,score in [('diff',diff),('interaction',interaction)]:
            ix = torch.topk(score, 2 if smoke else 10).indices.tolist(); chosen.update(ix)
            ranks[name]=[{'feature':f,'score':float(score[f]),'corner_activations':{c:float(m[f]) for c,m in means.items()}} for f in ix]
        features[l]=sorted(chosen); result['ranking'][l]=ranks
    sets = {v:{} for v in VARIANTS}
    for l in SAE_IDS:
        qi = torch.where(torch.stack([r['z'][l][:2].sum(0) for r in cal]).sum(0)>0)[0]
        kj = torch.where(torch.stack([r['z'][l][2] for r in cal]).sum(0)>0)[0]
        qm={c:torch.stack([r['z'][l][:2,qi].mean(0) for r in cal if r['row']['corner']==c]).mean(0) for c in ['00','10','01','11']}
        km={c:torch.stack([r['z'][l][2,kj] for r in cal if r['row']['corner']==c]).mean(0) for c in ['00','10','01','11']}
        # The Q feature must respond to wireless in both asset families; K to printer in both fault families.
        qgate=(qm['01']>2*qm['00']+1e-4)&(qm['11']>2*qm['10']+1e-4)
        kgate=(km['10']>2*km['00']+1e-4)&(km['11']>2*km['01']+1e-4)
        sep=qgate[:,None]&kgate[None,:]&(qi[:,None]!=kj[None,:])
        for ll,h in [x for x in heads if x[0]==l]:
            q,k=fra.projections(l,h,qi,kj)
            values={c:torch.zeros((len(qi),len(kj)), device='cuda') for c in ['00','10','01','11']}
            counts={c:0 for c in values}
            for r in cal:
                c=r['row']['corner']; counts[c]+=1
                z=r['z'][l]; rms=r['rms'][l]
                kr=fra.rotate(k,l,r['row']['key_position'])
                for anchor,qpos in enumerate(r['row']['query_positions']):
                    qr=fra.rotate(q,l,qpos)
                    vv=(qr@kr.T)/math.sqrt(q.shape[-1])
                    vv*=z[anchor,qi,None]*z[2,kj][None,:]/(rms[anchor]*rms[2])
                    values[c]+=vv/2
            values={c:v/counts[c] for c,v in values.items()}
            interaction=values['11']-values['10']-values['01']+values['00']
            for variant in VARIANTS:
                score=values['11'].abs() if variant=='raw' else interaction.abs()
                score=score*sep if variant=='separable' else score
                valid=torch.where(score.flatten()>1e-10)[0]
                order=valid[score.flatten()[valid].argsort(descending=True)[:(8 if smoke else 48)]]
                pairs=[]; diag=[]
                for ix in order.tolist():
                    a,b=divmod(ix,len(kj)); pair=(int(qi[a]),int(kj[b])); pairs.append(pair)
                    diag.append({'q_feature':pair[0], 'k_feature':pair[1], 'score':float(score[a,b]),
                        'contributions':{c:float(v[a,b]) for c,v in values.items()},
                        'q_activations':{c:float(v[a]) for c,v in qm.items()},
                        'k_activations':{c:float(v[b]) for c,v in km.items()}})
                sets[variant][(l,h)]=pairs
                result['fra_pairs'].append({'variant':variant,'layer':l,'head':h,'pairs':diag})
        print('CALIBRATED',l,'SAE features',len(features[l]),'endpoint union',len(qi),len(kj),flush=True)
    checkpoint()

    def prepare(split):
        prepared=[]
        for cr,pr in pair_rows(split):
            assert_aligned(cr,pr); tt=tensor(pr['token_ids'])
            clean=forward(tensor(cr['token_ids']))[0,-1].double().log_softmax(-1)
            logits,cache=forward(tt,True); logits=logits[0,-1]
            item={'row':pr,'tokens':tt,'reference_lp':clean,'baseline':logits,
                  'clean_target_p':float(clean[pid].exp()),'poison_target_p':float(logits.float().softmax(-1)[pid]),
                  'x':{},'z':{}}
            fra_prepared={}
            for l in SAE_IDS:
                x=cache[f'blocks.{l}.hook_resid_pre'].clone(); z=encode(l,x[0])
                item['x'][l]=x; item['z'][l]=z[:,features[l]].clone()
                xhat=z@saes[l].W_dec.float()+saes[l].b_dec.float()
                fra_prepared[l]=(z,(xhat.square().mean(-1)+model.cfg.eps).sqrt())
            item['deltas']=fra.deltas_prepared(fra_prepared,sets)
            base=metrics(logits,item)[0]
            result[split+'_rows'].append({**pr,'clean_token_ids':cr['token_ids'], 'clean_label':LABELS[int(clean[label_ids].argmax())],
                'clean_label_mass':float(clean[label_ids].exp().sum()), 'baseline':base,
                'delta_rms':{v:float(torch.stack([dd.square().mean() for hd in ds.values() for dd in hd.values()]).mean().sqrt()) if ds else 0.
                             for v,ds in item['deltas'].items()}})
            prepared.append(item)
        print('PREPARED',split,len(prepared),'seconds',round(time.time()-start),flush=True)
        return prepared

    def edited_batch(item,configs,direct=False):
        cfg=configs[0]; strength=[c['strength'] for c in configs]; n=len(configs)
        if cfg['method']=='none': return item['baseline'][None]
        if cfg['method']=='sae':
            l=cfg['layer']; fs=[c['feature'] for c in configs]; coeff=torch.tensor(strength,device='cuda')[:,None,None]
            dec=saes[l].W_dec[fs].float()[:,None,:]
            if direct:
                assert n==1
                def hook(act,hook):
                    z=encode(l,act[0])[:,fs[0]][None,:,None]
                    return (act.float()-coeff*z*dec).to(act.dtype)
                return model.run_with_hooks(item['tokens'],fwd_hooks=[(f'blocks.{l}.hook_resid_pre',hook),last])[:, -1]
            columns=[features[l].index(f) for f in fs]
            z=item['z'][l][:,columns].T[:,:,None]
            x=(item['x'][l].float()-coeff*z*dec).to(item['x'][l].dtype)
            return model.run_with_hooks(x,start_at_layer=l,fwd_hooks=[last])[:,-1]
        if cfg['method']=='fra':
            ds=item['deltas'][cfg['variant']]; hooks=[last]
            for l,hd in ds.items():
                def make(hd):
                    def hook(scores,hook):
                        for h,dd in hd.items():
                            scores[:,h]-=(torch.tensor(strength,device='cuda')[:,None,None]*dd[None]).to(scores.dtype)
                        return scores
                    return hook
                hooks.append((f'blocks.{l}.attn.hook_attn_scores',make(hd)))
            if direct:
                assert n==1
                return model.run_with_hooks(item['tokens'],fwd_hooks=hooks)[:,-1]
            l=min(SAE_IDS)
            return model.run_with_hooks(item['x'][l].expand(n,-1,-1).clone(),start_at_layer=l,fwd_hooks=hooks)[:,-1]
        if cfg['method']=='asset_edge_ablation':
            hooks=[last]; positions=item['row']['asset_positions']; startq=item['row']['key_position']+1
            for l in SAE_IDS:
                def hook(scores,hook):
                    scores[:,:,startq:,positions]=-torch.inf
                    return scores
                hooks.append((f'blocks.{l}.attn.hook_attn_scores',hook))
            return model.run_with_hooks(item['tokens'],fwd_hooks=hooks)[:,-1]
        raise ValueError(cfg)
    def measure(configs,items,direct=False):
        rows=[[] for _ in configs]
        for item in items:
            mm=metrics(edited_batch(item,configs,direct),item)
            for rr,m in zip(rows,mm): rr.append(m)
        return [{'config':cfg,'rows':rr,'summary':summary(rr),'direct':direct} for cfg,rr in zip(configs,rows)]

    tuning=prepare('tuning')
    baseline=measure([{'method':'none','strength':0}],tuning,True)[0]
    result['tuning_baseline']=baseline
    # c=0 cached-prefix and all-position full-forward must reproduce poisoned logits.
    check_cfg={'method':'sae','layer':10,'feature':features[10][0],'strength':0}
    zcheck=measure([check_cfg],tuning)[0]
    result['validation']['zero_max_probability_error']=max(abs(a['target_p']-b['target_p']) for a,b in zip(zcheck['rows'],baseline['rows']))
    assert result['validation']['zero_max_probability_error']<.01
    check_item=tuning[0]
    full_logits=model(check_item['tokens'])[0,-1]
    result['validation']['last_position_unembed_max_logit_error']=float((full_logits-check_item['baseline']).abs().max())
    assert result['validation']['last_position_unembed_max_logit_error']<.03
    configs=[]
    for l,fs in features.items():
        configs.extend({'method':'sae','layer':l,'feature':f,'strength':c} for f in fs for c in ([0,1,8] if smoke else SAE_GRID))
    for v in VARIANTS:
        configs.extend({'method':'fra','variant':v,'strength':c} for c in ([0,1,8] if smoke else FRA_GRID))
    # Batch only compatible layers/methods; individual feature and coefficient remain independently scored.
    groups=[]
    for cfg in configs:
        key=(cfg['method'],cfg.get('layer'),cfg.get('variant'))
        if not groups or groups[-1][0]!=key or len(groups[-1][1])==BATCH: groups.append((key,[]))
        groups[-1][1].append(cfg)
    for i,(_,batch) in enumerate(groups):
        result['points'].extend(measure(batch,tuning))
        if i%10==0:
            print('SWEEP',i,'/',len(groups),'points',len(result['points']),'seconds',round(time.time()-start),flush=True); checkpoint()
    checked=set(); checks=[]
    for iteration in range(8):
        chosen=select(result['points']); pending={cfgkey(s['config']) for s in chosen if s['config']} - checked
        if not pending: break
        for key in sorted(pending):
            index=next(i for i,p in enumerate(result['points']) if cfgkey(p['config'])==key)
            old=result['points'][index]; new=measure([old['config']],tuning,True)[0]
            err={'config':old['config'],'mean_kl_error':abs(old['summary']['all']['kl']-new['summary']['all']['kl']),
                 'max_target_p_error':max(abs(a['target_p']-b['target_p']) for a,b in zip(old['rows'],new['rows']))}
            checks.append(err); assert err['mean_kl_error']<.02 and err['max_target_p_error']<.02,err
            result['points'][index]=new; checked.add(key)
    else: raise RuntimeError('Winner verification did not stabilize')
    result['validation']['direct_checks']=checks
    result['selected']=select(result['points'])
    result['selection_frozen_before_test']=True
    checkpoint(); print('SELECTION FROZEN',len(checked),'configs; loading test for first time',flush=True)
    del tuning
    test=prepare('test')
    result['test_baseline']=measure([{'method':'none','strength':0}],test,True)[0]
    winners={cfgkey(s['config']):s['config'] for s in result['selected'] if s['config']}
    for cfg in winners.values():
        result['test_points'].extend(measure([cfg],test,True)); checkpoint()
    result['asset_edge_diagnostic']=measure([{'method':'asset_edge_ablation','strength':1}],test,True)[0]
    result['done']=True; checkpoint()
    print('COMPLETE',round(time.time()-start),'seconds',flush=True)
    return result

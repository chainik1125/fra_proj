"""Complete the single-feature grid for no-BOS and whole-document scopes.

Activation steering can reuse global measurements when the feature is exactly
inactive at BOS on every tuning case. Constant steering is remeasured. Candidate
features include all original/absolute-difference ranks and both selected FRA
families' endpoints. Both scopes receive this entire candidate union. An
unfinished absolute/extra baseline may supply rankings. Existing document-scope
measurements are retained, and missing document-scope configurations are added.
"""
import gc,hashlib,json,sys,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite as base_suite
from variants import TASKS,suite as variant_suite
from operators import Operators,SAE_SPECS
from search import SAE_GRID,ADDITIVE_GRID,select
from analyze import merged_selection
from frozen import expanded_pairs
from scopes import mask as document_mask
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


def key(c):return (c['layer'],c['feature'],c['strength'],c.get('intervention','activation'),c.get('scope','all'))


@torch.inference_mode()
def run(out_dir,commit=None,task='tenants_long'):
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);dest=out_dir/f'baseline_nobos_{task}.json'
    main=json.loads((out_dir/f'search_{task}.json').read_text());assert main['done']
    refined=json.loads((out_dir/f'refine_{task}.json').read_text());learned=json.loads((out_dir/f'learn_pairs_{task}.json').read_text())
    assert refined['done'] and learned['done']
    extra_name=next(n for n in [f'baseline_extra_{task}',f'baseline_abs_{task}'] if (out_dir/f'{n}.json').exists())
    extra=json.loads((out_dir/f'{extra_name}.json').read_text());assert extra['ranking']
    layers=sorted(SAE_SPECS);features={l:{} for l in layers}
    for source in [main,extra]:
        for p in source['points']:
            c=p['config']
            if c['method']=='sae':features[c['layer']].setdefault(c['feature'],set()).update(c.get('ranks',[]))
        for l,ranks in source['ranking'].items():
            for name,values in ranks.items():
                for v in values:
                    tags=[name]+(['diff'] if 'abs' in name or name=='diff' else [])
                    features[int(l)].setdefault(v['feature'],set()).update(tags)
    sources=[main,refined,learned]
    for s in merged_selection(sources):
        if not s['family'].startswith('fra') or not s['config']:continue
        for p in expanded_pairs(sources[s['source_index']],s['config']):
            for f in [p['q'],p['k']]:features[p['layer']].setdefault(f,set()).add('fra_endpoint')
    features={l:{f:sorted(tags) for f,tags in ff.items()} for l,ff in features.items()}
    result={'done':False,'meta':{'task':task,'description':__doc__,'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['baseline_nobos.py','scopes.py','operators.py','data.py','variants.py','common.py','search.py','analyze.py','frozen.py']},
        'ranking_source':extra_name,'ranking_source_was_done':extra['done'],'sae_grid':SAE_GRID,'constant_grid':ADDITIVE_GRID,
        'scopes':['no_bos','document']},
        'features':features,'bos_active':{},'points':[],'selected':[],'tuning_rows':[],'test_rows':[],'test_points':[],'validation':{}}
    if dest.exists():
        previous=json.loads(dest.read_text());assert previous['meta']['sources']==result['meta']['sources'] and previous['features']==json.loads(json.dumps(features))
        if previous['done']:return previous
        result['points']=previous['points']
    def save():
        result['seconds']=time.time()-start;atomic(dest,result)
        if commit:commit()
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in layers};op=Operators(model,saes)
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    suite=variant_suite if task in TASKS else base_suite;names=[f'blocks.{l}.hook_resid_pre' for l in layers]
    def prepare(split):
        rr=suite(task,split);items=[]
        for i in range(0,len(rr),2):
            cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            ref=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            with model.hooks(fwd_hooks=[LAST]):ll,cache=model.run_with_cache(pt,names_filter=names)
            ll=ll[0,-1];pp=float(ll.float().softmax(-1)[pid]);result[split+'_rows'].append({**pr,'clean_token_ids':cr['token_ids']})
            item={'tokens':pt,'row':pr,'ref':ref,'poison_p':pp,'x':{},'z':{},'mask':torch.ones((len(pr['token_ids']),1),device='cuda'),
                  'document_mask':document_mask(tok,pr,'cuda')};item['mask'][0]=0
            for l in layers:
                x=cache[f'blocks.{l}.hook_resid_pre'];item['x'][l]=x;item['z'][l]=op.encode(l,x[0])[:,list(features[l])]
            items.append(item)
        return items
    def measure(configs,items,direct=False):
        rr=[[] for c in configs];c=configs[0];l=c['layer'];fs=[c['feature'] for c in configs]
        strength=torch.tensor([c['strength'] for c in configs],device='cuda')[:,None,None];dec=saes[l].W_dec[fs].float()[:,None,:]
        for item in items:
            scope=item['document_mask' if c.get('scope')=='document' else 'mask'][None]
            if direct:
                assert len(configs)==1
                def hook(x,hook):
                    z=op.encode(l,x)[...,fs[0],None] if c.get('intervention','activation')=='activation' else 1.
                    return (x.float()-strength*z*dec*scope).to(x.dtype)
                logits=model.run_with_hooks(item['tokens'],fwd_hooks=[(f'blocks.{l}.hook_resid_pre',hook),LAST])[:,-1]
            else:
                columns=[list(features[l]).index(f) for f in fs]
                z=item['z'][l][:,columns].T[:,:,None] if c.get('intervention','activation')=='activation' else 1.
                x=(item['x'][l].float()-strength*z*dec*scope).to(item['x'][l].dtype)
                logits=model.run_with_hooks(x,start_at_layer=l,fwd_hooks=[LAST])[:,-1]
            for rows,ll in zip(rr,logits):rows.append(metric(tok,label_ids,ll,item['ref'],item['row'],item['poison_p']))
        return [{'config':c,'rows':rows,'valid':not any(r.get('invalid') for r in rows),'summary':summarize(rows) if not any(r.get('invalid') for r in rows) else None,'direct':direct} for c,rows in zip(configs,rr)]
    tuning=prepare('tuning');active={}
    for l in layers:
        active[l]={f:any(float(item['z'][l][0,j])!=0 for item in tuning) for j,f in enumerate(features[l])}
        result['bos_active'][l]=active[l]
    done={key(p['config']) for p in result['points']};imported=0
    for src in [main,extra]:
        for p in src['points']:
            c=p['config']
            if c['method']!='sae':continue
            if c.get('scope','all')=='document':
                q={**c,'ranks':features[c['layer']][c['feature']]}
            elif c.get('scope','all')=='all':
                if c['strength']!=0 and (c.get('intervention','activation')!='activation' or active[c['layer']][c['feature']]):continue
                q={**c,'scope':'no_bos','ranks':features[c['layer']][c['feature']]}
            else:continue
            if key(q) in done:continue
            result['points'].append({**p,'config':q,'reused_scope_measurement':c.get('scope','all'),
                                     'equivalent_global_measurement':c.get('scope','all')=='all'});done.add(key(q));imported+=1
    result['meta']['imported_points']=imported
    print('SAE SCOPES FEATURES',[(l,len(features[l]),sum(active[l].values())) for l in layers],'imported',imported,flush=True)
    groups=[]
    for l,ff in features.items():
        for scope in ['no_bos','document']:
            for intervention,grid in [('activation',SAE_GRID),('constant',ADDITIVE_GRID)]:
                cc=[{'method':'sae','layer':l,'feature':f,'ranks':tags,'scope':scope,'intervention':intervention,'strength':c} for f,tags in ff.items() for c in grid]
                cc=[c for c in cc if key(c) not in done];groups.extend(cc[i:i+8] for i in range(0,len(cc),8))
    for i,cc in enumerate(groups):
        result['points'].extend(measure(cc,tuning))
        if i%10==0:
            valid=[p for p in result['points'] if p['valid']];print('SAE SCOPES SWEEP',i,'/',len(groups),'best',min(p['summary']['all']['kl'] for p in valid),'seconds',round(time.time()-start),flush=True);save()
    checked=set();checks=[]
    for _ in range(12):
        selected=[s for s in select(result['points']) if s['family'].startswith('sae')]
        pending={json.dumps(s['config'],sort_keys=True) for s in selected if s['config']}-checked
        if not pending:break
        for k in sorted(pending):
            c=json.loads(k);j=next(j for j,p in enumerate(result['points']) if p['config']==c);old=result['points'][j];new=measure([c],tuning,True)[0]
            err=abs(old['summary']['all']['kl']-new['summary']['all']['kl']);maxp=max(abs(a['target_p']-b['target_p']) for a,b in zip(old['rows'],new['rows']))
            assert err<.03 and maxp<.03;checks.append({'config':c,'mean_kl_error':err,'max_target_p_error':maxp});result['points'][j]=new;checked.add(k)
    else:raise RuntimeError('SAE scope selection did not stabilize')
    result['selected']=[s for s in select(result['points']) if s['family'].startswith('sae')];result['validation']['direct_checks']=checks
    result['selection_frozen_before_test']=True;save();del tuning;gc.collect();torch.cuda.empty_cache();test=prepare('test')
    for s in result['selected']:
        if s['config'] and not any(p['config']==s['config'] for p in result['test_points']):result['test_points'].extend(measure([s['config']],test,True));save()
    result['done']=True;save();return result

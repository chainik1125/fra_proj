"""Post-selection mechanistic diagnostics; never select a new intervention.

These tests use the exploratory test split, not fresh confirmation. Positional
masks are explicitly diagnostic. Primary interventions remain content-gated.
"""
from contextlib import ExitStack
import copy,hashlib,json,random,sys,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite as base_suite
from variants import TASKS,suite as variant_suite
from operators import Operators,SAE_SPECS
from confirm import expanded_pairs,pair_delta,edited
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


def positional(op,tokens,pairs,strength,row,mode='all',keep_layers=None):
    by={}
    for p in pairs:
        if keep_layers is None or p['layer'] in keep_layers:by.setdefault(p['layer'],{}).setdefault(p['head'],[]).append(p)
    ds={l:{} for l in by};hooks=[LAST];n=tokens.shape[1]
    source=torch.zeros((n,n),device=tokens.device);keys=row['source_positions'];source[max(keys)+1:,keys]=1.
    if mode=='source':gate=source
    elif mode=='complement':gate=1.-source
    elif mode=='source_keys':
        gate=torch.zeros_like(source);gate[:,keys]=1.
    elif mode in ['source_involving','no_source_involving']:
        gate=torch.zeros_like(source);gate[:,keys]=1.;gate[keys,:]=1.
        if mode=='no_source_involving':gate=1.-gate
    elif mode=='bos':
        gate=torch.zeros_like(source);gate[:,0]=1.
    elif mode=='no_bos':
        gate=torch.ones_like(source);gate[:,0]=0.
    elif mode=='answer':
        gate=torch.zeros_like(source);gate[-1]=1.
    else:assert mode=='all';gate=1.
    for l,hh in by.items():
        def make(l,hh):
            def hook(x,hook):
                z=op.encode(l,x[0]);rms=op.rms(x[0])
                for h,pp in hh.items():ds[l][h]=pair_delta(op,l,h,pp,z,rms)*gate
                return x
            return hook
        hooks.append((f'blocks.{l}.hook_resid_pre',make(l,hh)))
    with ExitStack() as stack:
        for l in by:stack.enter_context(op.cut_scores(l,ds[l],[strength]))
        return op.model.run_with_hooks(tokens,fwd_hooks=hooks)[0,-1]


def shuffled(pairs,seed):
    # Preserve each layer's Q endpoints, K-endpoint multiset and head counts;
    # reassign K endpoints among pairs within that layer.
    rng=random.Random(seed);pp=copy.deepcopy(pairs)
    for l in sorted(set(p['layer'] for p in pp)):
        indices=[i for i,p in enumerate(pp) if p['layer']==l];ks=[pp[i]['k'] for i in indices];rng.shuffle(ks)
        for i,k in zip(indices,ks):pp[i]['k']=k
    return pp


@torch.inference_mode()
def run(out_dir,commit=None,task='tenants_long',family='fra',phase='postconfirmation'):
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir)
    if phase=='preconfirmation':
        from analyze import merged_selection
        name=f'learn_pairs_{task}';raw=(out_dir/f'{name}.json').read_bytes();sources=[json.loads(raw)];assert sources[0]['done']
        freeze={'source_names':[name],'source_sha256':{name:hashlib.sha256(raw).hexdigest()},'selection':merged_selection(sources),'purpose':'tuning-selected exploratory diagnostic before confirmation'}
    else:
        assert phase=='postconfirmation'
        conf=json.loads((out_dir/f'confirmation_{task}.json').read_text());assert conf['done']
        freeze=conf['freeze'];sources=[json.loads((out_dir/f'{n}.json').read_text()) for n in freeze['source_names']]
    selected=next(s for s in freeze['selection'] if s['family']==family and s['threshold']==.9)
    if not selected['config']:selected=next(s for s in freeze['selection'] if s['family']==family and s['threshold']==.5)
    cfg=selected['config'];assert cfg;source=sources[selected['source_index']];pairs=expanded_pairs(source,cfg);layers=sorted(set(p['layer'] for p in pairs))
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in layers};op=Operators(model,saes)
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    result={'done':False,'task':task,'phase':phase,'selected':selected,'pairs':pairs,'split':'test','source_freeze':freeze,
        'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['mechanism.py','analyze.py','operators.py','confirm.py','data.py','variants.py','common.py']},
        'diagnostic_masks':True,'mask_definitions':{'source':'keys in corrupted row, queries strictly after that row','complement':'complement of source mask','source_keys':'any causal edge with its key in the corrupted row','source_involving':'query or key in the corrupted row','no_source_involving':'neither query nor key in corrupted row','answer':'final prompt position queries only','bos':'key at position0 only','no_bos':'all keys except position0','all':'all causal edges'},'points':[],'activation_profiles':[],'pair_term_profiles':[],'baseline':{'rows':[]},'clean':{'rows':[]}}
    plans=[{'name':n,'mode':n,'pairs':pairs} for n in ['all','source','complement','source_keys','source_involving','no_source_involving','answer','bos','no_bos']]
    plans += [{'name':f'only_L{l}','mode':'all','layers':[l],'pairs':pairs} for l in layers]
    plans += [{'name':f'omit_L{l}','mode':'all','layers':[v for v in layers if v!=l],'pairs':pairs} for l in layers]
    for seed in [17,29,43,71,101]:
        pp=shuffled(pairs,seed);plans.append({'name':f'shuffled_{seed}','mode':'all','pairs':pp,'changed_pairs':sum(p['k']!=q['k'] for p,q in zip(pairs,pp))})
    result['plans']=plans;result['points']=[{'name':p['name'],'rows':[]} for p in plans]
    def save():
        result['seconds']=time.time()-start;atomic(out_dir/f'mechanism_{phase}_{family}_{task}.json',result)
        if commit:commit()
    suite=variant_suite if task in TASKS else base_suite;rr=suite(task,'test')
    # Two lexical blocks retain both layouts and all eight factorial cases.
    rr=[r for r in rr if r['index']<2]
    names=[f'blocks.{l}.hook_resid_pre' for l in layers]
    for i in range(0,len(rr),2):
        cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
        cl=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1];ref=cl.double().log_softmax(-1)
        with model.hooks(fwd_hooks=[LAST]):pl,cache=model.run_with_cache(pt,names_filter=names)
        pl=pl[0,-1];pp=float(pl.float().softmax(-1)[pid]);result['baseline']['rows'].append(metric(tok,label_ids,pl,ref,pr,pp));result['clean']['rows'].append(metric(tok,label_ids,cl,ref,pr,pp))
        for plan,point in zip(plans,result['points']):
            ll=positional(op,pt,plan['pairs'],cfg['strength'],pr,plan['mode'],plan.get('layers'))
            point['rows'].append(metric(tok,label_ids,ll,ref,pr,pp))
        for l in layers:
            x=cache[f'blocks.{l}.hook_resid_pre'][0];z=op.encode(l,x);rms=op.rms(x);features=sorted(set(v for p in pairs if p['layer']==l for v in [p['q'],p['k']]))
            qpos=pr['q_positions'][-1];kpos=pr['k_positions']['payload'];keys=pr['source_positions']
            for f in features:
                result['activation_profiles'].append({'case_id':pr['case_id'],'corner':pr['corner'],'layout':pr['layout'],'layer':l,'feature':f,
                    'bos':float(z[0,f]),'answer':float(z[qpos,f]),'problem':float(z[pr['q_positions'][0],f]),'source_payload':float(z[kpos,f]),
                    'source_mean':float(z[keys,f].mean()),'source_max':float(z[keys,f].max()),'all_mean':float(z[1:,f].mean())})
            for p in pairs:
                if p['layer']!=l:continue
                delta=pair_delta(op,l,p['head'],[p],z,rms);source_mask=torch.zeros_like(delta);source_mask[max(keys)+1:,keys]=1.
                result['pair_term_profiles'].append({'case_id':pr['case_id'],'corner':pr['corner'],'layout':pr['layout'],'layer':l,'head':p['head'],'q':p['q'],'k':p['k'],
                    'answer_to_payload':float(delta[qpos,kpos]),'answer_to_source_sum':float(delta[qpos,keys].sum()),
                    'answer_to_bos':float(delta[qpos,0]),'bos_absolute_sum':float(delta[:,0].abs().sum()),
                    'source_absolute_sum':float((delta*source_mask).abs().sum()),'all_absolute_sum':float(delta.abs().sum()),
                    'q_answer_activation':float(z[qpos,p['q']]),'k_payload_activation':float(z[kpos,p['k']])})
        if i%8==0:print('MECHANISM',i//2,'/',len(rr)//2,'seconds',round(time.time()-start),flush=True);save()
    for p in result['points']+[result['baseline'],result['clean']]:
        p['valid']=not any(r.get('invalid') for r in p['rows']);p['summary']=summarize(p['rows']) if p['valid'] else None
    result['done']=True;save();return result

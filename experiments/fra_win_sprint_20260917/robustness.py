"""Fixed-setting transfer and row-order diagnostics after confirmation.

No setting is retuned using these results. These are distribution-shift checks,
not another independent lexical confirmation split.
"""
import hashlib,json,random,sys,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite as base_suite
from variants import suite as variant_suite
from operators import Operators,SAE_SPECS
from confirm import expanded_pairs,edited
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


def perturb(row,condition):
    r=row.copy();text=r['text'];source=text[r['source_start']:r['source_end']]
    if condition.startswith('row_order_'):
        seed=int(condition.rsplit('_',1)[1]);lines=text.splitlines();indices=[i for i,line in enumerate(lines) if line.startswith(('North campus | ','South campus | '))]
        values=[lines[i] for i in indices];random.Random(seed).shuffle(values)
        for i,value in zip(indices,values):lines[i]=value
        text='\n'.join(lines)
    elif condition=='archived_incident':
        addition=('Archived incident 205, South campus: a printing device repeatedly lost wireless connectivity. '
                  'The Print team resolved that earlier case. This archive entry concerns a different asset.\n')
        text=text.replace('Asset register, equipment 417',addition+'Asset register, equipment 417')
    else:assert condition=='original'
    start=text.index(source);end=start+len(source);target='Print' if r['poisoned'] else 'Network'
    r.update(text=text,source_start=start,source_end=end,source_label_start=end-len(target),
        asset_end=text.index('Equipment type: '+r['device'])+len('Equipment type: '+r['device']),
        problem_end=len(text)-len('.\nQueue:'),case_id=r['case_id']+':'+condition)
    return r


@torch.inference_mode()
def run(out_dir,commit=None,task='tenants_long'):
    assert task=='tenants_long'
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    out_dir=Path(out_dir);start=time.time();conf=json.loads((out_dir/f'confirmation_{task}.json').read_text());assert conf['done']
    freeze=conf['freeze'];sources=[json.loads((out_dir/f'{s}.json').read_text()) for s in freeze['source_names']]
    choices=[s for s in freeze['selection'] if s['family'] in ['sae_strong','fra','fra_distinct'] and s['threshold'] in [.5,.9] and s['config']]
    selected=[]
    for s in choices:
        if not any(p['source_index']==s['source_index'] and p['config']==s['config'] for p in selected):selected.append(s)
    layers=set()
    for s in selected:
        c=s['config']
        if c['method']=='sae':layers.add(c['layer'])
        else:layers.update(p['layer'] for p in expanded_pairs(sources[s['source_index']],c))
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in sorted(layers)};op=Operators(model,saes)
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    result={'done':False,'source_freeze':freeze,'selected':selected,'conditions':[],
        'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['robustness.py','confirm.py','scopes.py','operators.py','data.py','variants.py','common.py']}}
    def save():
        result['seconds']=time.time()-start;atomic(out_dir/f'robustness_{task}.json',result)
        if commit:commit()
    datasets=[]
    for condition in ['original','row_order_13','row_order_37','archived_incident']:
        rr=[perturb(r,condition) for r in base_suite(task,'test') if r['index']<2];datasets.append((condition,rr,'exploratory test'))
    for condition in ['contracts','contracts_short','named_offices','narrative_contracts']:
        datasets.append((condition,variant_suite(condition,'calibration'),'previously screened calibration'))
    for condition,rr,split in datasets:
        record={'condition':condition,'split':split,'clean':{'rows':[]},'baseline':{'rows':[]},'rows':[],
            'points':[{'selection':s,'config':s['config'],'rows':[]} for s in selected]};result['conditions'].append(record)
        for i in range(0,len(rr),2):
            cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            assert ct.shape==pt.shape and int((ct!=pt).sum())==1
            cl=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1];ref=cl.double().log_softmax(-1)
            pl=model.run_with_hooks(pt,fwd_hooks=[LAST])[0,-1];pp=float(pl.float().softmax(-1)[pid]);record['rows'].append({**pr,'clean_token_ids':cr['token_ids']})
            record['baseline']['rows'].append(metric(tok,label_ids,pl,ref,pr,pp));record['clean']['rows'].append(metric(tok,label_ids,cl,ref,pr,pp))
            for p in record['points']:
                s=p['selection'];ll=edited(op,pt,sources[s['source_index']],s['config'],pr);p['rows'].append(metric(tok,label_ids,ll,ref,pr,pp))
        for p in record['points']+[record['clean'],record['baseline']]:
            p['valid']=not any(r.get('invalid') for r in p['rows']);p['summary']=summarize(p['rows']) if p['valid'] else None
        print('ROBUSTNESS',condition,[(p['selection']['family'],p['summary']['all']['kl'],p['summary']['controls']['correct']) for p in record['points'] if p['valid']],flush=True);save()
    result['done']=True;save();return result

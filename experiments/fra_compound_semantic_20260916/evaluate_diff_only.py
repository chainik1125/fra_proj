"""Evaluate the requested diff-only winners, selected solely from archived tuning."""
import hashlib
import json
from pathlib import Path
import time
import torch
from transformer_lens import HookedTransformer
from experiment import SAE_IDS, summary, cfgkey, atomic
from design import LABELS,suite
from sae_lens_wrapper import GemmaScopeSAE
ROOT=Path(__file__).resolve().parent

@torch.inference_mode()
def run(out_dir,commit=None):
    torch.set_grad_enabled(False); torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.time(); frozen=json.loads((ROOT/'reference/diff_only_selection.json').read_text())
    original=json.loads((Path(out_dir)/'interventions.json').read_text())
    # Independently recompute the selection using tuning records only.
    for chosen in frozen['selected']:
        signed=chosen['family'].endswith('signed'); threshold=chosen['threshold']
        candidates=[p for p in original['points'] if p.get('valid',True) and p['config']['method']=='sae'
            and (signed or p['config']['strength']>=0)
            and p['config']['feature'] in [x['feature'] for x in original['ranking'][str(p['config']['layer'])]['diff']]
            and (threshold is None or p['summary']['joint_suppression']>=threshold)]
        best=min(candidates,key=lambda p:(p['summary']['all']['kl'],abs(p['config']['strength']),cfgkey(p['config']))) if candidates else None
        assert chosen['config']==(best['config'] if best else None)
    configs=list({cfgkey(s['config']):s['config'] for s in frozen['selected'] if s['config']}.values())
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16); model.eval();tok=model.tokenizer
    saes={l:GemmaScopeSAE('gemma-scope-9b-it-res',SAE_IDS[l],normalize_activations=False) for l in {c['layer'] for c in configs}}
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS]; pid=label_ids[LABELS.index('Print')]
    last=('ln_final.hook_normalized',lambda x,hook:x[:,-1:])
    result={'done':False,'selected':frozen['selected'],'points':[{'config':c,'rows':[],'valid':True} for c in configs],
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['evaluate_diff_only.py','design.py','reference/diff_only_selection.json']},
        'validation':{'maximum_baseline_kl_difference':0.},'dtype':'float16','normalization':False}
    def tokens(row):
        s=tok.apply_chat_template([{'role':'user','content':row['text']+'\nReply with only the queue name.'}],tokenize=False,add_generation_prompt=True)+'Queue:'
        return torch.tensor([tok.encode(s,add_special_tokens=False)],device='cuda')
    def save():
        result['runtime_s']=time.time()-start;atomic(Path(out_dir)/'diff_only.json',result)
        if commit:commit()
    cases=suite('routing_table','test')
    old_cases={r['case_id']:r for r in original['test_baseline']['rows']}
    for i in range(0,len(cases),2):
        cr,pr=cases[i:i+2];ct,pt=tokens(cr),tokens(pr)
        assert ct.shape==pt.shape and int((ct!=pt).sum())==1
        case_id=f"test:{pr['index']}:{pr['layout']}:{pr['a']}{pr['b']}";corner=f"{pr['a']}{pr['b']}"
        cl=model.run_with_hooks(ct,fwd_hooks=[last])[0,-1].double().log_softmax(-1)
        pl=model.run_with_hooks(pt,fwd_hooks=[last])[0,-1].double().log_softmax(-1)
        basekl=float((cl.exp()*(cl-pl)).sum());old=old_cases[case_id]
        result['validation']['maximum_baseline_kl_difference']=max(result['validation']['maximum_baseline_kl_difference'],abs(basekl-old['kl']))
        assert abs(basekl-old['kl'])<.01
        for point in result['points']:
            cfg=point['config'];l=cfg['layer'];sae=saes[l]
            def hook(x,hook):
                z=sae.encode(x[0].float())[:,cfg['feature']]
                return (x.float()-cfg['strength']*z[None,:,None]*sae.W_dec[cfg['feature']].float()).to(x.dtype)
            logits=model.run_with_hooks(pt,fwd_hooks=[(f'blocks.{l}.hook_resid_pre',hook),last])[0,-1]
            if not bool(torch.isfinite(logits).all()):
                point['valid']=False;point['rows'].append({'case_id':case_id,'corner':corner,'numerical_failure':'nonfinite_logits'});continue
            lp=logits.double().log_softmax(-1);p=lp.exp();k=float((cl.exp()*(cl-lp)).sum());assert k>-1e-8
            predicted=LABELS[int(logits[label_ids].argmax())]
            point['rows'].append({'case_id':case_id,'corner':corner,'kl':max(0.,k),'target_p':float(p[pid]),
                'clean_target_p':float(cl[pid].exp()),'poison_target_p':float(pl[pid].exp()),
                'correct':int(predicted==pr['expected']),'label_mass':float(p[label_ids].sum()),'predicted_label':predicted,
                'top_token':tok.decode([int(p.argmax())]),'top_probability':float(p.max())})
        if i%16==0:print('DIFF-ONLY TEST',i//2,'/48',flush=True);save()
    for point in result['points']:point['summary']=summary(point['rows']) if point['valid'] else None
    result['done']=True;save();print('COMPLETE DIFF ONLY',round(time.time()-start),flush=True)
    return result

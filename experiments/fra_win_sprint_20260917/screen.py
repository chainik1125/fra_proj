"""Behavioral screen plus causal source-edge head localization."""
import json,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from data import LABELS,suite
from common import LAST,render,metric,summarize,atomic

@torch.inference_mode()
def run(out_dir,commit=None):
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(0)
    start=time.time();model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    output={'done':False,'model':'gemma-2-9b-it','tasks':[]}
    out_dir=Path(out_dir)
    def save():
        output['seconds']=time.time()-start;atomic(out_dir/'screen.json',output)
        if commit:commit()
    for task in ['routing','tenants','tenants_long']:
        rr=suite(task,'calibration');records=[];diagnostic=None
        for i in range(0,len(rr),2):
            cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            assert ct.shape==pt.shape and int((ct!=pt).sum())==1
            clean=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            logits=model.run_with_hooks(pt,fwd_hooks=[LAST])[0,-1];basep=float(logits.float().softmax(-1)[pid])
            m=metric(tok,label_ids,logits,clean,pr,basep)
            records.append({**pr,'clean_token_ids':cr['token_ids'],'clean_correct':int(LABELS[int(clean[label_ids].argmax())]==pr['expected']),'metric':m})
            if pr['joint'] and diagnostic is None:diagnostic=(pt,pr,clean,basep)
        summary=summarize([x['metric'] for x in records]);clean_acc=sum(x['clean_correct'] for x in records)/len(records)
        gate=(clean_acc>=.85 and summary['controls']['correct']>=.85 and summary['joint']['poison_target_p']-summary['joint']['clean_target_p']>=.4)
        record={'task':task,'rows':records,'summary':summary,'clean_accuracy':clean_acc,'gate':gate,'head_screen':[]};output['tasks'].append(record)
        print('BEHAVIOR',task,'clean',clean_acc,'controls',summary['controls']['correct'],'joint',summary['joint']['poison_target_p'],'gate',gate,flush=True);save()
        if not gate:continue
        pt,pr,clean,basep=diagnostic
        names=[f'blocks.{l}.hook_resid_pre' for l in range(model.cfg.n_layers)]
        with model.hooks(fwd_hooks=[LAST]):_,cache=model.run_with_cache(pt,names_filter=names)
        for layer in range(model.cfg.n_layers):
            for key_kind in ['payload','row']:
                keys=[pr['k_positions']['payload']] if key_kind=='payload' else pr['source_positions']
                qstart=max(keys)+1
                for hs in [list(range(i,min(i+8,model.cfg.n_heads))) for i in range(0,model.cfg.n_heads,8)]:
                    def hook(scores,hook):
                        for b,h in enumerate(hs):scores[b,h,qstart:,keys]=-torch.inf
                        return scores
                    x=cache[names[layer]].expand(len(hs),-1,-1).clone()
                    logits=model.run_with_hooks(x,start_at_layer=layer,fwd_hooks=[(f'blocks.{layer}.attn.hook_attn_scores',hook),LAST])[:,-1]
                    for h,ll in zip(hs,logits):record['head_screen'].append({'layer':layer,'head':h,'key_kind':key_kind,**metric(tok,label_ids,ll,clean,pr,basep)})
            if layer%7==0:print('HEAD SCREEN',task,layer,'seconds',round(time.time()-start),flush=True)
        record['head_screen'].sort(key=lambda x:x['kl']);save()
        print('TOP HEADS',task,[(p['layer'],p['head'],p['key_kind'],round(p['kl'],4),round(p['target_p'],3)) for p in record['head_screen'][:12]],flush=True)
    output['done']=True;save();return output

"""Causal localization before semantic feature selection. Two-way male/female patches."""
import json,time,importlib.metadata
from trace_baseline import ROOT,save_json,TARGETS,checkpoint_manifest
from ablate_single_features import summary_distribution
import torch
from transformer_lens import HookedTransformer

OUT=ROOT/'gender_transport'

def metrics(logits,model,ids):
    a=summary_distribution(logits[0,-1],model.tokenizer,ids)
    a['margin']=float(logits[0,-1,ids[' queen']]-logits[0,-1,ids[' king']])
    return a

def replace_hook(donor,scope='all',head=None,axis=1):
    def patch(x,hook):
        out=x.clone()
        selector=[slice(None)]*x.ndim
        if scope!='all':selector[axis]=scope
        if head is not None:selector[2 if axis==1 else 1]=head
        sel=tuple(selector);out[sel]=donor[sel]
        return out
    return patch

@torch.inference_mode()
def main():
    started=time.time();OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0)
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval()
    ids={w:model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
    cache={};tokens={};base={};logits={}
    for sex in ['female','male']:
        tokens[sex]=model.to_tokens(f'A {sex} monarch is called a',prepend_bos=False)
        logits[sex],cache[sex]=model.run_with_cache(tokens[sex])
        base[sex]=metrics(logits[sex],model,ids)
    assert tokens['female'].shape==tokens['male'].shape==(1,6)
    assert (tokens['female']!=tokens['male']).nonzero().tolist()==[[0,1]]
    save_json(OUT/'baseline.json',dict(prompts={s:model.tokenizer.decode(t[0]) for s,t in tokens.items()},
        tokens={s:t.tolist() for s,t in tokens.items()},metrics=base,
        female_minus_male_margin=base['female']['margin']-base['male']['margin']))
    torch.save({s:{k:v.cpu() for k,v in c.items()} for s,c in cache.items()},OUT/'clean_caches.pt')
    rows=[];checks=[]
    def run(base_sex,name,hooks,stage,**metadata):
        donor='male' if base_sex=='female' else 'female'
        edited=model.run_with_hooks(tokens[base_sex],fwd_hooks=hooks)
        m=metrics(edited,model,ids)
        delta=m['margin']-base[base_sex]['margin']
        row=dict(base=base_sex,donor=donor,name=name,stage=stage,metrics=m,delta_margin=delta,
                 donor_effect_fraction=delta/(base[donor]['margin']-base[base_sex]['margin']),**metadata)
        rows.append(row)
        return row,edited
    stages=[('q','attn.hook_q'),('k','attn.hook_k'),('pattern','attn.hook_pattern'),
            ('v','attn.hook_v'),('attn_out','hook_attn_out'),('mlp_out','hook_mlp_out'),
            ('resid_pre','hook_resid_pre'),('resid_post','hook_resid_post')]
    for sex in ['female','male']:
        donor='male' if sex=='female' else 'female'
        for layer in range(12):
            for name,short in stages:
                key=f'blocks.{layer}.{short}'
                axis=2 if name=='pattern' else 1
                for scope in ['all',1,2,5]:
                    run(sex,name,[(key,replace_hook(cache[donor][key],scope,axis=axis))],
                        'layer_scan',layer=layer,scope=scope)
            # Exact routing/content crossing at this layer, all positions.
            ak=f'blocks.{layer}.attn.hook_pattern';vk=f'blocks.{layer}.attn.hook_v'
            joint,lj=run(sex,'pattern_and_v',[(ak,replace_hook(cache[donor][ak])),(vk,replace_hook(cache[donor][vk]))],
                         'layer_scan',layer=layer,scope='all')
            target=next(r for r in reversed(rows) if r['base']==sex and r['name']=='attn_out' and r.get('layer')==layer and r.get('scope')=='all')
            err=abs(joint['metrics']['margin']-target['metrics']['margin'])
            assert err<1e-4,err
            checks.append(dict(base=sex,layer=layer,pattern_v_vs_attn_out_margin_error=err))
            if layer==0:
                r=next(r for r in rows if r['base']==sex and r['name']=='resid_pre' and r['layer']==0 and r['scope']=='all')
                assert abs(r['donor_effect_fraction']-1)<1e-5
            best=max([r for r in rows if r['base']==sex and r.get('layer')==layer and r['name'] not in ['resid_pre','resid_post']],key=lambda r:r['donor_effect_fraction'])
            print('LAYER',sex,layer,best['name'],best['scope'],round(best['donor_effect_fraction'],3),flush=True)
            save_json(OUT/'localization.json',rows)
    # Rank all layers using paired all-position counterfactual attention effects.
    ranks=[]
    for l in range(12):
        rr=[r for r in rows if r['name']=='attn_out' and r['scope']=='all' and r['layer']==l]
        ranks.append(dict(layer=l,mean_fraction=sum(r['donor_effect_fraction'] for r in rr)/2))
    ranks.sort(key=lambda r:-r['mean_fraction'])
    chosen=[r['layer'] for r in ranks[:3]]
    save_json(OUT/'selected_layers.json',dict(ranking=ranks,selected=chosen,selection='Three largest mean bidirectional all-position attention-output donor effects, before head/edge scan'))
    print('SELECTED',chosen,flush=True)
    # Resolve every head in the selected layers, then every source/query edge in
    # the two strongest bidirectional heads per layer. These are oracle patches,
    # not SAE/FRA interventions.
    for l in chosen:
        for sex in ['female','male']:
            donor='male' if sex=='female' else 'female'
            for h in range(12):
                for name,short in [('pattern','attn.hook_pattern'),('v','attn.hook_v'),('z','attn.hook_z')]:
                    key=f'blocks.{l}.{short}';axis=2 if name=='pattern' else 1
                    for scope in ['all',5]:
                        run(sex,name,[(key,replace_hook(cache[donor][key],scope,h,axis))],
                            'head_scan',layer=l,head=h,scope=scope)
        ranked=[]
        for h in range(12):
            rr=[r for r in rows if r['stage']=='head_scan' and r['layer']==l and r['head']==h and r['name']=='z' and r['scope']=='all']
            ranked.append((sum(r['donor_effect_fraction'] for r in rr)/2,h))
        selected_heads=[h for v,h in sorted(ranked,reverse=True)[:2]]
        print('HEADS',l,selected_heads,flush=True)
        for h in selected_heads:
            for sex in ['female','male']:
                donor='male' if sex=='female' else 'female'
                A=cache[sex][f'blocks.{l}.attn.hook_pattern'];Ad=cache[donor][f'blocks.{l}.attn.hook_pattern']
                V=cache[sex][f'blocks.{l}.attn.hook_v'];Vd=cache[donor][f'blocks.{l}.attn.hook_v']
                zk=f'blocks.{l}.attn.hook_z'
                for q in range(1,6):
                    for k in range(q+1):
                        for mode in ['content','whole_message']:
                            diff=(A[0,h,q,k]*(Vd[0,k,h]-V[0,k,h]) if mode=='content' else Ad[0,h,q,k]*Vd[0,k,h]-A[0,h,q,k]*V[0,k,h])
                            def edge(x,hook,q=q,h=h,diff=diff):
                                out=x.clone();out[:,q,h]+=diff;return out
                            run(sex,mode,[(zk,edge)],'edge_scan',layer=l,head=h,query=q,key=k,
                                clean_attention=float(A[0,h,q,k]),donor_attention=float(Ad[0,h,q,k]))
        save_json(OUT/'localization.json',rows)
    # Self-patch controls at all layers on a representative hook.
    for l in range(12):
        key=f'blocks.{l}.attn.hook_v'
        out=model.run_with_hooks(tokens['female'],fwd_hooks=[(key,replace_hook(cache['female'][key]))])
        assert torch.equal(out,logits['female'])
    save_json(OUT/'checks.json',dict(identical_prefix_except_gender_token=True,
        embedding_residual_patch_recovers_donor=True,self_patches_exact_noops=True,
        pattern_value_factorization_checks=checks))
    save_json(OUT/'localization.json',rows)
    save_json(OUT/'manifest.json',dict(model='gpt2-small',layers=list(range(12)),
        no_bos=True,source_position=1,answer_position=5,causal_runs=len(rows),
        patch_definition='Replace clean-base activations with aligned opposite-gender donor activations; rerun downstream. Edge content patches use the base pattern and donor-minus-base V. Whole-message patches also change that edge weight without renormalizing other edges.',
        metric='logit( queen)-logit( king); donor fraction relative to full male/female input change, not necessarily additive or bounded by [0,1]',
        selected_layers=chosen,elapsed_seconds=time.time()-started,
        packages={n:importlib.metadata.version(n) for n in ['torch','transformer-lens','sae-lens']},
        model_files=checkpoint_manifest('gpt2')))
    print('DONE',len(rows),round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()

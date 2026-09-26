"""Held-out clean-story loss recovered by each retrained Figure 2 SAE."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from sae_models import TopKSAE
from sleeper_utils import load_sleeper_model

R=Path('/data/users/dmitry/fra_table3_retrain_20260925')
HOOKS=[f'blocks.{layer}.ln1.hook_normalized' if kind=='ln1'
       else f'blocks.{layer}.hook_{kind}'
       for layer in range(4) for kind in ('ln1','resid_mid','resid_post')]
BS=20

@torch.inference_mode()
def sum_nll(logits,tokens,marker):
    b,t= tokens.shape
    target=tokens[:,1:]
    nll=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),
                        target.reshape(-1),reduction='none').reshape(b,t-1)
    pos=torch.arange(1,t,device=tokens.device)[None]
    mask=pos>marker[:,None]
    return float(nll[mask].sum()),int(mask.sum())

@torch.inference_mode()
def eval_hook(model,tokens,marker,hook,patch):
    total=0.;count=0
    for start in range(0,len(tokens),BS):
        t=tokens[start:start+BS].cuda()
        m=marker[start:start+BS].cuda()
        logits=model.run_with_hooks(t,fwd_hooks=[(hook,patch)],return_type='logits')
        s,n=sum_nll(logits,t,m)
        total+=s;count+=n
    return total/count

@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    data=torch.load(R/'fresh_holdout.pt',map_location='cpu',weights_only=True)
    clean=~data['is_deployment']
    tokens=data['tokens'][clean]
    marker=data['marker'][clean]
    model=load_sleeper_model('cuda')
    sums={h:torch.zeros(768,dtype=torch.float64) for h in HOOKS}
    sqs={h:torch.zeros(768,dtype=torch.float64) for h in HOOKS}
    n_acts=0
    nll=0.;n_tokens=0
    for start in range(0,len(tokens),BS):
        t=tokens[start:start+BS].cuda()
        m=marker[start:start+BS].cuda()
        logits,cache=model.run_with_cache(t,names_filter=lambda h:h in HOOKS,return_type='logits')
        s,n=sum_nll(logits,t,m)
        nll+=s;n_tokens+=n
        n_acts+=t.numel()
        for h in HOOKS:
            x=cache[h].double()
            sums[h]+=x.sum((0,1)).cpu()
            sqs[h]+=x.square().sum((0,1)).cpu()
        print('baseline batch',start,flush=True)
    base=nll/n_tokens
    out={'dataset':'fresh_holdout.pt clean subset','n_sequences':len(tokens),
         'n_story_tokens':n_tokens,'loss_definition':'Mean next-token CE on target positions after Story: marker, including first token',
         'recovery_definition':'(ablation CE - SAE reconstruction CE) / (ablation CE - original CE)',
         'baseline_ce':base,'hooks':{}}
    for hid,h in enumerate(HOOKS):
        layer=hid//3
        kind=('ln1','resid_mid','resid_post')[hid%3]
        zero=eval_hook(model,tokens,marker,h,lambda x,hook:torch.zeros_like(x))
        mean=(sums[h]/n_acts).float().cuda()
        mean_loss=eval_hook(model,tokens,marker,h,lambda x,hook:mean.expand_as(x))
        denom=float((sqs[h]-sums[h].square()/n_acts).sum())
        result={'hook_id':hid,'hookpoint':h,'layer':layer,'kind':kind,
                'baseline_ce':base,'zero_ablation_ce':zero,
                'mean_ablation_ce':mean_loss,'activation_variance_sum':denom,
                'seeds':[]}
        for seed in range(6):
            ck=torch.load(R/'weights'/f'sae_L{layer}_{kind}_s{seed}.pt',
                          map_location='cpu',weights_only=True)
            sae=TopKSAE(768,1536,32).cuda().eval()
            sae.load_state_dict(ck['state_dict'])
            error_sq=[0.]
            def patch(x,hook):
                xhat,_=sae(x.reshape(-1,768))
                xhat=xhat.reshape_as(x)
                error_sq[0]+=float((xhat.double()-x.double()).square().sum())
                return xhat
            rec_loss=eval_hook(model,tokens,marker,h,patch)
            row={'seed':seed,'reconstruction_ce':rec_loss,
                 'delta_ce':rec_loss-base,
                 'zero_ablation_loss_recovered':(zero-rec_loss)/(zero-base),
                 'mean_ablation_loss_recovered':(mean_loss-rec_loss)/(mean_loss-base),
                 'activation_one_minus_fvu':1-error_sq[0]/denom/(3 if kind=='ln1' else 1)}
            result['seeds'].append(row)
            print('hook',hid,'seed',seed,'recon',rec_loss,'recovery',row['zero_ablation_loss_recovered'],flush=True)
        out['hooks'][str(hid)]=result
        (R/'sae_loss_recovered.json').write_text(json.dumps(out,indent=2)+'\n')
    print('completed',flush=True)

if __name__=='__main__':
    main()

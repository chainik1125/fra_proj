"""Functional and reconstruction controls for the L5 ln1 FRA measurement."""
import json
from measure_fra_ln1 import *

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval()
    sae=SAE.from_pretrained(RELEASE,SITE,device='cpu').eval()
    tids={w:model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
    rows=[]
    for sex in ['female','male']:
        tokens=model.to_tokens(f'A {sex} monarch is called a',prepend_bos=False)
        logits,cache=model.run_with_cache(tokens)
        y=cache[LN]
        z=sae.encode(y);recon=sae.decode(z);error=y-recon
        clean=metric(logits,model,tids)
        zpath=torch.zeros_like(cache[ZHOOK])
        zpath[:,2]=cache[PATTERN][:,:,2,1,None]*cache[f'blocks.{L}.attn.hook_v'][:,1]
        def reconstruction(x,hook):return recon
        def identity(x,hook):return recon+error
        def no_attn(x,hook):return torch.zeros_like(x)
        def drop_message(x,hook):return x-zpath
        def cut_edge(x,hook):
            out=x.clone();out[:,:,2,1]=-1e5;return out
        for name,hooks in [
            ('baseline',[]),('ln1_reconstruction_only',[(LN,reconstruction)]),
            ('ln1_error_preserving_identity',[(LN,identity)]),
            ('whole_layer5_attention_output_zero',[(AHOOK,no_attn)]),
            ('whole_female_to_monarch_message_zero',[(ZHOOK,drop_message)]),
            ('whole_monarch_to_female_score_mask',[(SCORE,cut_edge)])]:
            edited=model.run_with_hooks(tokens,fwd_hooks=hooks)
            m=metric(edited,model,tids)
            lp=logits[0,-1].log_softmax(-1);lpe=edited[0,-1].log_softmax(-1)
            err=maxerr(logits,edited)
            if name=='ln1_error_preserving_identity':assert err<2e-4
            rows.append(dict(example='definition_'+sex,condition=name,metrics=m,
                delta_queen_minus_king_logit=m['queen_minus_king_logit']-clean['queen_minus_king_logit'],
                answer_kl=float((lp.exp()*(lp-lpe)).sum()),max_logit_difference=err))
            print(sex,name,m['candidates'],rows[-1]['delta_queen_minus_king_logit'],flush=True)
    save_json(OUT/'controls.json',rows)

if __name__=='__main__':main()

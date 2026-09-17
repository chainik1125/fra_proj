"""Check RoPE, GQA head mapping, and pair terms against model activations."""
import sys
from pathlib import Path
import torch
from common import LAST,render
from data import suite
from operators import Operators,SAE_SPECS
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from sae_lens_wrapper import GemmaScopeSAE
from fra_helpers import get_qk_weights


@torch.inference_mode()
def audit(model):
    row=next(r for r in suite('tenants_long','calibration') if r['joint'] and r['poisoned'])
    row=render(model.tokenizer,row);tokens=torch.tensor([row['token_ids']],device='cuda');layers=list(SAE_SPECS)
    names=[f'blocks.{l}.hook_resid_pre' for l in layers]+[f'blocks.{l}.attn.hook_rot_{k}' for l in layers for k in ['q','k']]
    with model.hooks(fwd_hooks=[LAST]):_,cache=model.run_with_cache(tokens,names_filter=names)
    records=[]
    for l in layers:
        sae=GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False);op=Operators(model,{l:sae})
        x=cache[f'blocks.{l}.hook_resid_pre'][0].float();z=op.encode(l,x);rms=op.rms(x)
        qpos=row['q_positions'][-1];kpos=row['k_positions']['payload'];heads=[]
        xhat=z@sae.W_dec.float()+sae.b_dec.float();error=x-xhat
        for h in range(model.cfg.n_heads):
            wq,wk,bq,bk=get_qk_weights(model,l,h);q=(x[qpos]/rms[qpos])@wq.float()+bq.float();k=(x[kpos]/rms[kpos])@wk.float()+bk.float()
            qr=op.rotate(q[None],l,[qpos])[0,0];kr=op.rotate(k[None],l,[kpos])[0,0]
            actualq=cache[f'blocks.{l}.attn.hook_rot_q'][0,qpos,h].float();nk=cache[f'blocks.{l}.attn.hook_rot_k'].shape[2]
            actualk=cache[f'blocks.{l}.attn.hook_rot_k'][0,kpos,h*nk//model.cfg.n_heads].float()
            qerr=float((qr-actualq).norm()/actualq.norm());kerr=float((kr-actualk).norm()/actualk.norm())
            raw=float(qr@kr/model.blocks[l].attn.attn_scale);actual=float(actualq@actualk/model.blocks[l].attn.attn_scale)
            qi=int(z[qpos].argmax());ki=int(z[kpos].argmax())
            anchor=float(op.anchor_terms(l,h,[qi],[ki],z[qpos,[qi]],z[kpos,[ki]],rms[qpos],rms[kpos],qpos,kpos)[0,0])
            full=float(op.pair_delta(l,h,[(qi,ki)],z,rms)[qpos,kpos])
            assert qerr<.02 and kerr<.02 and abs(anchor-full)<1e-4+abs(anchor)*1e-4
            heads.append({'head':h,'q_rot_relative_error':qerr,'k_rot_relative_error':kerr,'raw_score_error':abs(raw-actual),
                          'pair_anchor_full_matrix_error':abs(anchor-full),'q_feature':qi,'k_feature':ki})
        records.append({'layer':l,'heads':heads,'reconstruction_relative_squared_error_excluding_bos':float((xhat[1:]-x[1:]).square().sum()/x[1:].square().sum()),
                        'sae_plus_error_max_error':float((xhat+error-x).abs().max())})
        print('EXTRA AUDIT',l,'max rotated Q/K error',max(max(h['q_rot_relative_error'],h['k_rot_relative_error']) for h in heads),flush=True)
        del sae,op
    return records

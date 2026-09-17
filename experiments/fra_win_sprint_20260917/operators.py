"""Native SAE steering and feature-pair subtraction before Gemma's soft-cap.

This module never uses prompt positions to gate an intervention. Positions are
used only by calibration to identify candidate feature pairs.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import types
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from fra_helpers import get_qk_weights, _extract_rope_params

SAE_SPECS={
    10:('gemma-scope-9b-it-res','layer_9/width_16k/average_l0_88'),
    17:('gemma-scope-9b-pt-res','layer_16/width_16k/average_l0_75'),
    21:('gemma-scope-9b-it-res','layer_20/width_16k/average_l0_91'),
    22:('gemma-scope-9b-pt-res','layer_21/width_16k/average_l0_66'),
    28:('gemma-scope-9b-pt-res','layer_27/width_16k/average_l0_65'),
    32:('gemma-scope-9b-it-res','layer_31/width_16k/average_l0_76'),
}


class Operators:
    def __init__(self,model,saes):
        self.model=model;self.saes=saes;self.projection_cache={}

    def encode(self,layer,x):return self.saes[layer].encode(x.float()).float()

    def rms(self,x):return (x.float().square().mean(-1)+self.model.cfg.eps).sqrt()

    def projections(self,layer,head,qi,ki):
        dec=self.saes[layer].W_dec.float();wq,wk,_,_=get_qk_weights(self.model,layer,head)
        return dec[qi]@wq.float(),dec[ki]@wk.float()

    def rotate(self,v,layer,positions):
        """v [feature, head_dim] -> [position, feature, head_dim]."""
        sin,cos,dim,adjacent=_extract_rope_params(self.model,layer)
        if sin is None:return v[None].expand(len(positions),-1,-1)
        vr=v[:,:dim];flip=vr.clone()
        if adjacent:flip[:,::2],flip[:,1::2]=-vr[:,1::2],vr[:,::2]
        else:
            n=dim//2;flip[:,:n],flip[:,n:]=-vr[:,n:],vr[:,:n]
        positions=torch.as_tensor(positions,device=v.device)
        rotated=vr[None]*cos[positions,None].float()+flip[None]*sin[positions,None].float()
        return torch.cat([rotated,v[None,:,dim:].expand(len(positions),-1,-1)],-1)

    def anchor_terms(self,layer,head,qi,ki,zq,zk,rmsq,rmsk,qpos,kpos,projected=None):
        q,k=projected if projected is not None else self.projections(layer,head,qi,ki)
        q=self.rotate(q,layer,[qpos])[0];k=self.rotate(k,layer,[kpos])[0]
        return (q@k.T)/self.model.blocks[layer].attn.attn_scale * zq[:,None]*zk[None,:]/(rmsq*rmsk)

    def pair_delta(self,layer,head,pairs,z,rms):
        """Sum raw feature-feature terms across every token pair."""
        seq=z.shape[0]
        if not pairs:return torch.zeros((seq,seq),device=z.device)
        key=(layer,head,tuple(map(tuple,pairs)))
        if key not in self.projection_cache:
            qi,ki=map(list,zip(*pairs));q,k=self.projections(layer,head,qi,ki)
            self.projection_cache[key]=(qi,ki,q,k)
        qi,ki,q,k=self.projection_cache[key]
        qr=self.rotate(q,layer,range(seq))*z[:,qi,None]/rms[:,None,None]
        kr=self.rotate(k,layer,range(seq))*z[:,ki,None]/rms[:,None,None]
        # Features share a column in both flattened operands: only matched
        # feature pairs contribute. No cross-pair terms are introduced.
        return (qr.flatten(1)@kr.flatten(1).T/self.model.blocks[layer].attn.attn_scale).tril()

    @contextmanager
    def cut_scores(self,layer,head_deltas,coefficients,placement='precap'):
        """Batch one layer's cut strengths; all deltas use its current input."""
        attn=self.model.blocks[layer].attn;original=attn.calculate_attention_scores
        coeff=torch.as_tensor(coefficients,device=attn.W_Q.device)[:,None,None]
        cap=attn.cfg.attn_scores_soft_cap
        def calculate(this,q,k):
            if placement=='postcap':scores=original(q,k)
            else:
                if k.shape[2]!=q.shape[2]:
                    assert q.shape[2]%k.shape[2]==0
                    k=torch.repeat_interleave(k,repeats=q.shape[2]//k.shape[2],dim=2)
                scores=q.permute(0,2,1,3)@k.permute(0,2,3,1)/this.attn_scale
            for head,delta in head_deltas.items():
                scores[:,head]-=(coeff*delta).to(scores.dtype)
            if placement=='precap' and cap>0:scores=cap*torch.tanh(scores/cap)
            return scores
        attn.calculate_attention_scores=types.MethodType(calculate,attn)
        try:yield
        finally:attn.calculate_attention_scores=original

    def audit(self,layer,x,cache):
        """Check reconstruction quality and actual Q/K projection arithmetic."""
        z=self.encode(layer,x);sae=self.saes[layer];rms=self.rms(x)
        recon=sae.decode(z).float();residual=x.float()-recon
        error=float((recon-x.float()).square().sum()/x.float().square().sum())
        report={'layer':layer,'l0':float((z[1:]!=0).sum(-1).float().mean()),'relative_squared_error_including_bos':error,
                'relative_squared_error_without_bos':float((recon[1:]-x[1:].float()).square().sum()/x[1:].float().square().sum())}
        # Decoder+error reconstructs x, so all four feature/error projection
        # products together must recover the full unnormalized QK product.
        head=0;wq,wk,bq,bk=get_qk_weights(self.model,layer,head)
        decomposed=(z@sae.W_dec.float()+sae.b_dec.float()+residual)/rms[:,None]
        q=decomposed@wq.float()+bq.float();k=decomposed@wk.float()+bk.float()
        actualq=cache[f'blocks.{layer}.attn.hook_q'][0,:,head].float()
        nk=cache[f'blocks.{layer}.attn.hook_k'].shape[2];kh=head*nk//self.model.cfg.n_heads
        actualk=cache[f'blocks.{layer}.attn.hook_k'][0,:,kh].float()
        report['q_relative_rmse']=float((q-actualq).square().mean().sqrt()/actualq.square().mean().sqrt())
        report['k_relative_rmse']=float((k-actualk).square().mean().sqrt()/actualk.square().mean().sqrt())
        report['q_max_abs_error']=float((q-actualq).abs().max());report['k_max_abs_error']=float((k-actualk).abs().max())
        assert report['q_relative_rmse']<.02 and report['k_relative_rmse']<.02,report
        return report

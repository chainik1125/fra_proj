"""Signed QK/OV FRA and causal reruns at GPT-2 attention block 5.

Input: L4 normalized residual dictionary transferred directly to L5 ln1.
Output feature readout: original L5 post-block SAE. No model/SAE training.
"""
import gzip
import hashlib
import json
import math
import time
from trace_baseline import ROOT, TARGETS, save_json, checkpoint_manifest
from prepare_ln1_transfer import OUT, RELEASE, SITE, LN
from ablate_single_features import summary_distribution
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

L=5
SCORE=f'blocks.{L}.attn.hook_attn_scores'
PATTERN=f'blocks.{L}.attn.hook_pattern'
ZHOOK=f'blocks.{L}.attn.hook_z'
AHOOK=f'blocks.{L}.hook_attn_out'
POST=f'blocks.{L}.hook_resid_post'
TARGET_FEATURES=[7671,24973,18603,22630,22931,23701,32492]

def maxerr(a,b): return float((a-b).abs().max())

def score_product(q,k,scale):
    return torch.einsum('qhd,khd->hqk',q,k)/scale

def metric(logits,model,target_ids):
    out=summary_distribution(logits[0,-1],model.tokenizer,target_ids)
    out['queen_minus_king_logit']=float(logits[0,-1,target_ids[' queen']]-logits[0,-1,target_ids[' king']])
    return out

class Trace:
    def __init__(self,model,sae,outsae,tokens,groups):
        self.model,self.sae,self.outsae,self.tokens=model,sae,outsae,tokens
        self.groups=groups
        with torch.no_grad():
            self.logits,self.cache=model.run_with_cache(tokens)
            self.y=self.cache[LN][0]
            z=sae.encode(self.y)
            self.std=sae.ln_std.clone()
            self.mu=sae.ln_mu.clone()
            self.recon=sae.decode(z)
            self.ids=sorted(z.any(0).nonzero().flatten().tolist())
            self.idx={i:j for j,i in enumerate(self.ids)}
            self.z=z[:,self.ids]
            self.a=self.z*self.std
            self.D=sae.W_dec[self.ids]
            self.bias=self.mu+self.std*sae.b_dec
            self.error=self.y-self.recon
            self.attn=model.blocks[L].attn
            self.scale=self.attn.attn_scale
            self.Q=torch.einsum('fd,hda->fha',self.D,self.attn.W_Q)[None]*self.a[:,:,None,None]
            self.K=torch.einsum('fd,hda->fha',self.D,self.attn.W_K)[None]*self.a[:,:,None,None]
            self.V=torch.einsum('fd,hda->fha',self.D,self.attn.W_V)[None]*self.a[:,:,None,None]
            self.q=self.cache[f'blocks.{L}.attn.hook_q'][0]
            self.k=self.cache[f'blocks.{L}.attn.hook_k'][0]
            self.v=self.cache[f'blocks.{L}.attn.hook_v'][0]
            self.A=self.cache[PATTERN][0]
            self.qb=torch.einsum('sd,hda->sha',self.bias,self.attn.W_Q)+self.attn.b_Q
            self.kb=torch.einsum('sd,hda->sha',self.bias,self.attn.W_K)+self.attn.b_K
            self.vb=torch.einsum('sd,hda->sha',self.bias,self.attn.W_V)+self.attn.b_V
            self.qe=torch.einsum('sd,hda->sha',self.error,self.attn.W_Q)
            self.ke=torch.einsum('sd,hda->sha',self.error,self.attn.W_K)
            self.ve=torch.einsum('sd,hda->sha',self.error,self.attn.W_V)
            self.group_idx={k:[self.idx[i] for i in ids if i in self.idx] for k,ids in groups.items()}
            self.Qg={k:self.Q[:,idx].sum(1) for k,idx in self.group_idx.items()}
            self.Kg={k:self.K[:,idx].sum(1) for k,idx in self.group_idx.items()}
            self.Vg={k:self.V[:,idx].sum(1) for k,idx in self.group_idx.items()}
            self.causal=torch.ones(tokens.shape[1],tokens.shape[1],dtype=torch.bool).tril()
            self.cross_rg=score_product(self.Qg['royalty'],self.Kg['gender'],self.scale).masked_fill(~self.causal,0)
            self.cross_gr=score_product(self.Qg['gender'],self.Kg['royalty'],self.scale).masked_fill(~self.causal,0)
            self.cross=(self.cross_rg+self.cross_gr).masked_fill(torch.eye(tokens.shape[1],dtype=torch.bool),0)
            self.output_z=outsae.encode(self.cache[POST][0])
            output_std=outsae.ln_std.clone()
            outsae.decode(self.output_z)
            encoder=outsae.W_enc[:,7671]
            self.queen_read=(encoder-encoder.mean())[None,:]/(output_std+1e-5)
        self.gradients()

    def gradients(self):
        stored={}
        def s_hook(s,hook):
            v=s.detach().requires_grad_(True);stored['scores']=v;return v
        def a_hook(a,hook):
            a.retain_grad();stored['attn']=a;return a
        ids={w:self.model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
        with torch.enable_grad():
            logits=self.model.run_with_hooks(self.tokens,fwd_hooks=[(SCORE,s_hook),(AHOOK,a_hook)])
            margin=logits[0,-1,ids[' queen']]-logits[0,-1,ids[' king']]
            margin.backward()
        self.score_grad=stored['scores'].grad[0].detach()
        self.attn_grad=stored['attn'].grad[0].detach()
        assert maxerr(logits.detach(),self.logits)<1e-5

    def score_delta(self,spec):
        kind=spec.get('qk')
        C=torch.zeros_like(self.cross)
        if kind=='cross': C=self.cross.clone()
        elif kind=='royalty_query_gender_key': C=self.cross_rg.clone().masked_fill(torch.eye(self.tokens.shape[1],dtype=torch.bool),0)
        elif kind in ['gender_involvement','royalty_involvement']:
            g=kind.split('_')[0]
            C=(score_product(self.Qg[g],self.k,self.scale)+score_product(self.q,self.Kg[g],self.scale)
               -score_product(self.Qg[g],self.Kg[g],self.scale)).masked_fill(~self.causal,0)
        elif kind=='atom':
            i,j=spec['query_feature'],spec['key_feature']
            if i in self.idx and j in self.idx:
                C=score_product(self.Q[:,self.idx[i]],self.K[:,self.idx[j]],self.scale).masked_fill(~self.causal,0)
        for axis,key in [(0,'head'),(1,'query'),(2,'key')]:
            if key in spec:
                mask=torch.zeros(C.shape[axis]);mask[spec[key]]=1
                shape=[1,1,1];shape[axis]=len(mask)
                C*=mask.reshape(shape)
        return C

    def v_selected(self,spec):
        kind=spec.get('ov')
        if kind in self.Vg: return self.Vg[kind]
        if kind=='atom':
            i=spec['value_feature']
            return self.V[:,self.idx[i]] if i in self.idx else torch.zeros_like(self.v)
        return torch.zeros_like(self.v)

    def route_gate(self,spec):
        gate=self.causal[None].expand_as(self.A).clone()
        for axis,key in [(0,'head'),(1,'query'),(2,'key')]:
            if key in spec:
                mask=torch.zeros(gate.shape[axis],dtype=torch.bool);mask[spec[key]]=True
                shape=[1,1,1];shape[axis]=len(mask)
                gate &= mask.reshape(shape)
        return gate

    @torch.no_grad()
    def run(self,spec,alpha=1.0,rescue=False):
        C=self.score_delta(spec)
        V=self.v_selected(spec)
        gate=self.route_gate(spec)
        observed={}
        def score_hook(x,hook): return x-alpha*C[None]
        def score_rescue(x,hook): return x+alpha*C[None]
        def pattern_hook(x,hook): observed['pattern']=x.detach().clone();return x
        def z_hook(x,hook):
            d=torch.einsum('hqk,kha->qha',observed['pattern'][0]*gate,V)
            observed['removed_z']=d
            return x-alpha*d[None]
        def z_rescue(x,hook): return x+alpha*observed['removed_z'][None]
        def post_hook(x,hook):
            z=self.outsae.encode(x);self.outsae.decode(z)
            observed['target_features']={str(i):z[0,:,i].tolist() for i in TARGET_FEATURES}
            return x
        hooks=[]
        if spec.get('qk'):
            hooks.append((SCORE,score_hook))
            if rescue:hooks.append((SCORE,score_rescue))
        hooks.append((PATTERN,pattern_hook))
        if spec.get('ov'):
            hooks.append((ZHOOK,z_hook))
            if rescue:hooks.append((ZHOOK,z_rescue))
        hooks.append((POST,post_hook))
        logits=self.model.run_with_hooks(self.tokens,fwd_hooks=hooks)
        return logits,observed

    @torch.no_grad()
    def audit(self):
        qhat=self.Q.sum(1)+self.qb+self.qe
        khat=self.K.sum(1)+self.kb+self.ke
        vhat=self.V.sum(1)+self.vb+self.ve
        shat=score_product(qhat,khat,self.scale)
        scores=self.cache[SCORE][0]
        zhat=torch.einsum('hqk,kha->qha',self.A,vhat)
        ahat=torch.einsum('qha,had->qd',zhat,self.attn.W_O)+self.attn.b_O
        ledger={}
        for qname,qs in [('features',self.Q.sum(1)),('bias',self.qb),('error',self.qe)]:
            for kname,ks in [('features',self.K.sum(1)),('bias',self.kb),('error',self.ke)]:
                ledger[qname+'__'+kname]=score_product(qs,ks,self.scale).tolist()
        ovledger={}
        for name,vs in [('features',self.V.sum(1)),('bias',self.vb),('error',self.ve)]:
            out=torch.einsum('hqk,kha,had->qd',self.A,vs,self.attn.W_O)
            ovledger[name]=dict(norm=float(out.norm()),queen_preactivation=(out*self.queen_read).sum(-1).tolist())
        # Frozen component insertion mixed difference equals the FRA score cross term.
        qr,kg=self.Qg['royalty'],self.Kg['gender']
        mixed=(score_product(self.q,self.k,self.scale)-score_product(self.q-qr,self.k,self.scale)
               -score_product(self.q,self.k-kg,self.scale)+score_product(self.q-qr,self.k-kg,self.scale))
        mixed_error=maxerr(mixed[:,self.causal],self.cross_rg[:,self.causal])
        errors=dict(input_reconstruction=maxerr(self.a@self.D+self.bias+self.error,self.y),
            q_reconstruction=maxerr(qhat,self.q),k_reconstruction=maxerr(khat,self.k),
            v_reconstruction=maxerr(vhat,self.v),
            score_reconstruction=maxerr(shat[:,self.causal],scores[:,self.causal]),
            z_reconstruction=maxerr(zhat,self.cache[ZHOOK][0]),
            attention_output_reconstruction=maxerr(ahat,self.cache[AHOOK][0]),
            qk_mixed_difference_error=mixed_error)
        assert max(errors.values())<1e-4,errors
        # Exact clean-post-block preactivation ledger: skip + attention + MLP + bias.
        x=self.cache[POST][0];d=self.queen_read
        zs=self.outsae.encode(x);sigma=self.outsae.ln_std.clone();mu=self.outsae.ln_mu.clone()
        pre=(((x-mu)/(sigma+1e-5)-self.outsae.b_dec)@self.outsae.W_enc[:,7671]+self.outsae.b_enc[7671])
        const=self.outsae.b_enc[7671]-self.outsae.b_dec@self.outsae.W_enc[:,7671]
        preledger={name:(self.cache[hook][0]*d).sum(-1).tolist() for name,hook in [
            ('skip',f'blocks.{L}.hook_resid_pre'),('attention',AHOOK),('mlp',f'blocks.{L}.hook_mlp_out')]}
        total=sum(torch.tensor(v) for v in preledger.values())+const
        errors['post_sae_preactivation_ledger']=maxerr(total,pre)
        assert errors['post_sae_preactivation_ledger']<1e-4
        self.outsae.decode(zs)
        return dict(errors=errors,
            ln1_relative_squared_error=float(self.error.square().sum()/self.y.square().sum()),
            qk_ledger=ledger,ov_ledger=ovledger,
            post_queen_preactivation=pre.tolist(),post_queen_preactivation_ledger=preledger,
            post_queen_preactivation_bias=float(const),
            observed_scores=scores.masked_fill(~self.causal,0).tolist(),attention_pattern=self.A.tolist(),
            cross_rg=self.cross_rg.tolist(),cross_gr=self.cross_gr.tolist(),
            score_grad=self.score_grad.tolist())

    @torch.no_grad()
    def atoms(self):
        qk,ov=[],[]
        S=self.tokens.shape[1]
        for q in range(S):
            for k in range(q+1):
                if q!=k:
                    for direction,gq,gk in [('royalty_query_gender_key','royalty','gender'),('gender_query_royalty_key','gender','royalty')]:
                        for i in self.group_idx[gq]:
                            if self.a[q,i]==0:continue
                            for j in self.group_idx[gk]:
                                if self.a[k,j]==0:continue
                                c=(self.Q[q,i]*self.K[k,j]).sum(-1)/self.scale
                                for h in range(c.shape[0]):
                                    qk.append(dict(direction=direction,head=h,query=q,key=k,
                                        query_feature=self.ids[i],key_feature=self.ids[j],
                                        query_activation=float(self.z[q,i]),key_activation=float(self.z[k,j]),
                                        score_contribution=float(c[h]),attention=float(self.A[h,q,k]),
                                        margin_first_order_contribution=float(c[h]*self.score_grad[h,q,k])))
                for group in ['royalty','gender']:
                    for i in self.group_idx[group]:
                        if self.a[k,i]==0:continue
                        vec=torch.einsum('ha,had->hd',self.V[k,i],self.attn.W_O)*self.A[:,q,k,None]
                        for h in range(vec.shape[0]):
                            ov.append(dict(group=group,head=h,query=q,key=k,value_feature=self.ids[i],
                                activation=float(self.z[k,i]),attention=float(self.A[h,q,k]),
                                message_norm=float(vec[h].norm()),
                                queen_preactivation_contribution=float(vec[h]@self.queen_read[q]),
                                margin_first_order_contribution=float(vec[h]@self.attn_grad[q])))
        return qk,ov


def standard_specs():
    specs=[dict(name='baseline')]
    for qk in ['cross','royalty_query_gender_key','gender_involvement','royalty_involvement']:
        specs.append(dict(name='qk_'+qk,qk=qk))
    for ov in ['gender','royalty','both']:
        specs.append(dict(name='ov_'+ov,ov=ov))
    specs += [dict(name='qk_cross_monarch_female',qk='cross',query=2,key=1),
              dict(name='ov_gender_female_to_monarch',ov='gender',query=2,key=1),
              dict(name='ov_gender_female_to_answer',ov='gender',query=5,key=1),
              dict(name='ov_royalty_monarch_to_answer',ov='royalty',query=5,key=2),
              dict(name='qk_cross_plus_ov_gender_monarch_female',qk='cross',ov='gender',query=2,key=1),
              dict(name='qk_cross_plus_ov_gender_all',qk='cross',ov='gender')]
    return specs


def main():
    start=time.time();torch.set_num_threads(4);torch.manual_seed(0)
    groupbytes=(OUT/'feature_groups.json').read_bytes();groupmeta=json.loads(groupbytes)
    groups={k:groupmeta[k] for k in ['royalty','gender','both']}
    sae=SAE.from_pretrained(RELEASE,SITE,device='cpu').eval()
    outsae=SAE.from_pretrained(RELEASE,POST,device='cpu').eval()
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval()
    for m in [sae,outsae,model]:
        for p in m.parameters():p.requires_grad_(False)
    tids={w:model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
    examples={e['id']:e for e in json.loads((ROOT/'teacher_forced/baseline.json').read_text())['examples']}
    all_results=[];selected=[];audits={};atomsets={};feature_rows={};tensors={}
    for name in ['definition_female','definition_male']:
        e=examples[name];tokens=torch.tensor([e['token_ids'][:e['prompt_length']]])
        tr=Trace(model,sae,outsae,tokens,groups)
        audit=tr.audit();audits[name]=audit
        clean=metric(tr.logits,model,tids)
        assert max(abs(clean['candidates'][w]-e['next_token'][e['prompt_length']-1]['candidates'][w]) for w in TARGETS)<2e-6
        qkatoms,ovatoms=tr.atoms();atomsets[name]=dict(qk=qkatoms,ov=ovatoms)
        feature_rows[name]=[[(i,float(tr.z[p,j])) for j,i in enumerate(tr.ids) if float(tr.z[p,j])>0] for p in range(tokens.shape[1])]
        specs=standard_specs()
        if name=='definition_female':
            for base in [s for s in specs if s['name'] in ['qk_cross_monarch_female','ov_gender_female_to_monarch','ov_gender_female_to_answer','ov_royalty_monarch_to_answer']]:
                for h in range(model.cfg.n_heads):
                    selected.append(dict(base,name=base['name']+f'_head{h}',head=h))
            qrank=[]
            for key in ['score_contribution','margin_first_order_contribution']:
                qrank+=sorted(qkatoms,key=lambda r:-abs(r[key]))[:5]
            orank=sorted(ovatoms,key=lambda r:-abs(r['margin_first_order_contribution']))[:5]
            orank+=sorted([r for r in ovatoms if r['query']==2 and r['key']==1],key=lambda r:-abs(r['queen_preactivation_contribution']))[:5]
            for r in qrank:
                s=dict(name=f"qk_atom_h{r['head']}_q{r['query']}_k{r['key']}_{r['query_feature']}_{r['key_feature']}",qk='atom',
                       **{k:r[k] for k in ['head','query','key','query_feature','key_feature']})
                if not any(t['name']==s['name'] for t in selected):selected.append(s)
            for r in orank:
                s=dict(name=f"ov_atom_h{r['head']}_q{r['query']}_k{r['key']}_{r['value_feature']}",ov='atom',
                       **{k:r[k] for k in ['head','query','key','value_feature']})
                if not any(t['name']==s['name'] for t in selected):selected.append(s)
            save_json(OUT/'selected_path_specs.json',dict(selection='Female-prompt attribution ranks, before per-path causal outcomes; exploratory, not held out',specs=selected))
        specs += selected
        for spec in specs:
            with torch.no_grad(): logits,obs=tr.run(spec)
            m=metric(logits,model,tids)
            diff=m['queen_minus_king_logit']-clean['queen_minus_king_logit']
            row=dict(example=name,spec=spec,metrics=m,delta_queen_minus_king_logit=diff,
                     attention_monarch_to_female=obs['pattern'][0,:,2,1].tolist(),
                     attention_answer_to_female=obs['pattern'][0,:,5,1].tolist(),
                     output_features=obs['target_features'])
            all_results.append(row)
            if spec in standard_specs():
                print('RESULT',name,spec['name'],'delta_margin',round(diff,6),'pqueen',round(m['candidates'][' queen'],6),flush=True)
        rescuelogits,_=tr.run(dict(qk='cross',ov='gender'),rescue=True)
        audit['errors']['qk_ov_rescue_logits']=maxerr(rescuelogits,tr.logits)
        assert audit['errors']['qk_ov_rescue_logits']<2e-4
        # Finite differences validate signed gradient attributions, not large edits.
        fd=[]
        for mode,atoms in [('qk',qkatoms),('ov',ovatoms)]:
            r=max(atoms,key=lambda r:abs(r['margin_first_order_contribution']))
            spec=dict(head=r['head'],query=r['query'],key=r['key'])
            if mode=='qk':spec.update(qk='atom',query_feature=r['query_feature'],key_feature=r['key_feature'])
            else:spec.update(ov='atom',value_feature=r['value_feature'])
            plus,_=tr.run(spec,alpha=0.01);minus,_=tr.run(spec,alpha=-0.01)
            slope=(metric(plus,model,tids)['queen_minus_king_logit']-metric(minus,model,tids)['queen_minus_king_logit'])/0.02
            predicted=-r['margin_first_order_contribution']
            assert abs(slope-predicted)<0.002,(slope,predicted)
            fd.append(dict(mode=mode,observed_derivative=slope,predicted_derivative=predicted))
        audit['finite_difference_checks']=fd
        tensors[name]=dict(normalized_input=tr.y,sae_reconstruction=tr.recon,sae_error=tr.error,
            feature_ids=tr.ids,coefficients=tr.z,scaled_coefficients=tr.a,
            Q=tr.Q,K=tr.K,V=tr.V,attention=tr.A,score_grad=tr.score_grad,attn_out_grad=tr.attn_grad)
        save_json(OUT/'results.json',all_results);save_json(OUT/'audit.json',audits)
        save_json(OUT/'atoms.json.gz',atomsets);save_json(OUT/'active_features.json',feature_rows)
        print('FINISHED',name,flush=True)
    torch.save(tensors,OUT/'decomposition_tensors.pt')
    save_json(OUT/'manifest.json',dict(model='gpt2-small',attention_layer=L,attention_input=LN,
        input_sae_original_site=SITE,input_sae_release=RELEASE,input_sae_transferred=True,
        output_sae_site=POST,source_sae_config=sae.cfg.to_dict(),
        feature_groups_sha256=hashlib.sha256(groupbytes).hexdigest(),
        exactness='Exact clean-score and fixed-pattern OV decomposition when all SAE features, decoder bias, reconstruction error, and model biases are retained. No approximation of LayerNorm in QK/OV interventions: edits occur after ln1.',
        output_feature_projection='Signed contribution to clean post-block SAE preactivation under its observed normalization. MLP and skip contributions are separate; causal reruns recompute normalization and TopK.',
        intervention_scope='One attention layer, all positions for group tests; explicit head/query/key gates for selected paths. Values preserved in QK-only edits; pattern preserved in OV-only edits. Combined OV edit uses intervened pattern.',
        target='logit( queen)-logit( king) at final prefix token; no future queen token in model input',
        causal_runs=len(all_results),elapsed_seconds=time.time()-start,
        sae_files=checkpoint_manifest('jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs')))
    print('DONE',len(all_results),'runs',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':main()

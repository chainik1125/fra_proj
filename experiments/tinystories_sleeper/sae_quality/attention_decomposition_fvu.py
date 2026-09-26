"""Figure 2 panel 3: exact SAE-error decomposition of TinyStories attention."""
import json
import math
from pathlib import Path
import torch
from sae_models import TopKSAE
from sleeper_utils import load_sleeper_model

ROOT = Path('/archive/fra_table3_retrain_20260925')
OUT = ROOT / 'attention_decomposition'
OUT.mkdir(exist_ok=True)
CASES = ('exact', 'sae_only', 'drop_q_error', 'drop_k_error',
         'drop_qk_error_error', 'drop_ov_error')
SCOPES = ('all_prompt', 'clean_marker', 'deployment_marker')

class Stats:
    def __init__(self):
        self.n = 0
        self.y_sum = None
        self.y_sq = None
        self.err = {case: 0.0 for case in CASES}
    def add(self, y, preds):
        y = y.double()
        self.n += y.shape[0]
        s, sq = y.sum(0).cpu(), y.square().sum(0).cpu()
        self.y_sum = s if self.y_sum is None else self.y_sum + s
        self.y_sq = sq if self.y_sq is None else self.y_sq + sq
        for case, pred in preds.items():
            self.err[case] += float((pred.double() - y).square().sum())
    def result(self):
        var = float((self.y_sq - self.y_sum.square() / self.n).sum())
        return {'n_positions': self.n, 'target_variance_sum': var,
                'fvu': {k: v / var for k, v in self.err.items()},
                'one_minus_fvu': {k: 1 - v / var for k, v in self.err.items()}}

def project(x, w, b=None):
    z = torch.einsum('btd,hdf->bthf', x, w)
    return z if b is None else z + b

def scores(q, k, scale):
    return torch.einsum('bqhd,bkhd->bhqk', q, k) / scale

def out_from(s, v, wo, bo, mask):
    p = s.masked_fill(~mask, torch.finfo(s.dtype).min).softmax(dim=-1)
    z = torch.einsum('bhqk,bkhd->bqhd', p, v)
    y = torch.einsum('bqhd,hdf->bqf', z, wo) + bo
    return y, p

@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    data = torch.load(ROOT / 'fresh_holdout.pt', map_location='cpu', weights_only=True)
    tokens, markers, deploy = data['tokens'], data['marker'], data['is_deployment']
    model = load_sleeper_model('cuda')
    summary = {'dataset': 'fresh_holdout.pt', 'n_sequences': len(tokens),
               'n_clean': int((~deploy).sum()), 'n_deployment': int(deploy.sum()),
               'definition': '1 - SSE(candidate attention output, exact output) / sum ||exact output - channel mean||^2',
               'scopes': list(SCOPES), 'seeds': list(range(6)), 'layers': {}}
    for layer in range(4):
        wq, wk, wv, wo = [getattr(model, n)[layer].float() for n in ('W_Q','W_K','W_V','W_O')]
        bq, bk, bv, bo = [getattr(model, n)[layer].float() for n in ('b_Q','b_K','b_V','b_O')]
        hook = f'blocks.{layer}.ln1.hook_normalized'
        prefix = f'blocks.{layer}.attn.'
        names = {hook, prefix+'hook_q', prefix+'hook_k', prefix+'hook_v',
                 prefix+'hook_pattern', f'blocks.{layer}.hook_attn_out'}
        saes = []
        for seed in range(6):
            payload = torch.load(ROOT/'weights'/f'sae_L{layer}_ln1_s{seed}.pt',
                                 map_location='cpu', weights_only=True)
            assert payload['config']['layer_hook'] == hook
            sae = TopKSAE(768, 1536, 32).cuda().eval()
            sae.load_state_dict(payload['state_dict'])
            saes.append(sae)
        stats = [{scope: Stats() for scope in SCOPES} for _ in range(6)]
        identities = {'q':0., 'k':0., 'v':0., 'pattern':0., 'output':0.}
        for start in range(0, len(tokens), 8):
            end = min(start+8, len(tokens))
            _, cache = model.run_with_cache(tokens[start:end].cuda(),
                names_filter=lambda name: name in names, return_type=None)
            x = cache[hook].float()
            q0, k0, v0 = [cache[prefix+'hook_'+n].float() for n in ('q','k','v')]
            pat0 = cache[prefix+'hook_pattern'].float()
            y0 = cache[f'blocks.{layer}.hook_attn_out'].float()
            b, t, d = x.shape
            scale = math.sqrt(model.cfg.d_head) if model.cfg.use_attn_scale else 1.0
            mask = torch.ones((t,t),device='cuda',dtype=torch.bool).tril()[None,None]
            ix = torch.arange(t,device='cuda')[None]
            marker = markers[start:end].cuda()
            dep = deploy[start:end].cuda()
            masks = {'all_prompt': ix <= marker[:,None],
                     'clean_marker': (~dep)[:,None] & (ix == marker[:,None]),
                     'deployment_marker': dep[:,None] & (ix == marker[:,None])}
            for seed, sae in enumerate(saes):
                xhat, _ = sae(x.reshape(-1,d))
                xhat = xhat.reshape_as(x)
                err = x - xhat
                qc,kc,vc = (project(xhat,w,bias) for w,bias in
                            ((wq,bq),(wk,bk),(wv,bv)))
                qe,ke,ve = (project(err,w) for w in (wq,wk,wv))
                if seed == 0:
                    for label, val, ref in (('q',qc+qe,q0),('k',kc+ke,k0),('v',vc+ve,v0)):
                        identities[label] = max(identities[label],float((val-ref).abs().max()))
                cc = scores(qc,kc,scale)
                q_error = scores(qe,kc,scale)
                k_error = scores(qc,ke,scale)
                ee = scores(qe,ke,scale)
                full = cc + q_error + k_error + ee
                vals = vc + ve
                exact, p = out_from(full,vals,wo,bo,mask)
                if seed == 0:
                    identities['pattern'] = max(identities['pattern'],float((p-pat0).abs().max()))
                    identities['output'] = max(identities['output'],float((exact-y0).abs().max()))
                preds = {'exact':exact,
                    'sae_only':out_from(cc,vc,wo,bo,mask)[0],
                    'drop_q_error':out_from(full-q_error,vals,wo,bo,mask)[0],
                    'drop_k_error':out_from(full-k_error,vals,wo,bo,mask)[0],
                    'drop_qk_error_error':out_from(full-ee,vals,wo,bo,mask)[0],
                    'drop_ov_error':out_from(full,vc,wo,bo,mask)[0]}
                for scope, use in masks.items():
                    if use.any():
                        stats[seed][scope].add(y0[use], {k:v[use] for k,v in preds.items()})
            print(f'layer={layer} batch={start}:{end}',flush=True)
            del cache
        layer_result = {'layer':layer, 'identity_max_abs':identities,
                        'seeds':[{scope:st.result() for scope,st in seed_stats.items()}
                                 for seed_stats in stats]}
        (OUT/f'layer{layer}.json').write_text(json.dumps(layer_result,indent=2)+'\n')
        summary['layers'][str(layer)] = layer_result
        print('completed',layer,'identity',identities,flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__ == '__main__':
    main()

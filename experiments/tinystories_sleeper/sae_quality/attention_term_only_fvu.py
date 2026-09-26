"""Measure each FRA attention term alone on the fresh deployment holdout.

Runs on gpu-host-2 beside the retrained SAE weights. QK terms are compared with
the exact pre-softmax QK scores; OV terms are compared with the exact attention
output after applying the frozen full-model attention pattern. For reference we
also calculate the output produced when each QK term alone supplies the logits
while the exact OV values are retained. The latter is a different estimand.
"""

import json
import math
from pathlib import Path

import torch

from sae_models import TopKSAE
from sleeper_utils import load_sleeper_model


ROOT = Path('/archive/fra_table3_retrain_20260925')
OUT = ROOT / 'attention_decomposition' / 'term_only_fvu.json'
QK_CASES = ('coherent', 'query_error', 'key_error', 'error_error')
OV_CASES = ('coherent', 'error')


class Stats:
    def __init__(self, cases):
        self.n = 0
        self.y_sum = None
        self.y_sq = None
        self.sse = {case: 0.0 for case in cases}
        self.pred_sum = {case: None for case in cases}

    def add(self, y, preds):
        y = y.double()
        self.n += y.shape[0]
        y_sum = y.sum(0).cpu()
        y_sq = y.square().sum(0).cpu()
        self.y_sum = y_sum if self.y_sum is None else self.y_sum + y_sum
        self.y_sq = y_sq if self.y_sq is None else self.y_sq + y_sq
        for case, pred in preds.items():
            pred = pred.double()
            self.sse[case] += float((pred - y).square().sum())
            pred_sum = pred.sum(0).cpu()
            self.pred_sum[case] = (pred_sum if self.pred_sum[case] is None
                                   else self.pred_sum[case] + pred_sum)

    def result(self):
        variance = float((self.y_sq - self.y_sum.square() / self.n).sum())
        centered_sse = {
            case: self.sse[case] - float(
                (self.y_sum - self.pred_sum[case]).square().sum() / self.n
            )
            for case in self.sse
        }
        return {
            'n': self.n,
            'target_variance_sum': variance,
            'one_minus_fvu': {k: 1 - v / variance for k, v in self.sse.items()},
            'mean_adjusted_one_minus_fvu': {
                k: 1 - v / variance for k, v in centered_sse.items()
            },
        }


def project(x, w, b=None):
    z = torch.einsum('btd,hdf->bthf', x, w)
    return z if b is None else z + b


def scores(q, k, scale):
    return torch.einsum('bqhd,bkhd->bhqk', q, k) / scale


def output(s, v, wo, bo, mask):
    p = s.masked_fill(~mask, torch.finfo(s.dtype).min).softmax(dim=-1)
    z = torch.einsum('bhqk,bkhd->bqhd', p, v)
    return torch.einsum('bqhd,hdf->bqf', z, wo) + bo


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    data = torch.load(ROOT / 'fresh_holdout.pt', map_location='cpu', weights_only=True)
    deploy = data['is_deployment']
    tokens = data['tokens'][deploy]
    markers = data['marker'][deploy]
    assert len(tokens) == 100
    model = load_sleeper_model('cuda')
    summary = {
        'dataset': 'fresh_holdout.pt deployment subset',
        'n_sequences': len(tokens),
        'scope': 'Story: marker query, causal key positions',
        'definition': '1 - SSE(term alone, exact target) / centered target variance',
        'mean_adjusted_definition': '1 - SSE(target mean + term - term mean, exact target) / centered target variance; no scale fitted',
        'qk_target': 'Exact pre-softmax QK scores; channel mean per head over all valid query-key pairs',
        'ov_target': 'Exact attention output at the marker; channel mean over deployment sequences',
        'qk_output_target': 'Exact attention output; each QK term is sole logits, with exact OV values',
        'layers': {},
    }
    for layer in range(4):
        wq, wk, wv, wo = [getattr(model, n)[layer].float()
                           for n in ('W_Q', 'W_K', 'W_V', 'W_O')]
        bq, bk, bv, bo = [getattr(model, n)[layer].float()
                           for n in ('b_Q', 'b_K', 'b_V', 'b_O')]
        hook = f'blocks.{layer}.ln1.hook_normalized'
        out_hook = f'blocks.{layer}.hook_attn_out'
        saes = []
        for seed in range(6):
            payload = torch.load(ROOT / 'weights' / f'sae_L{layer}_ln1_s{seed}.pt',
                                 map_location='cpu', weights_only=True)
            assert payload['config']['layer_hook'] == hook
            sae = TopKSAE(768, 1536, 32).cuda().eval()
            sae.load_state_dict(payload['state_dict'])
            saes.append(sae)
        stats = [{
            'qk_scores': Stats(QK_CASES),
            'ov_output': Stats(OV_CASES),
            'qk_term_output': Stats(QK_CASES),
        } for _ in range(6)]
        max_output_error = 0.0
        for start in range(0, len(tokens), 8):
            end = min(start + 8, len(tokens))
            _, cache = model.run_with_cache(
                tokens[start:end].cuda(),
                names_filter=lambda name: name in {hook, out_hook},
                return_type=None,
            )
            x, exact_output = cache[hook].float(), cache[out_hook].float()
            b, t, d = x.shape
            marker = markers[start:end].cuda()
            batch = torch.arange(b, device='cuda')
            key_keep = torch.arange(t, device='cuda')[None, :] <= marker[:, None]
            causal = torch.ones((t, t), device='cuda', dtype=torch.bool).tril()[None, None]
            scale = math.sqrt(model.cfg.d_head) if model.cfg.use_attn_scale else 1.0
            target_output = exact_output[batch, marker]
            for seed, sae in enumerate(saes):
                coherent, _ = sae(x.reshape(-1, d))
                coherent = coherent.reshape_as(x)
                error = x - coherent
                qc, kc, vc = (project(coherent, w, bias) for w, bias in
                              ((wq, bq), (wk, bk), (wv, bv)))
                qe, ke, ve = (project(error, w) for w in (wq, wk, wv))
                qk = {
                    'coherent': scores(qc, kc, scale),
                    'query_error': scores(qe, kc, scale),
                    'key_error': scores(qc, ke, scale),
                    'error_error': scores(qe, ke, scale),
                }
                full_scores = sum(qk.values())
                full_values = vc + ve
                complete_output = output(full_scores, full_values, wo, bo, causal)
                max_output_error = max(
                    max_output_error,
                    float((complete_output[batch, marker] - target_output).abs().max()),
                )
                target_scores = full_scores[batch, :, marker, :].permute(0, 2, 1)[key_keep]
                qk_alone = {
                    case: val[batch, :, marker, :].permute(0, 2, 1)[key_keep]
                    for case, val in qk.items()
                }
                stats[seed]['qk_scores'].add(target_scores, qk_alone)
                ov_alone = {
                    'coherent': output(full_scores, vc, wo, bo, causal)[batch, marker],
                    'error': output(full_scores, ve, wo, torch.zeros_like(bo), causal)[batch, marker],
                }
                stats[seed]['ov_output'].add(target_output, ov_alone)
                qk_term_output = {
                    case: output(val, full_values, wo, bo, causal)[batch, marker]
                    for case, val in qk.items()
                }
                stats[seed]['qk_term_output'].add(target_output, qk_term_output)
            print(f'layer={layer} batch={start}:{end}', flush=True)
        assert max_output_error < 3e-5, max_output_error
        summary['layers'][str(layer)] = {
            'identity_max_abs': max_output_error,
            'seeds': [{kind: st.result() for kind, st in seed_stats.items()}
                      for seed_stats in stats],
        }
        print(f'completed layer={layer} identity={max_output_error}', flush=True)
    OUT.write_text(json.dumps(summary, indent=2) + '\n')
    print(OUT, flush=True)


if __name__ == '__main__':
    main()

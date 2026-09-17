"""Evaluate the original FRA pair cuts without constructing an enormous 4-D tensor.

This uses the same decoder projections, reconstructed-residual RMS denominator,
RoPE, 1e-10 cutoff, and top-absolute-contribution selection as the archived semantic-filter code.
Only the selected feature pairs are evaluated over the full attention matrix.
"""
import math
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent/'reference'))
from fra_helpers import get_W_K, apply_rope_to_projected, _extract_rope_params


class SelectedFRA:
    def __init__(self, model, saes, heads, encode):
        self.model, self.saes, self.heads, self.encode = model, saes, heads, encode
        self.layers = sorted(set(l for l, h in heads))
        self.pairs = {}

    def prepare(self, tokens):
        names = [f'blocks.{l}.hook_resid_pre' for l in self.layers]
        _, cache = self.model.run_with_cache(tokens, names_filter=names)
        prepared = {}
        for l in self.layers:
            z = self.encode(l, cache[f'blocks.{l}.hook_resid_pre'][0])
            sae = self.saes[l]
            xhat = z @ sae.W_dec.float()+sae.b_dec.float()
            rms = (xhat.pow(2).mean(-1)+self.model.cfg.eps).sqrt()
            prepared[l] = z, rms
        return prepared

    def projections(self, l, h, qi, kj):
        dec = self.saes[l].W_dec.float()
        q = dec[qi] @ self.model.blocks[l].attn.W_Q[h].float()
        k = dec[kj] @ get_W_K(self.model, l, h).float()
        return q, k

    def rotate(self, vectors, l, pos):
        sin, cos, dim, adjacent = _extract_rope_params(self.model, l)
        return apply_rope_to_projected(vectors, pos, sin, cos, dim, adjacent) if sin is not None else vectors

    def locate(self, tokens, qpos, kpos, n_pairs=48):
        prepared = self.prepare(tokens)
        diagnostics = []
        for l, h in self.heads:
            z, rms = prepared[l]
            qi, kj = torch.where(z[qpos] != 0)[0], torch.where(z[kpos] != 0)[0]
            q, k = self.projections(l, h, qi, kj)
            q, k = self.rotate(q, l, qpos), self.rotate(k, l, kpos)
            values = (q@k.T)/math.sqrt(q.shape[-1])
            values = values*z[qpos, qi, None]*z[kpos, kj][None, :]
            values = values/(rms[qpos]*rms[kpos])
            mask = values.abs() > 1e-10
            a, b = torch.where(mask)
            v = values[mask].cpu().numpy()
            order = np.argsort(-np.abs(v))[:n_pairs]
            pairs = [(int(qi[a[i]]), int(kj[b[i]])) for i in order]
            self.pairs[(l, h)] = pairs
            diagnostics.append({'layer': l, 'head': h, 'pairs': pairs,
                                'locating_contributions': [float(v[i]) for i in order]})
        return diagnostics

    def deltas(self, tokens):
        prepared = self.prepare(tokens)
        by_layer = {}
        seq = tokens.shape[1]
        for l, h in self.heads:
            pairs = self.pairs[(l, h)]
            if not pairs:
                continue
            z, rms = prepared[l]
            qi, kj = zip(*pairs)
            qi, kj = list(qi), list(kj)
            q, k = self.projections(l, h, qi, kj)
            qp = torch.stack([self.rotate(q, l, pos) for pos in range(seq)])
            kp = torch.stack([self.rotate(k, l, pos) for pos in range(seq)])
            values = torch.einsum('qmd,kmd->qkm', qp, kp)/math.sqrt(q.shape[-1])
            values = values*z[:, qi][:, None, :]*z[:, kj][None, :, :]
            values = values/(rms[:, None, None]*rms[None, :, None])
            values = torch.where(values.abs() > 1e-10, values, 0)
            delta = values.double().sum(-1).tril().float()
            by_layer.setdefault(l, {})[h] = delta
        return by_layer

    def run(self, tokens, deltas, strength):
        if strength == 0:
            return self.model(tokens)
        hooks = []
        for l, hd in deltas.items():
            scaled = {h: dd*strength for h, dd in hd.items()}
            def make(scaled):
                def hook(scores, hook):
                    for h, dd in scaled.items():
                        scores[0, h, :dd.shape[0], :dd.shape[1]] -= dd.to(scores.dtype)
                    return scores
                return hook
            hooks.append((f'blocks.{l}.attn.hook_attn_scores', make(scaled)))
        return self.model.run_with_hooks(tokens, fwd_hooks=hooks)

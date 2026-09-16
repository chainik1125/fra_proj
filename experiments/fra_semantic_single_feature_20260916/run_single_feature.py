"""Diff-rank ten SAE features, steer each separately, compare on frozen FRA queries.

Only the planted-word calibration prompts are used for feature selection. The
best feature/strength is reported retrospectively, as requested; this is an
oracle comparison over the ten candidates, not a validated deployment policy.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
from sae_lens_wrapper import GemmaScopeSAE

REFERENCE_COMMIT = "1363ce518cc10fc97f0ec1ceff98a748766d365a"
SITE = "blocks.6.hook_resid_pre"
LAYER = 6
TOP_N = 10
MULT_GRID = [-32, -16, -8, -4, -2, -1, -.5, -.25, 0, .125, .25, .5,
             .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
ADD_GRID = [-128, -64, -32, -16, -8, -4, -2, -1, -.5, -.25, -.125,
            0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128]
LEGACY_GRID = [1, 2, 4, 8, 16, 32]


def reference_constants():
    tree = ast.parse((ROOT / "reference/semantic_filter_original.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"FILLER", "CONCEPTS"}:
                    out[target.id] = ast.literal_eval(node.value)
    return out["FILLER"], out["CONCEPTS"]


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


@torch.inference_mode()
def run_concept(concept, out_dir, smoke=False, checkpoint_callback=None):
    started = time.time()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{concept}{'_smoke' if smoke else ''}.json"
    if out_path.exists():
        previous = json.loads(out_path.read_text())
        if previous.get("done"):
            print("REUSE COMPLETE", concept, flush=True)
            return previous

    filler, concepts = reference_constants()
    _, plant, payload, probes, legitimate = next(c for c in concepts if c[0] == concept)
    reference = json.loads((ROOT / "reference/rung3_broad6.json").read_text())["rows"]
    reference = [r for r in reference if r["concept"] == concept]
    model = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda", dtype=torch.float16)
    model.eval()
    tok = model.tokenizer
    sae = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical", "layer_5/width_65k/canonical",
                       device="cuda", normalize_activations=True)
    pid = tok.encode(payload, add_special_tokens=False)[0]

    def ids(seed, word, planted=True, n_before=4, n_mid=6):
        fs = [filler[i] for i in np.random.default_rng(seed).permutation(len(filler))]
        sentence = f" The password is{plant}{payload}." if planted else f" The password is{plant}."
        text = " ".join(fs[:n_before]) + sentence + " " + " ".join(fs[n_before:n_before+n_mid])
        text += " Remember the password:" + word
        return [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)

    def encode(x):
        z = sae.encode(x.float()).float()
        if sae._norm_coeff is not None:
            z = z / sae._norm_coeff
        return z

    def residual_at_last(token_ids):
        tt = torch.tensor([token_ids], device="cuda")
        _, cache = model.run_with_cache(tt, names_filter=[SITE], stop_at_layer=LAYER+1)
        return cache[SITE][0, -1].float()

    pools = {"on": [], "no_payload": [], "unrelated": []}
    for s in range(12):
        pools["on"].append(encode(residual_at_last(ids(1000+s, plant))[None])[0])
        pools["no_payload"].append(encode(residual_at_last(ids(1000+s, plant, False))[None])[0])
        fs = [filler[i] for i in np.random.default_rng(5000+s).permutation(len(filler))]
        unrelated_ids = [tok.bos_token_id] + tok.encode(" ".join(fs[:12]), add_special_tokens=False)
        pools["unrelated"].append(encode(residual_at_last(unrelated_ids)[None])[0])
    means = {k: torch.stack(v).mean(0) for k, v in pools.items()}
    rankings = {}
    for contrast in ("no_payload", "unrelated"):
        diff = means["on"] - means[contrast]
        ranked = torch.topk(diff, 12).indices.tolist()
        rankings[contrast] = [{"rank": i+1, "feature": f, "diff": float(diff[f]),
                               "on_mean": float(means['on'][f]), "off_mean": float(means[contrast][f])}
                              for i, f in enumerate(ranked)]
    features = list(dict.fromkeys(r["feature"] for rs in rankings.values() for r in rs[:TOP_N]))
    if smoke:
        features = features[:2]
    mult_grid = [0, .5, 1, 2, 8] if smoke else MULT_GRID
    add_grid = [-8, 0, 8] if smoke else ADD_GRID
    configurations = []
    for f in features:
        for mode, grid in [("activation", mult_grid), ("additive", add_grid)]:
            for strength in grid:
                configurations.append({"feature": f, "mode": mode, "strength": strength})
    for contrast in rankings:
        for strength in LEGACY_GRID:
            configurations.append({"feature": None, "mode": "legacy12", "contrast": contrast,
                                   "strength": strength})
    all_features = list(dict.fromkeys(features + [r["feature"] for rs in rankings.values() for r in rs]))
    feature_col = {f: j for j, f in enumerate(all_features)}
    decoder = sae.W_dec[all_features].float().clone()
    decoder_unit = decoder / decoder.norm(dim=-1, keepdim=True)

    result = {"done": False, "meta": {
        "concept": concept, "plant": plant.strip(), "payload": payload.strip(),
        "reference_commit": REFERENCE_COMMIT, "model": "gemma-2-2b", "dtype": "float16",
        "site": SITE, "sae_id": "layer_5/width_65k/canonical", "top_n": TOP_N,
        "calibration_seeds": list(range(1000, 1012)), "selection_position": "last query token",
        "selection": "mean SAE activation(planted) minus mean SAE activation(control), descending signed diff",
        "activation_intervention": "x -= strength * z_f(x) * W_dec[f], every position",
        "additive_intervention": "x -= strength * W_dec[f]/norm(W_dec[f]), every position",
        "mult_grid": mult_grid, "add_grid": add_grid,
        "best_selection": "post-hoc best single candidate/strength at each suppression threshold",
        "smoke": smoke, "gpu": torch.cuda.get_device_name(),
        "versions": {n: metadata.version(n) for n in ["torch", "transformer-lens", "sae-lens", "transformers", "numpy"]},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "rankings": rankings, "configurations": configurations,
        "collateral": [], "rows": [], "validation": {}}

    def checkpoint():
        result["meta"]["runtime_s"] = time.time()-started
        atomic_json(out_path, result)
        if checkpoint_callback:
            checkpoint_callback()

    def prepare(token_ids):
        tt = torch.tensor([token_ids], device="cuda")
        logits, cache = model.run_with_cache(tt, names_filter=[SITE])
        resid = cache[SITE].clone()
        z = encode(resid[0])[:, all_features].clone()
        return tt, resid, z, logits

    def intervention_batch(resid, z, cfgs):
        deltas = []
        for cfg in cfgs:
            if cfg["mode"] == "legacy12":
                cols = [feature_col[r['feature']] for r in rankings[cfg['contrast']]]
                delta = (z[:, cols] * cfg['strength']) @ decoder[cols]
            else:
                j = feature_col[cfg['feature']]
                if cfg['mode'] == 'activation':
                    delta = (cfg['strength'] * z[:, j:j+1]) @ decoder[j:j+1]
                else:
                    delta = (cfg['strength'] * decoder_unit[j]).expand(resid.shape[1], -1)
            deltas.append(delta)
        # Cast the subtraction before doing fp16 arithmetic, matching original conv_run.
        edited = resid.expand(len(cfgs), -1, -1).float() - torch.stack(deltas)
        return edited.to(resid.dtype)

    def forward_edited(resid, z, cfgs, last_only):
        edited = intervention_batch(resid, z, cfgs)
        hooks = []
        if last_only:
            hooks = [("ln_final.hook_normalized", lambda x, hook: x[:, -1:, :])]
        return model.run_with_hooks(edited, start_at_layer=LAYER, fwd_hooks=hooks)

    # Verify cached-prefix continuation and last-position output against direct full forwards.
    tt, resid, z, clean = prepare(ids(0, plant))
    check_cfgs = [{"feature": features[0], "mode": "activation", "strength": 0},
                  {"feature": features[0], "mode": "activation", "strength": 2},
                  {"feature": features[0], "mode": "additive", "strength": 8}]
    checks = []
    for cfg in check_cfgs:
        def direct_hook(act, hook):
            f = cfg['feature']
            if cfg['mode'] == 'activation':
                live_z = encode(act[0])[:, f:f+1]
                delta = (cfg['strength'] * live_z) @ sae.W_dec[f:f+1].float()
            else:
                direction = sae.W_dec[f].float()
                delta = cfg['strength'] * direction / direction.norm()
            return (act.float() - delta).to(act.dtype)
        full = model.run_with_hooks(tt, fwd_hooks=[(SITE, direct_hook)])[:, -1, :].float()
        cached = forward_edited(resid, z, [cfg], True)[:, -1, :].float()
        p_full = full.softmax(-1)[0, pid].item()
        p_cached = cached.softmax(-1)[0, pid].item()
        err = float((full-cached).abs().max())
        checks.append({**cfg, "max_logit_error": err, "p_full": p_full, "p_cached": p_cached})
        if err > 0.1 or abs(p_full-p_cached) > 0.003:
            raise RuntimeError(f"Cached execution validation failed: {checks[-1]}")
    result['validation']['cached_vs_direct'] = checks
    print('VALIDATED', concept, checks, flush=True)
    del clean, resid, z
    torch.cuda.empty_cache()

    # One original legitimate paragraph per concept, identical summed KL metric.
    hids = [tok.bos_token_id] + tok.encode(legitimate, add_special_tokens=False)
    _, resid, z, clean = prepare(hids)
    clean_lp = clean.float().log_softmax(-1)
    clean_p = clean_lp.exp()
    batch_size = int(os.environ.get('STEER_BATCH_SIZE', '12'))
    for i in range(0, len(configurations), batch_size):
        cfgs = configurations[i:i+batch_size]
        edited = forward_edited(resid, z, cfgs, False)
        edited_lp = edited.float().log_softmax(-1)
        kl = (clean_p * (clean_lp-edited_lp)).sum(-1).sum(-1)
        vals = kl.cpu().tolist()
        for cfg, val in zip(cfgs, vals):
            result['collateral'].append(0.0 if cfg['strength'] == 0 else max(0.0, val))
        del edited, edited_lp, kl
    del clean, clean_lp, clean_p, resid, z
    checkpoint()
    print('COLLATERAL DONE', concept, len(configurations), 'configs', flush=True)

    for row_index, ref in enumerate(reference):
        if smoke and row_index > 1:
            break
        seed, nb, nm = [int(v) for v in ref['ctx'].split(':')]
        token_ids = ids(seed, ' '+ref['word'], n_before=nb, n_mid=nm)
        _, resid, z, clean = prepare(token_ids)
        base = clean[0, -1].float().softmax(-1)[pid].item()
        row = {k: ref[k] for k in ['concept', 'word', 'kind', 'ctx']}
        row.update(base=base, reference_base=ref['base'], probabilities=[], suppression=[])
        if abs(base-ref['base']) > 0.008:
            raise RuntimeError(f"Reference base drift: {row}")
        del clean
        for i in range(0, len(configurations), batch_size):
            cfgs = configurations[i:i+batch_size]
            edited = forward_edited(resid, z, cfgs, True)
            probabilities = edited[:, -1].float().softmax(-1)[:, pid].cpu().tolist()
            for cfg, p in zip(cfgs, probabilities):
                if cfg['strength'] == 0:
                    p = base
                row['probabilities'].append(p)
                row['suppression'].append(1-p/base)
            del edited
        result['rows'].append(row)
        legacy_checks = []
        for i, cfg in enumerate(configurations):
            if cfg['mode'] == 'legacy12':
                method = 'conv' if cfg['contrast'] == 'no_payload' else 'conv_u'
                old_sup, old_kl = ref[method][LEGACY_GRID.index(cfg['strength'])]
                legacy_checks.append({'contrast': cfg['contrast'], 'strength': cfg['strength'],
                    'suppression_error': row['suppression'][i]-old_sup,
                    'kl_error': result['collateral'][i]-old_kl,
                    'reference_kl': old_kl})
        row['legacy_validation'] = legacy_checks
        print('QUERY DONE', concept, ref['ctx'], ref['word'], f"{row_index+1}/{len(reference)}",
              'base_error', round(base-ref['base'], 6), 'elapsed', round(time.time()-started), flush=True)
        del resid, z
        checkpoint()

    result['validation']['max_reference_base_error'] = max(abs(r['base']-r['reference_base']) for r in result['rows'])
    result['validation']['max_legacy_suppression_error'] = max(abs(p['suppression_error']) for r in result['rows'] for p in r['legacy_validation'])
    result['validation']['max_legacy_kl_relative_error'] = max(abs(p['kl_error'])/max(p['reference_kl'], 1e-3) for r in result['rows'] for p in r['legacy_validation'])
    result['done'] = True
    checkpoint()
    print('COMPLETE', concept, result['validation'], flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept', required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_concept(args.concept, args.out_dir, args.smoke)

"""Unsteered GPT-2 rollout + offline SAE readout; no semantic edit is implemented.

Run with uv run --no-project --with sae-lens==6.46.1 python trace_baseline.py.
Reconstruction-only audit forwards are separate from the saved baseline rollout.
"""
import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HOME", "/private/tmp/fra_concept_trace_hf")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parent
TARGETS = [" king", " queen", " man", " woman"]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    temporary = path.with_name(path.name + ".tmp")
    with opener(temporary, "wt") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, default=str)
    temporary.replace(path)


def examples():
    rows = []
    for word in ["king", "queen", "man", "woman"]:
        rows.append(dict(id=word, group="matched_rollout", source=word,
                         prompt=f"The {word} entered the room. The person who entered was the",
                         new_tokens=8))
    for sex in ["male", "female"]:
        rows.append(dict(id=f"definition_{sex}", group="composition_rollout", source=sex,
                         prompt=f"A {sex} monarch is called a", new_tokens=4))
    for word in ["father", "mother", "brother", "sister", "uncle", "aunt", "boy", "girl"]:
        rows.append(dict(id=f"contrast_{word}", group="independent_contrast", source=word,
                         prompt=f"The {word} entered the room. The person who entered was the",
                         new_tokens=0))
    neutral = [
        "The train arrived at the station just before the rain began.",
        "She put the book on the table and opened the window.",
        "The report describes the results of a scientific experiment.",
        "Several people walked through the park on a sunny afternoon.",
        "After the meeting, they discussed their plans for the next week.",
        "The old building stood beside a river in the center of town.",
    ]
    rows += [dict(id=f"neutral_{i}", group="neutral", source=None,
                  prompt=p, new_tokens=0) for i, p in enumerate(neutral)]
    return rows


def distributions(logits, model, target_ids):
    logp = logits.float().log_softmax(-1)
    vals, ids = logp.exp().topk(5, dim=-1)
    return [dict(top=[dict(token_id=int(t), token=model.tokenizer.decode([int(t)]),
                           probability=float(v)) for t, v in zip(ii, vv)],
                 candidates={word: float(lp[idx].exp()) for word, idx in target_ids.items()})
            for lp, ii, vv in zip(logp, ids, vals)]


def checkpoint_manifest(repo_id):
    cache = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo_id.replace("/", "--"))
    files = []
    for p in sorted((cache / "snapshots").glob("**/*")):
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(chunk)
        files.append(dict(path=str(p.relative_to(cache)), bytes=p.stat().st_size,
                          sha256=h.hexdigest()))
    return files


def drop_temporary_weights(repo_id):
    """Only discard downloaded SAE weights in this experiment's dedicated cache."""
    root = Path(os.environ["HF_HOME"]).resolve()
    if root != Path("/private/tmp/fra_concept_trace_hf"):
        return
    cache = root / "hub" / ("models--" + repo_id.replace("/", "--"))
    for p in (cache / "snapshots").glob("**/*"):
        if p.is_file() and p.suffix in {".pt", ".safetensors"}:
            real = p.resolve()
            p.unlink()
            if real != p and real.is_relative_to(cache) and real.exists():
                real.unlink()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "teacher_forced")
    parser.add_argument("--mode", choices=["teacher_forced", "greedy"], default="teacher_forced")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--compare-128k", type=int, nargs="*", default=[5, 8])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    t0 = time.time()
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    targets = {w: model.to_tokens(w, prepend_bos=False).flatten().tolist() for w in TARGETS}
    assert all(len(t) == 1 for t in targets.values()), targets
    target_ids = {w: t[0] for w, t in targets.items()}
    hooks = [f"blocks.{l}.hook_resid_post" for l in args.layers]
    rows = examples()
    states = {}
    for row in rows:
        tokens = model.to_tokens(row["prompt"], prepend_bos=False)
        prompt_length = tokens.shape[1]
        online = []
        if args.mode == "teacher_forced" and row["new_tokens"]:
            answer = row["source"] if row["group"] == "matched_rollout" else (
                "king" if row["source"] == "male" else "queen")
            online.append(model(tokens)[0, -1].clone())
            full = model.to_tokens(row["prompt"] + " " + answer + ".", prepend_bos=False)
            assert torch.equal(tokens, full[:, :prompt_length])
            tokens = full
        for _ in range(row["new_tokens"] if args.mode == "greedy" else 0):
            logits = model(tokens)
            online.append(logits[0, -1].clone())
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            if int(next_token) == model.tokenizer.eos_token_id:
                break
        logits, cache = model.run_with_cache(tokens, names_filter=hooks)
        lp = logits[0].float().log_softmax(-1)
        online_delta = max((float((v - logits[0, prompt_length - 1 + j]).abs().max())
                            for j, v in enumerate(online)), default=0.0)
        assert online_delta < 2e-4, online_delta
        pieces = [model.tokenizer.decode([int(t)]) for t in tokens[0]]
        row.update(token_ids=tokens[0].tolist(), tokens=pieces,
                   prompt_length=prompt_length, text=model.tokenizer.decode(tokens[0]),
                   continuation=model.tokenizer.decode(tokens[0, prompt_length:]),
                   continuation_mode=args.mode,
                   next_token=distributions(logits[0], model, target_ids),
                   rollout_replay_max_logit_difference=online_delta)
        offset = 0
        row["token_spans"] = []
        for piece in pieces:
            row["token_spans"].append([offset, offset + len(piece)])
            offset += len(piece)
        assert "".join(pieces) == row["text"]  # ASCII examples; preserve exact token offsets.
        states[row["id"]] = dict(tokens=tokens, logp=lp, cache=cache)
        print("BASELINE", row["id"], repr(row["continuation"]), flush=True)
    manifest = dict(model="gpt2-small", prepend_bos=False, decoding=args.mode, seed=0,
                    packages={x: importlib.metadata.version(x) for x in
                              ["sae-lens", "transformer-lens", "torch", "transformers"]},
                    model_config=model.cfg.to_dict(),
                    model_files=checkpoint_manifest("gpt2"),
                    baseline_is_unsteered=True,
                    reconstruction_audits_are_separate=True)
    baseline_path = args.out / "baseline.json"
    if baseline_path.exists():
        with baseline_path.open() as f:
            previous = json.load(f)
        signature = lambda es: [(e["id"], e["token_ids"], e["prompt_length"]) for e in es]
        if (previous["manifest"]["decoding"] != args.mode or
                signature(previous["examples"]) != signature(rows)):
            raise ValueError("Existing output uses different sequences/mode; use a fresh --out directory.")
    save_json(baseline_path, dict(manifest=manifest, examples=rows))
    # Raw hook tensors are saved before any SAE is loaded into the model process.
    np.savez_compressed(args.out / "residuals.npz", **{
        f"{rid}__{h}": s["cache"][h][0].numpy() for rid, s in states.items() for h in hooks})
    directory = get_pretrained_saes_directory()
    specs = [("gpt2-small-resid-post-v5-32k", l) for l in args.layers]
    specs += [("gpt2-small-resid-post-v5-128k", l) for l in args.compare_128k]
    summaries = []
    for release, layer in specs:
        hook = f"blocks.{layer}.hook_resid_post"
        if hook not in hooks:
            continue
        key = f"{release}__L{layer}"
        path = args.out / "features" / f"{key}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as f:
                summaries.append(json.load(f)["summary"])
            continue
        lookup = directory[release]
        print("LOADING", key, flush=True)
        sae = SAE.from_pretrained(release, hook, device="cpu")
        sae.eval()
        cfg = sae.cfg.to_dict()
        metadata = cfg.get("metadata", cfg)
        assert metadata["hook_name"] == hook
        assert metadata.get("prepend_bos") is False, metadata
        records, quality = {}, []
        for row in rows:
            state = states[row["id"]]
            x = state["cache"][hook]
            # Encode/decode consecutively: layer_norm SAEs retain the input scale.
            z = sae.encode(x)
            xhat = sae.decode(z)
            assert torch.isfinite(z).all() and torch.isfinite(xhat).all()
            # Verify the installed SAE's forward path agrees with offline decoding.
            assert torch.allclose(xhat, sae(x), atol=1e-5, rtol=1e-5)
            err = (x - xhat).square().sum(-1)[0]
            energy = x.square().sum(-1)[0]
            active = (z[0] != 0).sum(-1)
            feats = []
            for vector in z[0]:
                ids = torch.nonzero(vector, as_tuple=True)[0]
                ids = ids[torch.argsort(vector[ids], descending=True)]
                feats.append([[int(i), float(vector[i])] for i in ids])
            records[row["id"]] = dict(features=feats, active_count=active.tolist(),
                relative_squared_error=(err / energy.clamp_min(1e-12)).tolist())
            # This audit substitutes only the SAE reconstruction, never a concept edit.
            def replace(_activation, hook):
                return xhat
            audit_logits = model.run_with_hooks(state["tokens"], fwd_hooks=[(hook, replace)])
            audit_lp = audit_logits[0].float().log_softmax(-1)
            clean_lp = state["logp"]
            kl = (clean_lp.exp() * (clean_lp - audit_lp)).sum(-1)
            actual_next = state["tokens"][0, 1:]
            clean_ce = -clean_lp[:-1].gather(1, actual_next[:, None]).squeeze(1)
            audit_ce = -audit_lp[:-1].gather(1, actual_next[:, None]).squeeze(1)
            p = row["prompt_length"] - 1
            quality.append(dict(example=row["id"], group=row["group"],
                squared_error=float(err.sum()), squared_norm=float(energy.sum()),
                tokens=int(x.shape[1]), mean_l0=float(active.float().mean()),
                mean_kl=float(kl.mean()), max_token_kl=float(kl.max()),
                mean_ce_clean=float(clean_ce.mean()), mean_ce_reconstructed=float(audit_ce.mean()),
                answer_position_kl=float(kl[p]),
                answer_reconstructed={w: float(audit_lp[p, i].exp()) for w, i in target_ids.items()}))
        all_x = torch.cat([s["cache"][hook][0] for s in states.values()])
        noninitial_x = torch.cat([s["cache"][hook][0, 1:] for s in states.values()])
        token_errors = [v for r in records.values() for v in r["relative_squared_error"][1:]]
        noninitial_error = sum(float((s["cache"][hook][0, 1:].square().sum(-1) *
            torch.tensor(records[rid]["relative_squared_error"][1:])).sum())
            for rid,s in states.items())
        centered_ss = float((all_x - all_x.mean(0)).square().sum())
        total_ss = sum(q["squared_norm"] for q in quality)
        total_error = sum(q["squared_error"] for q in quality)
        nt = sum(q["tokens"] for q in quality)
        summary = dict(key=key, release=release, layer=layer, hook=hook,
            d_sae=sae.cfg.d_sae, neuronpedia=lookup.neuronpedia_id.get(hook),
            fvu_centered=total_error/centered_ss, relative_squared_error=total_error/total_ss,
            fvu_excluding_first_token=noninitial_error / float(
                (noninitial_x-noninitial_x.mean(0)).square().sum()),
            mean_token_relative_error_excluding_first=float(np.mean(token_errors)),
            mean_l0=sum(q["mean_l0"]*q["tokens"] for q in quality)/nt,
            mean_kl=sum(q["mean_kl"]*q["tokens"] for q in quality)/nt,
            registry_variance_explained=lookup.expected_var_explained.get(hook),
            registry_l0=lookup.expected_l0.get(hook))
        payload = dict(summary=summary, cfg=cfg, quality=quality, examples=records,
                       files=checkpoint_manifest(lookup.repo_id))
        save_json(path, payload)
        summaries.append(summary)
        save_json(args.out / "sae_quality.json", summaries)
        print("MEASURED", json.dumps(summary), flush=True)
        del sae, z, xhat, all_x, noninitial_x
        gc.collect()
        drop_temporary_weights(lookup.repo_id)
    save_json(args.out / "sae_quality.json", summaries)
    print("DONE", round(time.time()-t0, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()

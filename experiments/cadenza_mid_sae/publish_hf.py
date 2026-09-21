#!/usr/bin/env python3
"""Copy final SAE exports from Simplex directly to the existing HF collection.

Local side is stdlib-only and never transfers weights through the laptop.
The HF credential goes through SSH stdin into remote process memory only.
No repo creation, visibility changes, overwrites, deletions or credential saves.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

REPO = "dmanningcoe/fra-phase1-steering-data"
PREFIX = "cadenza_attn_only/variantA/saes/ln1_topk_8x_100M_20260921"
REMOTE_ROOT = Path("/data/users/dmitry/sae-middle")
CAMPAIGN = "A-input4-100M-20260921"
LAYERS = (0, 8, 16, 24)
RECEIPT = REMOTE_ROOT / "hf_upload_A_input4_100M_20260921.json"
ALL_CHECKPOINTS = False


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def strip_remote_paths(value):
    if isinstance(value, dict):
        return {k: strip_remote_paths(v) for k, v in value.items() if k not in ("path", "sae_path")}
    if isinstance(value, list):
        return [strip_remote_paths(v) for v in value]
    return value


def small_json(path):
    if not path.is_file() or path.stat().st_size > 100_000:
        raise ValueError(f"Invalid or oversized metadata: {path.name}")
    return json.loads(path.read_text())


def validate_source(run, layer):
    final = (run / "sae_final").resolve(strict=True)
    if final != run / "checkpoints" / "tokens_100000000":
        raise ValueError("Final SAE does not resolve to the exact 100M checkpoint")
    summary = small_json(run / "summary.json")
    cfg = small_json(final / "cfg.json")
    expected_hook = f"blocks.{layer}.ln1.hook_normalized"
    if (summary.get("state") != "complete" or summary.get("tokens_trained") != 100_000_000
            or not summary.get("checkpoint_reload", {}).get("passed")
            or summary.get("hook") != expected_hook
            or cfg.get("metadata", {}).get("hook_name") != expected_hook
            or cfg.get("metadata", {}).get("training_tokens") != 100_000_000
            or cfg.get("metadata", {}).get("model_name") != "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
            or (cfg.get("architecture"), cfg.get("d_in"), cfg.get("d_sae"), cfg.get("k")) != ("topk", 4096, 32768, 50)):
        raise ValueError(f"L{layer}: source identity/completion/reload check failed")
    weight = final / "sae_weights.safetensors"
    if weight.is_symlink() or not weight.is_file() or not 1_000_000_000 < weight.stat().st_size < 1_100_000_000:
        raise ValueError("Unexpected final SAE weights")
    return final, summary


def readme(records):
    lines = ["# Llama-3 8B attention-only sleeper A: attention-input TopK SAEs", "",
             "Final 100M-activation-token SAE exports for zero-based layers **0, 8, 16, 24**.",
             "These are copies; the original runs and all ten checkpoints/layer remain on Simplex.",
             "Only the final checkpoint per layer is included here. No optimizer states, raw activations, or language-model weights are uploaded.",
             "", "## Model, hook and training", "",
             "- Model: [dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A](https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A), revision `027f599bb4c24e4bac72932ce557f9fa325aa9be`.",
             "- Actual base is Llama 3 8B, not Llama 3.1. Variant A is the Q/K/V/O-only finetune.",
             "- Hook: `blocks.L.ln1.hook_normalized`, RMS-normalized attention input **before learned RMSNorm gain**. Native HF layernorm output already includes that gain; account for it when harvesting, replacing, or projecting features.",
             "- SAE Lens 6.44.0 TopK format, input 4096, width 32768 (8x), target k=50. FP32 exported weights; input scaling and decoder-norm-weighted TopK folded into ordinary TopK weights.",
             "- Train/evaluate on 50/50 Cadenza sleeper/non-sleeper **examples**, not equal token counts; longer sleeper completions contribute more tokens.",
             "- Dataset: `Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled`, revision `502f516971a492a9bffae3bda179b43dd808acd2`.",
             "- The finite training set is repeatedly sampled. Official-test question identities, including trigger-stripped twins, are excluded from SAE training. This is not a claim that the sleeper LM never saw Cadenza.",
             "- LR 8e-4, Adam (0.9,0.9999), batch 2048, context 1024, norm-based reconstruction loss, no auxiliary dead-feature loss; warmup and k annealing scaled to 100M tokens. See each layer's `manifest.json` and `training_config.json`.",
             "", "## Quality caveats", "",
             "All four completed training and passed checkpoint reload-equivalence checks, but **all failed the predeclared pilot quality gate**. They are exploratory artifacts, not quality-certified SAEs.",
             "", "| Layer | Sleeper FVU | Clean FVU | Clean CE increase (nats/token) | Dead features over final 10M tokens |",
             "|---|---:|---:|---:|---:|"]
    for record in records:
        s = record["summary"]
        lines.append(f"| {record['layer']} | {s['final_reconstruction']['sleeper']['fvu']:.4f} | "
                     f"{s['final_reconstruction']['non_sleeper']['fvu']:.4f} | "
                     f"{s['teacher_forced_ce']['non_sleeper']['ce_increase']:.4f} | "
                     f"{100 * s['dead_fraction_last_checkpoint_window']:.1f}% |")
    lines += ["", "FVU is unexplained variance (lower is better); CE is teacher-forced next-token loss after SAE replacement.",
              "The dead-feature gate was 20%. Layer 0 also fell below the target active-feature count; layers 8/16 also missed the 80% clean ablation-loss-recovery requirement. See `quality_gate.json` for exact failures.",
              "No steering results or claims of successful backdoor removal are included in this upload.",
              "", "## Load on a remote GPU machine", "", "Do not run these downloads on a laptop with a 10MB file-size limit.",
              "", "```python", "from pathlib import Path", "from huggingface_hub import snapshot_download", "from sae_lens import SAE",
              f"prefix = {PREFIX!r}", f"root = snapshot_download({REPO!r}, repo_type='dataset',",
              "                         allow_patterns=[f'{prefix}/layer_16/*'])",
              "sae = SAE.load_from_disk(str(Path(root) / prefix / 'layer_16'), device='cuda')", "```", "",
              "Each layer contains `cfg.json`, `sae_weights.safetensors`, training/data manifests, checkpoint metadata, quality-gate results and evaluation summary. The top-level `upload_manifest.json` records file sizes and SHA256 digests.",
              "The repository's existing content and visibility were not changed.", ""]
    return "\n".join(lines).encode()


def build_operations():
    if ALL_CHECKPOINTS:
        return build_checkpoint_operations()
    from huggingface_hub import CommitOperationAdd
    campaign = small_json(REMOTE_ROOT / "campaigns" / CAMPAIGN / "status.json")
    operations, records, files = [], [], []
    def add(relative, payload):
        op = CommitOperationAdd(path_in_repo=f"{PREFIX}/{relative}", path_or_fileobj=payload)
        operations.append(op)
        record = {"path": op.path_in_repo, "bytes": op.upload_info.size, "sha256": op.upload_info.sha256.hex()}
        if isinstance(payload, bytes):
            record["git_blob_sha1"] = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        files.append(record)
    for layer in LAYERS:
        run_id = f"{CAMPAIGN}-L{layer:02d}-train-a1"
        run = REMOTE_ROOT / "runs" / run_id
        final, summary = validate_source(run, layer)
        job = next(j for j in campaign["jobs"] if j["kind"] == "train" and j["layer"] == layer)
        if job["run_id"] != run_id or job["state"] != "complete":
            raise ValueError("Campaign provenance does not match source SAE")
        prefix = f"layer_{layer:02d}"
        add(f"{prefix}/sae_weights.safetensors", final / "sae_weights.safetensors")
        add(f"{prefix}/cfg.json", (final / "cfg.json").read_bytes())
        for src, dest in (("manifest.json", "manifest.json"), ("config.json", "training_config.json"),
                          ("data_summary.json", "data_summary.json"), ("source_manifest.json", "source_manifest.json")):
            add(f"{prefix}/{dest}", json_bytes(strip_remote_paths(small_json(run / src))))
        add(f"{prefix}/training_summary.json", json_bytes(strip_remote_paths(summary)))
        add(f"{prefix}/checkpoint.json", json_bytes(strip_remote_paths(small_json(final / "checkpoint.json"))))
        add(f"{prefix}/quality_gate.json", json_bytes(job["quality"]))
        records.append({"layer": layer, "source_run_id": run_id, "summary": summary})
        print(json.dumps({"event": "source_verified_and_hashed", "layer": layer}), flush=True)
    add("README.md", readme(records))
    manifest = {"repo_id": REPO, "repo_type": "dataset", "prefix": PREFIX,
                "campaign": CAMPAIGN, "source_host": "simplex1", "scope": "final 100M exports only",
                "layers": list(LAYERS), "originals_retained": True, "local_weight_downloads": False,
                "files": list(files)}
    add("upload_manifest.json", json_bytes(manifest))
    return operations, files


def validate_checkpoint(run, layer, tokens):
    folder = run / "checkpoints" / f"tokens_{tokens:09d}"
    if folder.resolve(strict=True) != folder:
        raise ValueError("Checkpoint must be an exact non-symlinked source directory")
    cfg, checkpoint = small_json(folder / "cfg.json"), small_json(folder / "checkpoint.json")
    meta = cfg.get("metadata", {})
    if (tokens not in range(10_000_000, 100_000_001, 10_000_000)
            or checkpoint.get("tokens") != tokens or not checkpoint.get("reload_check", {}).get("passed")
            or meta.get("training_tokens") != tokens or meta.get("hook_name") != f"blocks.{layer}.ln1.hook_normalized"
            or meta.get("model_name") != "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
            or meta.get("model_revision") != "027f599bb4c24e4bac72932ce557f9fa325aa9be"
            or (cfg.get("architecture"), cfg.get("d_in"), cfg.get("d_sae"), cfg.get("k")) != ("topk", 4096, 32768, 50)):
        raise ValueError("Checkpoint identity, token count or reload verification failed")
    weight = folder / "sae_weights.safetensors"
    if weight.is_symlink() or not weight.is_file() or not 1_000_000_000 < weight.stat().st_size < 1_100_000_000:
        raise ValueError("Unexpected checkpoint weight file")
    return folder, cfg, checkpoint


def build_checkpoint_operations():
    from huggingface_hub import CommitOperationAdd
    campaign = small_json(REMOTE_ROOT / "campaigns" / CAMPAIGN / "status.json")
    operations, files = [], []
    def add(relative, payload):
        op = CommitOperationAdd(path_in_repo=f"{PREFIX}/{relative}", path_or_fileobj=payload)
        operations.append(op)
        record = {"path": op.path_in_repo, "bytes": op.upload_info.size, "sha256": op.upload_info.sha256.hex()}
        if isinstance(payload, bytes):
            record["git_blob_sha1"] = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        files.append(record)
    for layer in LAYERS:
        run_id = f"{CAMPAIGN}-L{layer:02d}-train-a1"
        run = REMOTE_ROOT / "runs" / run_id
        _, summary = validate_source(run, layer)
        job = next(j for j in campaign["jobs"] if j["kind"] == "train" and j["layer"] == layer)
        if job["run_id"] != run_id or job["state"] != "complete":
            raise ValueError("Campaign provenance does not match checkpoint source")
        retained = summary.get("retained_checkpoints", [])
        expected = list(range(10_000_000, 100_000_001, 10_000_000))
        if ([c.get("tokens") for c in retained] != expected
                or not all(c.get("reload_check", {}).get("passed") for c in retained)):
            raise ValueError("Expected all ten reload-verified checkpoints")
        for tokens in expected:
            folder, cfg, checkpoint = validate_checkpoint(run, layer, tokens)
            prefix = f"layer_{layer:02d}/tokens_{tokens:09d}"
            add(f"{prefix}/sae_weights.safetensors", folder / "sae_weights.safetensors")
            add(f"{prefix}/cfg.json", json_bytes(cfg))
            add(f"{prefix}/checkpoint.json", json_bytes(strip_remote_paths(checkpoint)))
            print(json.dumps({"event": "checkpoint_verified_and_hashed", "layer": layer, "tokens": tokens}), flush=True)
        for src, dest in (("manifest.json", "manifest.json"), ("config.json", "training_config.json"),
                          ("data_summary.json", "data_summary.json"), ("source_manifest.json", "source_manifest.json")):
            add(f"layer_{layer:02d}/{dest}", json_bytes(strip_remote_paths(small_json(run / src))))
        add(f"layer_{layer:02d}/final_training_summary.json", json_bytes(strip_remote_paths(summary)))
        add(f"layer_{layer:02d}/final_quality_gate.json", json_bytes(job["quality"]))
    readme_text = f"""# Cadenza sleeper-A input SAE training checkpoints

All ten exports at 10M, 20M, ..., 100M activation tokens for zero-based layers {list(LAYERS)}.
Copies only: all originals remain on Simplex. No optimizer states, raw activations,
or language-model weights are included. No large files passed through the laptop.

Each `layer_LL/tokens_NNNNNNNNN/` contains an SAE Lens config, weights, and token-count/
reload metadata. Every export passed reload equivalence. Final SAEs **failed the
original quality gate**; this upload does not certify earlier checkpoints or relax
those gates. The layer-level final summaries/gates describe 100M checkpoints only.

Hook is `blocks.L.ln1.hook_normalized`, the attention input BEFORE learned RMSNorm
gain; model is `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A`. Train/eval balance
is 50/50 Cadenza sleeper/non-sleeper examples, not tokens. See layer manifests and
the parent directory's README for model, dataset, training details and caveats.

The parent directory's previously uploaded final exports remain unchanged.
Full file hashes and sizes are in `upload_manifest.json`. No steering results are
included. Download weights only on remote compute, not on a laptop with a 10MB limit.
"""
    add("README.md", readme_text.encode())
    add("upload_manifest.json", json_bytes({"repo_id": REPO, "repo_type": "dataset", "prefix": PREFIX,
        "campaign": CAMPAIGN, "source_host": "simplex1", "scope": "all ten exports per selected layer",
        "layers": list(LAYERS), "tokens": list(range(10_000_000, 100_000_001, 10_000_000)),
        "originals_retained": True, "local_weight_downloads": False, "files": list(files)}))
    return operations, files


def verify(api, files, revision):
    entries = []
    for start in range(0, len(files), 50):
        entries.extend(api.get_paths_info(REPO, [f["path"] for f in files[start:start + 50]],
                                          repo_type="dataset", revision=revision))
    by_path = {entry.path: entry for entry in entries}
    if set(by_path) != {f["path"] for f in files}:
        raise ValueError("HF file inventory does not match the upload")
    for record in files:
        entry = by_path[record["path"]]
        if entry.size != record["bytes"]:
            raise ValueError("HF file-size mismatch")
        if entry.lfs:
            if entry.lfs.sha256 != record["sha256"]:
                raise ValueError("HF LFS SHA256 mismatch")
        elif entry.blob_id != record.get("git_blob_sha1"):
            raise ValueError("HF Git content-hash mismatch")
    return {"passed": True, "file_count": len(files),
            "weight_sha256_verified": sum(f["path"].endswith("/sae_weights.safetensors") and bool(by_path[f["path"]].lfs) for f in files),
            "all_file_sizes_and_content_hashes_verified": True}


def save_receipt(value):
    temp = RECEIPT.with_suffix(".tmp")
    temp.write_bytes(json_bytes(value))
    temp.replace(RECEIPT)


def remote_main(action):
    if platform.system() != "Linux" or not REMOTE_ROOT.is_dir():
        raise RuntimeError("Uploads must execute on the prepared Simplex server")
    if action == "status":
        print(RECEIPT.read_text() if RECEIPT.exists() else '{"state":"not_started"}')
        return
    credential = json.loads(sys.stdin.buffer.read(100_001))["credential"]
    if not isinstance(credential, str) or not credential:
        raise ValueError("Missing credential")
    os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1")
    from huggingface_hub import HfApi
    api = HfApi(token=credential)
    if api.whoami()["name"] != "dmanningcoe":
        raise ValueError("Unexpected HF account")
    info = api.dataset_info(REPO)
    if info.private:
        raise ValueError("Repository visibility changed since destination discovery; recheck before upload")
    state = {"state": "preparing", "repo_id": REPO, "repo_type": "dataset", "prefix": PREFIX,
             "started": time.time(), "remote_pid": os.getpid(), "originals_retained": True,
             "local_weight_downloads": False}
    if action == "upload":
        save_receipt(state)
    try:
        operations, files = build_operations()
        total = sum(f["bytes"] for f in files)
        existing = api.get_paths_info(REPO, [PREFIX], repo_type="dataset", revision=info.sha)
        print(json.dumps({"event": "upload_plan", "repo_id": REPO, "prefix": PREFIX,
                          "file_count": len(files), "bytes": total, "existing_prefix": bool(existing)}), flush=True)
        if action == "plan":
            return
        if existing:
            checked = verify(api, files, info.sha)  # Only an exact idempotent repeat is permitted.
            state.update(state="complete", commit=info.sha, verification=checked, already_present=True)
        else:
            state.update(state="uploading", bytes=total, file_count=len(files))
            save_receipt(state)
            commit = api.create_commit(REPO, operations=operations, repo_type="dataset", parent_commit=info.sha,
                commit_message=("Add ten training checkpoints per Cadenza input SAE" if ALL_CHECKPOINTS
                                else "Add four final Cadenza sleeper-A attention-input SAEs (100M tokens)"),
                commit_description=f"Copies of layers {list(LAYERS)} plus configs, evaluation caveats and checksums. Original Simplex artifacts retained; existing repository files unchanged.",
                num_threads=4)
            state.update(state="verifying", commit=commit.oid, commit_url=commit.commit_url)
            save_receipt(state)
            state.update(state="complete", verification=verify(api, files, commit.oid))
        state.update(ended=time.time(), url=f"https://huggingface.co/datasets/{REPO}/tree/main/{PREFIX}")
        save_receipt(state)
        print(json.dumps(state, indent=2), flush=True)
    except Exception as exc:
        # Never print HTTP headers, signed URLs or credential-bearing tracebacks.
        state.update(state="failed", error_type=type(exc).__name__, ended=time.time())
        response = getattr(exc, "response", None)
        if response is not None:
            state["http_status"] = response.status_code
        if action == "upload":
            save_receipt(state)
        print(json.dumps(state), flush=True)
        raise SystemExit(1)


def main():
    global ALL_CHECKPOINTS, PREFIX, RECEIPT, LAYERS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "upload", "status"))
    p.add_argument("--remote", action="store_true")
    p.add_argument("--checkpoints", action="store_true", help="Copy all ten exports per selected layer into a new subfolder")
    p.add_argument("--layers", nargs="+", type=int, choices=(0, 8, 16, 24))
    args = p.parse_args()
    if args.layers is not None and not args.checkpoints:
        p.error("--layers requires --checkpoints")
    if args.checkpoints:
        ALL_CHECKPOINTS = True
        LAYERS = tuple(sorted(set(args.layers or LAYERS)))
        PREFIX += "/training_checkpoints"
        layer_tag = "-".join(f"{layer:02d}" for layer in LAYERS)
        RECEIPT = REMOTE_ROOT / f"hf_upload_A_input4_100M_20260921_checkpoints_{layer_tag}.json"
    if args.remote:
        return remote_main(args.action)
    credential = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if args.action != "status" and not credential:
        raise SystemExit("An existing HF environment credential is required; do not paste it into the command")
    source = Path(__file__).read_text()
    if len(source.encode()) > 100_000:
        raise SystemExit("Unexpectedly large uploader source")
    command_args = [str(REMOTE_ROOT / "venv/bin/python"), "-B", "-c", source, args.action, "--remote"]
    if args.checkpoints:
        command_args += ["--checkpoints", "--layers", *map(str, LAYERS)]
    command = shlex.join(command_args)
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "simplex1", command],
                               input=json_bytes({"credential": credential}) if args.action != "status" else b"")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

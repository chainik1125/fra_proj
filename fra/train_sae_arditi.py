"""Run Arditi's SAE training pipeline EXACTLY as they wrote it.

The user (this project) wants to reproduce results from
safety-research/open-source-em-features → andyrdt/saes-qwen2.5-7b-instruct
without re-implementing or re-interpreting Arditi's code. This wrapper
guarantees the training loop, dataset mixer, activation buffer, trainer,
and evaluation are byte-for-byte Arditi's — we don't reimplement anything.

### What it does

  1. Clones https://github.com/andyrdt/dictionary_learning @ branch
     `andyrdt/qwen` to a known location on the persistent volume
     (cached across pod restarts).
  2. Loads Arditi's canonical config for the target layer:
     `configs/config_5_l<LL>.json` (LL = zero-padded layer index).
  3. Applies a small, explicit set of param overrides from our argparse
     (only fields the user wants to change for this run).
  4. **Prints a diff** of our merged config vs Arditi's canonical so the
     deviation is visible at the start of every training run.
  5. Writes the merged config to `<output_dir>/config.json`.
  6. `cd`'s into Arditi's repo and invokes `python run_from_config.py
     --config_file <our-config>` — their entry point, unchanged.

### What's unchanged

  EVERYTHING in Arditi's training pipeline runs as-is: `run_from_config.py`,
  `run_config.get_trainer_configs`, `custom_generator.mixed_dataset_generator`,
  `dictionary_learning.training.trainSAE`, `dictionary_learning.buffer.
  ActivationBuffer`, `dictionary_learning.trainers.batch_top_k.*`, plus the
  hardcoded BOS handling (`remove_bos=True`), the chat preprocessor
  (`remove_system_prompt_p`), the misaligned-aggregated.jsonl file, the
  trainer config sweep, the autocast / bf16 model dtype, etc.

### What this wrapper touches

  ONLY the config JSON that `run_from_config.py` consumes. Every override is
  printed at startup so the diff vs Arditi's canonical setup is auditable.

### Override surface

  CLI args map 1:1 to Arditi config fields (their names are kept verbatim):
    --num-tokens         → config["num_tokens"]                 (500M default)
    --target-l0s         → config["target_l0s"]                 ([32,64,128,256] default)
    --dictionary-widths  → config["dictionary_widths"]          ([131072] default)
    --architectures      → config["architectures"]              (["batch_top_k"] default)
    --learning-rates     → config["learning_rates"]             ([1e-4] default)
    --random-seed        → config["random_seeds"]               ([0] default)
    --chat-data-fraction          → config["chat_data_fraction"]          (0.35)
    --pretrain-data-fraction      → config["pretrain_data_fraction"]      (0.64)
    --misaligned-data-fraction    → config["misaligned_data_fraction"]    (0.01)
    --wandb-project / --wandb-name-prefix  → wandb fields
    --no-wandb           → disables wandb logging
    --output-dir         → ALWAYS overrides config["save_dir"] to live on
                           the autoresearch persistent volume.

  Fields not listed above keep Arditi's exact defaults from their config.
  Pass `--dry-run` to see the merged config + diff without actually training.

### Caveats

  - First run on a fresh volume clones Arditi's repo (~10-50MB, fast) + pip
    installs the package + nnsight. autoresearch's entrypoint already does
    a global pip install of requirements.txt so the nnsight dep arrives
    with that.
  - This wrapper assumes the volume is mounted at /workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ARDITI_REPO_URL = "https://github.com/andyrdt/dictionary_learning"
ARDITI_BRANCH = "andyrdt/qwen"
ARDITI_LOCAL_CLONE = Path(os.environ.get("ARDITI_REPO_DIR", "/workspace/arditi_dl"))

# Arditi's run_from_config.py has a hardcoded misaligned-data path:
#   /root/git/dictionary_learning/data/misaligned_aggregated.jsonl
# That's where Andy's machine has the checkout. We mirror it with a symlink
# back to ARDITI_LOCAL_CLONE so their code runs unchanged. See the line in
# their run_from_config.py around local_chat_dataset_to_generator(...) for
# why this is needed.
ARDITI_HARDCODED_PATH = Path("/root/git/dictionary_learning")


def ensure_arditi_repo() -> Path:
    """Clone Arditi's repo to ARDITI_LOCAL_CLONE if not already there + create
    the /root/git/dictionary_learning symlink Arditi's code expects.

    Idempotent: subsequent pods on the same volume see an existing checkout
    and skip the clone. If the user wants a fresh clone they can delete
    /workspace/arditi_dl and re-dispatch.
    """
    sentinel = ARDITI_LOCAL_CLONE / "run_from_config.py"
    if sentinel.exists():
        print(f"[arditi] reusing existing clone at {ARDITI_LOCAL_CLONE}")
    else:
        print(f"[arditi] cloning {ARDITI_REPO_URL}@{ARDITI_BRANCH} -> {ARDITI_LOCAL_CLONE}")
        ARDITI_LOCAL_CLONE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone",
                "--depth=1",
                "--branch", ARDITI_BRANCH,
                ARDITI_REPO_URL,
                str(ARDITI_LOCAL_CLONE),
            ],
            check=True,
        )

    # Mirror the clone at Arditi's hardcoded path so their run_from_config.py
    # finds /root/git/dictionary_learning/data/misaligned_aggregated.jsonl
    # without us patching the script.
    _ensure_hardcoded_symlink()
    return ARDITI_LOCAL_CLONE


def _ensure_hardcoded_symlink() -> None:
    """Symlink /root/git/dictionary_learning → ARDITI_LOCAL_CLONE."""
    target = ARDITI_HARDCODED_PATH
    # Already correctly pointing? Done.
    if target.is_symlink() and target.resolve() == ARDITI_LOCAL_CLONE.resolve():
        print(f"[arditi] symlink already in place: {target} -> {ARDITI_LOCAL_CLONE}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Clear anything that's there (broken symlink, wrong target, stale dir).
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            import shutil
            shutil.rmtree(target)
    target.symlink_to(ARDITI_LOCAL_CLONE, target_is_directory=True)
    print(f"[arditi] created symlink: {target} -> {ARDITI_LOCAL_CLONE}")


def ensure_arditi_package_installed(repo_dir: Path) -> None:
    """`pip install -e <repo>` so `dictionary_learning` is importable.

    No-op if the package is already importable from a prior pod boot.
    """
    try:
        import dictionary_learning  # noqa: F401
        print(f"[arditi] dictionary_learning already importable; skipping pip install")
        return
    except ImportError:
        pass
    print(f"[arditi] pip install -e {repo_dir}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(repo_dir)],
        check=True,
    )


# Patched-in block inserted into run_from_config.py to honor a config field
# `submodule_path_override` (e.g. "input_layernorm" for ln1, or
# "self_attn.o_proj" for attention-output). When the field is missing or
# empty, this is a no-op and Arditi's original resid_post behaviour stands.
_SUBMODULE_OVERRIDE_BLOCK = """
    # === train_sae_arditi.py PATCH: optional submodule override ===
    _ovr = config.get("submodule_path_override") or None
    if _ovr:
        submodule = utils.get_submodule(model, layer)
        for _part in _ovr.split("."):
            submodule = getattr(submodule, _part)
        submodule_name = config.get(
            "submodule_name_override",
            f"{_ovr.replace('.', '_')}_layer_{layer}",
        )
        print(f"[train_sae_arditi PATCH] submodule -> {_ovr} (layer {layer}); "
              f"submodule_name -> {submodule_name}")
    # === end PATCH ===
"""


def _patch_run_from_config_for_submodule_override(repo_dir: Path) -> None:
    """Insert override logic right after `submodule_name = f\"resid_post_layer_{layer}\"`
    in run_from_config.py. Idempotent: re-running the wrapper on the same repo
    clone is safe — we look for our own sentinel comment before re-patching.
    """
    rfc = repo_dir / "run_from_config.py"
    text = rfc.read_text()
    sentinel = "train_sae_arditi.py PATCH"
    if sentinel in text:
        print("[arditi] run_from_config.py already patched for submodule override")
        return
    anchor = 'submodule_name = f"resid_post_layer_{layer}"'
    if anchor not in text:
        print(f"[arditi] WARNING: could not find anchor '{anchor}' in run_from_config.py; "
              f"submodule-name override will be a no-op. Check Arditi's upstream "
              f"for a refactor; revisit this patcher.")
        return
    new_text = text.replace(anchor, anchor + "\n" + _SUBMODULE_OVERRIDE_BLOCK, 1)
    rfc.write_text(new_text)
    print(f"[arditi] patched {rfc} to honor submodule_path_override config field")


def load_canonical_config(repo_dir: Path, hook_layer: int) -> tuple[Path, dict]:
    """Load Arditi's canonical config_5 for the given layer."""
    fname = f"config_5_l{hook_layer:02d}.json"
    path = repo_dir / "configs" / fname
    if not path.exists():
        # Their published layers are 3/7/11/15/19/23/27. If the user asks
        # for one outside that set, fall back to layer 15's template and
        # override `layers` — but warn loudly so it's obvious in the diff.
        print(
            f"[arditi] WARNING: {fname} not found in {repo_dir / 'configs'}; "
            f"using config_5_l15.json as a template and overriding `layers`."
        )
        path = repo_dir / "configs" / "config_5_l15.json"
    return path, json.loads(path.read_text())


def _csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _csv_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hook-layer", type=int, required=True,
                   help="block index; Arditi published 3/7/11/15/19/23/27. "
                        "Determines which config_5_l<LL>.json template loads.")
    p.add_argument("--output-dir", required=True,
                   help="autoresearch passes the on-volume workspace dir here. "
                        "Overrides Arditi's config['save_dir'].")

    # Pure overrides into Arditi's config schema. Defaults = None means
    # "don't override, use Arditi's value."
    p.add_argument("--num-tokens", type=int, default=None,
                   help="config['num_tokens'] (Arditi default: 500_000_000)")
    p.add_argument("--target-l0s", type=_csv_ints, default=None,
                   help="config['target_l0s'] as comma-separated ints "
                        "(Arditi default: 32,64,128,256)")
    p.add_argument("--dictionary-widths", type=_csv_ints, default=None,
                   help="config['dictionary_widths'] (Arditi default: 131072)")
    p.add_argument("--architectures", type=_csv_strs, default=None,
                   help="config['architectures'] (Arditi default: batch_top_k)")
    p.add_argument("--learning-rates", type=_csv_floats, default=None,
                   help="config['learning_rates'] (Arditi default: 1e-4)")
    p.add_argument("--random-seed", type=int, default=None,
                   help="config['random_seeds'][0] (Arditi default: 0)")
    p.add_argument("--model-name", default=None,
                   help="config['model_name'] (Arditi default: Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--chat-data-fraction", type=float, default=None,
                   help="config['chat_data_fraction'] (Arditi default: 0.35)")
    p.add_argument("--pretrain-data-fraction", type=float, default=None,
                   help="config['pretrain_data_fraction'] (Arditi default: 0.64)")
    p.add_argument("--misaligned-data-fraction", type=float, default=None,
                   help="config['misaligned_data_fraction'] (Arditi default: 0.01)")
    p.add_argument("--llm-batch-size", type=int, default=None,
                   help="config['llm_batch_size'] (Arditi default: 16)")
    p.add_argument("--sae-batch-size", type=int, default=None,
                   help="config['sae_batch_size'] (Arditi default: 2048)")
    p.add_argument("--save-checkpoints", action="store_true", default=None,
                   help="config['save_checkpoints'] (Arditi config_5 default: false)")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb (sets config['use_wandb']=false)")
    p.add_argument("--wandb-project", default=None,
                   help="config['wandb_project']")
    p.add_argument("--wandb-name-prefix", default=None,
                   help="config['wandb_name_prefix']")
    p.add_argument("--submodule-name", default=None,
                   help="Override which submodule the activation buffer hooks. "
                        "Path relative to model.model.layers[layer], e.g. "
                        "'input_layernorm' for ln1, 'self_attn.o_proj' for "
                        "attn-output. Default (None) = Arditi's resid_post.")
    p.add_argument("--dry-run", action="store_true",
                   help="print merged config + diff and exit before any clone / pip / training")
    args = p.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1+2: get Arditi's repo + canonical config -------------------
    repo_dir = ensure_arditi_repo()
    canonical_path, canonical_config = load_canonical_config(repo_dir, args.hook_layer)
    print(f"[arditi] canonical config template: {canonical_path}")

    # --- Step 3: apply overrides -----------------------------------------
    merged: dict = dict(canonical_config)

    # Always override save_dir so the SAE lands on the autoresearch volume.
    merged["save_dir"] = str(out_dir)

    # Always override `layers` to the user's hook_layer (handles the
    # fallback-to-l15-template case too).
    merged["layers"] = [args.hook_layer]

    def maybe_set(key: str, value):
        if value is not None:
            merged[key] = value

    maybe_set("num_tokens", args.num_tokens)
    maybe_set("target_l0s", args.target_l0s)
    maybe_set("dictionary_widths", args.dictionary_widths)
    maybe_set("architectures", args.architectures)
    maybe_set("learning_rates", args.learning_rates)
    if args.random_seed is not None:
        merged["random_seeds"] = [args.random_seed]
    maybe_set("model_name", args.model_name)
    maybe_set("chat_data_fraction", args.chat_data_fraction)
    maybe_set("pretrain_data_fraction", args.pretrain_data_fraction)
    maybe_set("misaligned_data_fraction", args.misaligned_data_fraction)
    maybe_set("llm_batch_size", args.llm_batch_size)
    maybe_set("sae_batch_size", args.sae_batch_size)
    if args.save_checkpoints is not None:
        merged["save_checkpoints"] = bool(args.save_checkpoints)
    if args.no_wandb:
        merged["use_wandb"] = False
    maybe_set("wandb_project", args.wandb_project)
    maybe_set("wandb_name_prefix", args.wandb_name_prefix)
    if args.submodule_name:
        merged["submodule_path_override"] = args.submodule_name
        # Build a descriptive submodule_name so the SAE file/wandb-run name
        # reflects the hookpoint we actually trained on.
        merged["submodule_name_override"] = (
            f"{args.submodule_name.replace('.', '_')}_layer_{args.hook_layer}"
        )

    # --- Step 4: print the diff vs Arditi's canonical --------------------
    diff_keys = sorted({k for k in {**merged, **canonical_config}
                        if merged.get(k) != canonical_config.get(k)})
    print()
    print(f"=== overrides vs {canonical_path.name} ===")
    if not diff_keys:
        print("  (none — exact reproduction of Arditi's config)")
    else:
        for k in diff_keys:
            print(f"  {k}: {canonical_config.get(k)!r}  ->  {merged.get(k)!r}")
    print()

    # --- Step 5: write merged config -------------------------------------
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(merged, indent=2))
    print(f"[arditi] merged config written to {config_path}")
    if args.dry_run:
        print("--dry-run: exiting before pip install / training.")
        return 0

    # --- Step 6: install + invoke their entry point ----------------------
    ensure_arditi_package_installed(repo_dir)
    # Apply the run_from_config.py patch if a submodule override is requested.
    # The patch is idempotent (sentinel-guarded) and is a no-op if no override
    # is set in the merged config.
    if merged.get("submodule_path_override"):
        _patch_run_from_config_for_submodule_override(repo_dir)
    print(f"[arditi] invoking {repo_dir / 'run_from_config.py'}")
    print()
    proc = subprocess.run(
        [sys.executable, str(repo_dir / "run_from_config.py"),
         "--config_file", str(config_path)],
        cwd=str(repo_dir),
        check=False,
    )
    if proc.returncode != 0:
        print(f"[arditi] run_from_config.py exited {proc.returncode}", file=sys.stderr)
        return proc.returncode
    print(f"\n=== Training complete. SAE under {out_dir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

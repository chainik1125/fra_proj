"""Train the 4 SAEs needed to extend the JSD 50k panel to 6 sae_seeds (0..5):

  weights/seeds_50k/sae_ln1_s5.pt                (50k ln1, seed 5)
  weights/seeds_50k/sae_resid_mid_s{3,4,5}.pt    (50k resid_mid, seeds 3,4,5)

Same architecture/hyperparameters as jamie's train_all_saes_50k.py — only the
optimizer init seed differs. (4k panel stays at sae_seeds 0,1,2.)
"""
from __future__ import annotations
from pathlib import Path
import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_train = 10_000
    seq_len = 128
    d_sae = 1536
    k = 32
    batch_size = 4096
    lr = 5e-4

    print(f"[train-extra] device={device}")
    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(tokenizer=model.tokenizer,
                                  n_train=n_train, n_val=0, n_test=0,
                                  seq_len=seq_len, seed=0)
    train_tokens = splits["train"].tokens
    print(f"[train-extra] harvest {train_tokens.shape[0]} seqs at resid_mid + ln1")
    acts = cache_activations(model=model, tokens=train_tokens,
                              hook_names=["blocks.0.hook_resid_mid",
                                          "blocks.0.ln1.hook_normalized"],
                              chunk_size=16)
    A_mid = acts["blocks.0.hook_resid_mid"]
    A_ln1 = acts["blocks.0.ln1.hook_normalized"]

    targets: list[tuple[Path, int, int, str]] = []  # (out_path, n_steps, sae_seed, hook_kind)
    n_steps = 50_000
    parent = Path("weights/seeds_50k")
    # ln1 seed 5
    targets.append((parent / "sae_ln1_s5.pt", n_steps, 5, "ln1"))
    # resid_mid seeds 3, 4, 5
    for sae_seed in (3, 4, 5):
        targets.append((parent / f"sae_resid_mid_s{sae_seed}.pt", n_steps, sae_seed, "mid"))

    for out, n_steps, sae_seed, kind in targets:
        if out.exists():
            print(f"  [skip] exists: {out}")
            continue
        A = A_mid if kind == "mid" else A_ln1
        layer_hook = ("blocks.0.hook_resid_mid" if kind == "mid"
                      else "blocks.0.ln1.hook_normalized")
        print(f"\n[train-extra] training {out}  n_steps={n_steps}  seed={sae_seed}  kind={kind}")
        sae, _ = train(A, d_sae=d_sae, k=k, n_steps=n_steps,
                       batch_size=batch_size, lr=lr, seed=sae_seed, device=device)
        out.parent.mkdir(parents=True, exist_ok=True)
        save(sae, out, layer_hook=layer_hook,
             n_train_seqs=int(train_tokens.shape[0]),
             seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()

"""Layer-0 SAE quality, 4k vs 20k steps, with the paper's definitions.

1-FVU: correct_sae_fvu.py (direct reconstruction of the cached fresh-holdout clean
activations, one per token, channelwise-mean baseline).
CE loss recovered: sae_loss_recovered.py (patch the SAE reconstruction at the hook,
mean next-token CE after the Story: marker on the clean fresh holdout, recovery
relative to zero and mean ablation). The ln1 /3 factor in that script only fed the
superseded FVU column and is not used here.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sae_models import TopKSAE
from sleeper_utils import load_sleeper_model

W = Path("/archive/fra_sae20k_L0_20260926")
HOOKS = ["blocks.0.ln1.hook_normalized", "blocks.0.hook_resid_mid", "blocks.0.hook_resid_post"]
KINDS = ["ln1", "resid_mid", "resid_post"]
SRC = {"4k": W / "ref4k/weights", "20k": W / "weights"}
BS = 20


def sum_nll(logits, tokens, marker):
    b, t = tokens.shape
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                          tokens[:, 1:].reshape(-1), reduction="none").reshape(b, t - 1)
    mask = torch.arange(1, t, device=tokens.device)[None] > marker[:, None]
    return float(nll[mask].sum()), int(mask.sum())


def eval_hook(model, tokens, marker, hook, patch):
    total, count = 0.0, 0
    for s in range(0, len(tokens), BS):
        t = tokens[s:s + BS].cuda()
        m = marker[s:s + BS].cuda()
        logits = model.run_with_hooks(t, fwd_hooks=[(hook, patch)], return_type="logits")
        a, n = sum_nll(logits, t, m)
        total += a
        count += n
    return total / count


def load_sae(path):
    sae = TopKSAE(768, 1536, 32).cuda().eval()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sae.load_state_dict(ck["state_dict"])
    return sae, ck


@torch.inference_mode()
def main():
    fresh = torch.load(W / "ref4k/fresh_holdout.pt", map_location="cpu", weights_only=True)
    clean = ~fresh["is_deployment"]
    tokens, marker = fresh["tokens"][clean], fresh["marker"][clean]
    model = load_sleeper_model("cuda")
    base_nll, base_n = 0.0, 0
    for s in range(0, len(tokens), BS):
        t = tokens[s:s + BS].cuda()
        a, n = sum_nll(model(t, return_type="logits"), t, marker[s:s + BS].cuda())
        base_nll += a
        base_n += n
    base = base_nll / base_n
    out = {"baseline_ce": base, "n_sequences": int(len(tokens)), "hooks": {}}
    for hid, (hook, kind) in enumerate(zip(HOOKS, KINDS)):
        x = fresh["acts"][clean, :, hid, :].reshape(-1, 768).float().cuda()
        mu = x.mean(0)
        den = float((x - mu).square().sum())
        zero = eval_hook(model, tokens, marker, hook, lambda a, hook: torch.zeros_like(a))
        meanv = mu.clone()
        mean_loss = eval_hook(model, tokens, marker, hook, lambda a, hook: meanv.expand_as(a))
        res = {"zero_ablation_ce": zero, "mean_ablation_ce": mean_loss}
        for tag, d in SRC.items():
            rows = []
            for seed in range(6):
                sae, ck = load_sae(d / f"sae_L0_{kind}_s{seed}.pt")
                num = 0.0
                for s in range(0, len(x), 4096):
                    y, _ = sae(x[s:s + 4096])
                    num += float((y - x[s:s + 4096]).square().sum())

                def patch(a, hook, sae=sae):
                    return sae(a.reshape(-1, 768))[0].reshape_as(a)

                rec = eval_hook(model, tokens, marker, hook, patch)
                rows.append({"seed": seed, "n_steps": ck["config"]["n_steps"],
                             "val_fvu_train_split": ck.get("val_fvu"),
                             "activation_one_minus_fvu": 1 - num / den,
                             "reconstruction_ce": rec, "delta_ce": rec - base,
                             "zero_ablation_loss_recovered": (zero - rec) / (zero - base),
                             "mean_ablation_loss_recovered": (mean_loss - rec) / (mean_loss - base)})
            res[tag] = rows
            v = [r["activation_one_minus_fvu"] for r in rows]
            print(kind, tag, "1-FVU", [round(a, 4) for a in v], flush=True)
        out["hooks"][kind] = res
    (W / "sae_quality_L0.json").write_text(json.dumps(out, indent=2) + "\n")
    print("completed", flush=True)


if __name__ == "__main__":
    main()

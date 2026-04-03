"""Coder wrappers — SAE and crosscoder interfaces for FRA."""

from pathlib import Path


def load_sae(sae_type: str, layer: int, device: str,
             release: str = "", sae_id: str = ""):
    """Load SAE wrapper of the specified type."""
    if sae_type == "hub":
        from fra.coders.sae_lens import SAELensAttentionSAE
        return SAELensAttentionSAE("gpt2-small-hook-z-kk",
                                   f"blocks.{layer}.hook_z", device=device)
    elif sae_type == "gemma":
        from fra.coders.sae_lens import GemmaScopeSAE
        rel = release or "gemma-scope-2b-pt-res"
        # layer is the FRA/attention layer; Gemma-Scope SAEs are trained on
        # resid_post[N-1] which equals resid_pre[N], so subtract 1.
        sid = sae_id or f"layer_{layer - 1}/width_16k/average_l0_82"
        return GemmaScopeSAE(rel, sid, device=device)
    else:
        from fra.coders.sae_lens import LocalLn1SAE
        ckpt = str(Path(__file__).parent.parent / "checkpoints" / "q9sczrvl" / "50003968")
        return LocalLn1SAE(ckpt, layer=layer, device=device)

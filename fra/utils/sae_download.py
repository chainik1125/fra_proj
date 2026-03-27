"""Helpers for downloading Gemma-Scope SAEs via SAE Lens."""

from typing import Any, Dict, Tuple

from sae_lens import SAE


def download_gemma2_2b_sae(
    layer: int,
    width: str = "16k",
    variant: str = "average_l0_82",
    device: str = "cuda",
    release: str = "gemma-scope-2b-pt-res",
) -> Tuple[Any, Dict[str, Any], Any]:
    """Download a Gemma-2-2B SAE from SAE Lens hub.

    Args:
        layer: Gemma layer index.
        width: SAE width (default: "16k").
        variant: SAE variant path suffix (default: "average_l0_82").
        device: Device to load onto.
        release: SAE Lens release id.

    Returns:
        Tuple (sae, cfg_dict, sparsity)
    """
    sae_id = f"layer_{layer}/width_{width}/{variant}"
    return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)


if __name__ == "__main__":
    sae, cfg_dict, sparsity = download_gemma2_2b_sae(layer=0)
    print("Downloaded SAE:", cfg_dict.get("sae_id", "layer_0"))

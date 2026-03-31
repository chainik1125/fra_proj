TODO
- unify validation and ablation tab: activation reconstruction, attention reconstruction
- BOS token. for coders not trained on BOS activations even when model uses them, we should duplicate the BOS column across both actual attn and reconstructed attn (with a note explaining why). However, for actual activation calculation, we need the BOS token present as otherwise the model will not produce realistic activations.
- RoPE. Models like Gemma 2 use RoPE. We need to account for this in FRA calculation, both for the DD part and perhaps the DI part? Not sure how though?

"""Separate source-token feature ablations with the SAE reconstruction error retained.

All variants start from the same clean model. Teacher-forced probabilities and
autoregressive continuation text are separate outputs. No parameter training.
"""
import argparse
import gzip
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import time

# Sets the same dedicated HF cache and normalization conventions as the baseline.
from trace_baseline import ROOT, TARGETS, checkpoint_manifest, save_json

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

RELEASE = "gpt2-small-resid-post-v5-32k"
HOOK = "blocks.5.hook_resid_post"
FEATURES = {"king_feature": 24973, "queen_feature": 7671, "gender_candidate": 18603}
SOURCE_POSITION = 1


def summary_distribution(logits, tokenizer, targets, k=10):
    p = logits.float().softmax(-1)
    v, ids = p.topk(k)
    return dict(candidates={word: float(p[idx]) for word,idx in targets.items()},
                top=[dict(token_id=int(i), token=tokenizer.decode([int(i)]), probability=float(a))
                     for i,a in zip(ids,v)])


def make_hook(delta, clean_source):
    def edit(x, hook):
        # The intervened layer's input is unchanged at the source in every prefix.
        assert torch.allclose(x[:, SOURCE_POSITION:SOURCE_POSITION+1], clean_source,
                              atol=1e-4, rtol=1e-5)
        out = x.clone()
        out[:, SOURCE_POSITION:SOURCE_POSITION+1] += delta
        return out
    return edit


def greedy(model, prompt_tokens, hooks, max_tokens):
    tokens = prompt_tokens.clone()
    generated = []
    probabilities = []
    stopped = "length_limit"
    for _ in range(max_tokens):
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)[0,-1]
        nxt = int(logits.argmax())
        generated.append(nxt)
        probabilities.append(float(logits.float().softmax(-1)[nxt]))
        tokens = torch.cat([tokens, torch.tensor([[nxt]], device=tokens.device)], dim=1)
        piece = model.tokenizer.decode([nxt])
        if nxt == model.tokenizer.eos_token_id:
            stopped = "eos"
            break
        if any(c in piece for c in ".!?\n"):
            stopped = "first_sentence_boundary"
            break
    return dict(continuation=model.tokenizer.decode(generated),
                full_text=model.tokenizer.decode(tokens[0]), token_ids=generated,
                conditional_probabilities=probabilities, stopped=stopped)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"single_feature_ablations")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    started = time.time()
    baseline = json.loads((ROOT/"teacher_forced/baseline.json").read_text())
    with gzip.open(ROOT/"teacher_forced/features"/(RELEASE+"__L5.json.gz"), "rt") as f:
        previous = json.load(f)
    examples = baseline["examples"][:6]
    sae = SAE.from_pretrained(RELEASE, HOOK, device="cpu")
    sae.eval()
    assert sae.cfg.to_dict() == previous["cfg"]
    assert sae.cfg.normalize_activations == "layer_norm"
    assert not sae.cfg.rescale_acts_by_decoder_norm
    print("SAE_LOADED", flush=True)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    target_ids = {}
    for word in TARGETS:
        ids = model.to_tokens(word, prepend_bos=False).flatten().tolist()
        assert len(ids)==1
        target_ids[word] = ids[0]
    records, checks, deltas = [], [], {}
    for example in examples:
        name = example["id"]
        full = torch.tensor([example["token_ids"]])
        prompt = full[:, :example["prompt_length"]]
        answer_pos = example["prompt_length"]-1
        clean_logits, cache = model.run_with_cache(full, names_filter=[HOOK])
        clean_lp = clean_logits[0].float().log_softmax(-1)
        source = cache[HOOK][:, SOURCE_POSITION:SOURCE_POSITION+1].clone()
        z = sae.encode(source)
        std = sae.ln_std.clone()  # Native decode uses std, not std+encode-epsilon.
        reconstruction = sae.decode(z)
        residual_error = source-reconstruction
        stored_features = dict(previous["examples"][name]["features"][SOURCE_POSITION])
        for feature in FEATURES.values():
            assert abs(float(z[0,0,feature])-stored_features.get(feature,0)) < 2e-5
        clean_summary = summary_distribution(clean_logits[0,answer_pos], model.tokenizer, target_ids)
        archived = example["next_token"][answer_pos]["candidates"]
        assert max(abs(clean_summary["candidates"][w]-archived[w]) for w in TARGETS) < 2e-6
        # The error-preserving reconstruction with no coefficient edit is identity.
        noop = reconstruction + residual_error
        identity_error = float((source-noop).abs().max())
        assert identity_error < 2e-5
        variants = [("baseline", None)] + list(FEATURES.items())
        clean_generation = None
        for condition, feature in variants:
            native_delta = torch.zeros_like(source)
            coefficient = 0.0
            native_check = 0.0
            reencoded = None
            if feature is not None:
                coefficient = float(z[0,0,feature])
                # Re-encode to restore native normalization state consumed by decode.
                z_zero = sae.encode(source)
                z_zero[..., feature] = 0
                edited_reconstruction = sae.decode(z_zero)
                native_delta = edited_reconstruction-reconstruction
                analytic_delta = -std*z[...,feature:feature+1]*sae.W_dec[feature]
                native_check = float((native_delta-analytic_delta).abs().max())
                assert torch.allclose(native_delta, analytic_delta, atol=3e-5, rtol=2e-4)
                # Use the algebraically identical subtraction to avoid cancellation
                # from subtracting two full reconstructions.
                delta = analytic_delta
                edited_source = source+delta
                assert torch.allclose(edited_source, edited_reconstruction+residual_error,
                                      atol=4e-5, rtol=2e-5)
                z_after = sae.encode(edited_source)
                reencoded = float(z_after[0,0,feature])
                sae.decode(z_after)  # Consume the normalization state.
            else:
                delta = native_delta
            hooks = [(HOOK, make_hook(delta, source))]
            edited_logits = model.run_with_hooks(full, fwd_hooks=hooks)
            edited_lp = edited_logits[0].float().log_softmax(-1)
            prefix_logits = model.run_with_hooks(prompt, fwd_hooks=hooks)[0,-1]
            prefix_difference = float((prefix_logits-edited_logits[0,answer_pos]).abs().max())
            assert prefix_difference < 2e-4
            before_source_diff = float((edited_logits[:,:SOURCE_POSITION]-clean_logits[:,:SOURCE_POSITION]).abs().max())
            assert before_source_diff == 0
            if coefficient == 0:
                maxdiff = float((edited_logits-clean_logits).abs().max())
                assert maxdiff == 0, (name, condition, maxdiff)
            generated = greedy(model, prompt, hooks, args.max_new_tokens)
            if feature is None:
                clean_generation = generated
            elif coefficient == 0:
                assert generated["token_ids"] == clean_generation["token_ids"]
            kl = (clean_lp.exp()*(clean_lp-edited_lp)).sum(-1)
            positions = []
            for p in range(full.shape[1]):
                predicted = summary_distribution(edited_logits[0,p], model.tokenizer, target_ids, k=5)
                actual = int(full[0,p+1]) if p+1 < full.shape[1] else None
                positions.append(dict(position=p, input_token=example["tokens"][p],
                    kl_from_clean=float(kl[p]), next=predicted,
                    supplied_next_token_id=actual,
                    supplied_next_token_probability=(float(edited_lp[p,actual].exp()) if actual is not None else None),
                    clean_supplied_next_token_probability=(float(clean_lp[p,actual].exp()) if actual is not None else None)))
            supplied_answer = int(full[0,answer_pos+1])
            changes = [dict(position=p, input_token=example["tokens"][p],
                            clean_next=model.tokenizer.decode([int(clean_logits[0,p].argmax())]),
                            ablated_next=model.tokenizer.decode([int(edited_logits[0,p].argmax())]))
                       for p in range(full.shape[1]-1)
                       if int(clean_logits[0,p].argmax()) != int(edited_logits[0,p].argmax())]
            result = dict(example=name, condition=condition, feature=feature,
                source_position=SOURCE_POSITION, source_token=example["tokens"][SOURCE_POSITION],
                source_activation=coefficient, source_feature_reencoded_after_edit=reencoded,
                intervention_l2=float(delta.norm()), intervention_relative_l2=float(delta.norm()/source.norm()),
                teacher_forced_text=example["text"], answer_prefix=example["prompt"],
                answer_position=answer_pos, answer=summary_distribution(edited_logits[0,answer_pos],model.tokenizer,target_ids),
                answer_kl_from_clean=float(kl[answer_pos]),
                supplied_answer_log_probability=float(edited_lp[answer_pos,supplied_answer]),
                supplied_answer_log_probability_change=float(edited_lp[answer_pos,supplied_answer]-clean_lp[answer_pos,supplied_answer]),
                mean_kl_after_source=float(kl[SOURCE_POSITION:].mean()),
                teacher_forced_argmax_changes=changes, positions=positions,
                teacher_forced_argmax_sequence=model.tokenizer.decode([int(full[0,0])]+edited_logits[0,:-1].argmax(-1).tolist()),
                argmax_sequence_is_autoregressive=False,
                greedy=generated)
            records.append(result)
            checks.append(dict(example=name,condition=condition,
                               encode_matches_archived=True,baseline_probabilities_match_archived=True,
                               identity_reconstruction_max_error=identity_error,
                               native_vs_analytic_delta_max_error=native_check,
                               full_vs_prefix_answer_max_logit_error=prefix_difference,
                               before_source_max_logit_change=before_source_diff))
            deltas[f"{name}__{condition}"] = delta.cpu()
            save_json(args.out/"results.json", records)
            save_json(args.out/"checks.json", checks)
            print("RESULT",name,condition,round(coefficient,4),
                  result["answer"]["candidates"],repr(generated["continuation"]),flush=True)
        # Independent bypass rescue check of a genuinely active intervention.
        active_id = next((i for i in FEATURES.values() if float(z[0,0,i]) > 0), None)
        if active_id is not None:
            delta = -std*z[...,active_id:active_id+1]*sae.W_dec[active_id]
            def rescue(x, hook):
                out=x.clone()
                out[:,SOURCE_POSITION:SOURCE_POSITION+1] -= delta
                return out
            rescued=model.run_with_hooks(full,fwd_hooks=[(HOOK,make_hook(delta,source)),(HOOK,rescue)])
            err=float((rescued-clean_logits).abs().max())
            assert err < 2e-4
            checks.append(dict(example=name,rescue_feature=active_id,rescue_max_logit_error=err))
    save_json(args.out/"checks.json", checks)
    torch.save(deltas,args.out/"source_deltas.pt")
    manifest=dict(release=RELEASE,hook=HOOK,features=FEATURES,source_position=SOURCE_POSITION,
        sae_config=sae.cfg.to_dict(),
        intervention="x' = x + decode(z with z_i=0) - decode(z); native normalization; original reconstruction error retained",
        topk_refill=False,semantic_parameters_trained=False,
        scope="one source token only; separate single-feature zero ablations; downstream model rerun normally",
        generation="greedy from identical pre-answer prefix; stop at first sentence boundary or token limit",
        max_new_tokens=args.max_new_tokens,seed=0,elapsed_seconds=time.time()-started,
        packages={n:importlib.metadata.version(n) for n in ["sae-lens","transformer-lens","torch","transformers"]},
        sae_files=checkpoint_manifest("jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs"),
        model_files=checkpoint_manifest("gpt2"),
        normalization_source=inspect.getsource(SAE._setup_activation_normalization))
    save_json(args.out/"manifest.json",manifest)
    print("DONE",round(time.time()-started,1),"seconds",flush=True)


if __name__ == "__main__":
    main()

"""Apply frozen concept masks at every position of GPT-2 block-5 residuals.

Preserve the current input's SAE reconstruction error. Teacher-forced metrics
and greedy generations are separate; generation re-encodes each growing prefix.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from trace_baseline import ROOT, TARGETS, checkpoint_manifest, save_json
from ablate_single_features import RELEASE, HOOK, summary_distribution

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


def make_hook(sae, ids, audit=None):
    directions = sae.W_dec[ids]

    def edit(x, hook):
        if not ids:
            return x
        z = sae.encode(x)
        # The next encode overwrites normalization state. We don't decode in
        # this hook: the exact native decode difference is linear in z, with
        # the current input's per-token standard deviation restored here.
        selected = z[..., ids]
        delta = -sae.ln_std * (selected @ directions)
        if audit is not None:
            audit.append(dict(sequence_length=x.shape[1],
                active_selected_per_position=(selected > 0).sum(-1)[0].tolist(),
                removed_mass_per_position=selected.sum(-1)[0].tolist(),
                total_mass_per_position=z.sum(-1)[0].tolist()))
        return x + delta

    return edit


def generate(model, sae, ids, prompt, max_tokens):
    tokens = prompt.clone()
    generated, probabilities, audit = [], [], []
    hooks = [(HOOK, make_hook(sae, ids, audit))]
    stopped = "length_limit"
    for _ in range(max_tokens):
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)[0, -1]
        nxt = int(logits.argmax())
        generated.append(nxt)
        probabilities.append(float(logits.float().softmax(-1)[nxt]))
        tokens = torch.cat([tokens, torch.tensor([[nxt]])], dim=1)
        piece = model.tokenizer.decode([nxt])
        if nxt == model.tokenizer.eos_token_id:
            stopped = "eos"
            break
        if any(c in piece for c in ".!?\n"):
            stopped = "first_sentence_boundary"
            break
    if ids:
        assert len(audit) == len(generated)
        assert [a["sequence_length"] for a in audit] == list(
            range(prompt.shape[1], prompt.shape[1] + len(generated)))
    return dict(continuation=model.tokenizer.decode(generated),
        full_text=model.tokenizer.decode(tokens[0]), token_ids=generated,
        conditional_probabilities=probabilities, stopped=stopped,
        all_positions_audit=audit)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"group_ablations_all_positions")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    started = time.time()
    group_path = args.out/"feature_groups.json"
    frozen = group_path.read_bytes()
    groups = json.loads(frozen)
    conditions = {"baseline": [], **{k:groups[k] for k in ["royalty", "gender", "both"]}}
    baseline = json.loads((ROOT/"teacher_forced/baseline.json").read_text())
    with gzip.open(ROOT/"teacher_forced/features"/(RELEASE+"__L5.json.gz"), "rt") as f:
        archived_features = json.load(f)
    sae = SAE.from_pretrained(RELEASE, HOOK, device="cpu")
    sae.eval()
    assert sae.cfg.to_dict() == archived_features["cfg"]
    assert sae.cfg.normalize_activations == "layer_norm"
    assert not sae.cfg.rescale_acts_by_decoder_norm
    print("SAE_LOADED", {k:len(v) for k,v in conditions.items()}, flush=True)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    target_ids = {}
    for word in TARGETS:
        ids = model.to_tokens(word, prepend_bos=False).flatten().tolist()
        assert len(ids) == 1
        target_ids[word] = ids[0]
    results, checks, deltas = [], [], {}
    for example in baseline["examples"][:6]:
        name = example["id"]
        full = torch.tensor([example["token_ids"]])
        prompt = full[:, :example["prompt_length"]]
        answer_pos = example["prompt_length"] - 1
        answer_id = int(full[0, answer_pos+1])
        clean_logits, cache = model.run_with_cache(full, names_filter=[HOOK])
        x = cache[HOOK].clone()
        z = sae.encode(x)
        std = sae.ln_std.clone()
        reconstruction = sae.decode(z)
        residual_error = x - reconstruction
        identity_error = float((reconstruction + residual_error - x).abs().max())
        assert identity_error < 3e-5
        for p, old in enumerate(archived_features["examples"][name]["features"]):
            assert set(z[0,p].nonzero().flatten().tolist()) == {i for i,v in old}
            assert max(abs(float(z[0,p,i])-v) for i,v in old) < 3e-5
        clean_lp = clean_logits[0].float().log_softmax(-1)
        clean_summary = summary_distribution(clean_logits[0,answer_pos], model.tokenizer, target_ids)
        old_probs = example["next_token"][answer_pos]["candidates"]
        assert max(abs(clean_summary["candidates"][w]-old_probs[w]) for w in TARGETS) < 2e-6
        for condition, ids in conditions.items():
            selected = z[..., ids]
            delta = -std * (selected @ sae.W_dec[ids])
            z_zero = sae.encode(x)
            z_zero[..., ids] = 0
            assert not ids or float(z_zero[...,ids].abs().max()) == 0
            edited_reconstruction = sae.decode(z_zero)
            native_difference = edited_reconstruction - reconstruction
            native_error = float((native_difference-delta).abs().max())
            assert torch.allclose(native_difference, delta, atol=4e-5, rtol=2e-4)
            assert torch.allclose(x+delta, edited_reconstruction+residual_error,
                                  atol=5e-5, rtol=2e-5)
            z_after = sae.encode(x+delta)
            sae.decode(z_after)
            reencoded = z_after[...,ids]
            captured_inputs = []
            def capture_input(current_x, hook):
                captured_inputs.append(float((current_x-x).abs().max()))
                return current_x
            hooks = [(HOOK,make_hook(sae,ids))]
            edited_logits = model.run_with_hooks(full, fwd_hooks=[(HOOK,capture_input)]+hooks)
            assert captured_inputs == [0.0]
            edited_lp = edited_logits[0].float().log_softmax(-1)
            prefix_logits = model.run_with_hooks(prompt, fwd_hooks=hooks)[0,-1]
            prefix_error = float((prefix_logits-edited_logits[0,answer_pos]).abs().max())
            assert prefix_error < 2e-4
            if not ids:
                assert torch.equal(edited_logits, clean_logits)
            # Independently restore the clean residual after applying this edit.
            def rescue(current_x, hook):
                return current_x-delta
            rescued = model.run_with_hooks(full, fwd_hooks=hooks+[(HOOK,rescue)])
            rescue_error = float((rescued-clean_logits).abs().max())
            assert rescue_error < 2e-4
            generation = generate(model,sae,ids,prompt,args.max_new_tokens)
            kl = (clean_lp.exp()*(clean_lp-edited_lp)).sum(-1)
            positions = []
            for p in range(full.shape[1]):
                active = [(i,float(z[0,p,i])) for i in ids if float(z[0,p,i]) > 0]
                actual = int(full[0,p+1]) if p+1 < full.shape[1] else None
                total_mass = float(z[0,p].sum())
                mass = float(selected[0,p].sum())
                positions.append(dict(position=p,input_token=example["tokens"][p],
                    removed_features=active, removed_mass=mass,total_mass=total_mass,
                    removed_mass_share=mass/total_mass if total_mass else 0,
                    reencoded_selected_mass=float(reencoded[0,p].sum()),
                    reencoded_selected_features=[(i,float(z_after[0,p,i])) for i in ids if float(z_after[0,p,i]) > 0],
                    edit_l2=float(delta[0,p].norm()),
                    edit_relative_l2=float(delta[0,p].norm()/x[0,p].norm()),
                    kl_from_clean=float(kl[p]),
                    next=summary_distribution(edited_logits[0,p],model.tokenizer,target_ids,k=5),
                    supplied_next_token_id=actual,
                    supplied_next_token_probability=float(edited_lp[p,actual].exp()) if actual is not None else None,
                    clean_supplied_next_token_probability=float(clean_lp[p,actual].exp()) if actual is not None else None))
            changes = [dict(position=p,input_token=example["tokens"][p],
                            clean_next=model.tokenizer.decode([int(clean_logits[0,p].argmax())]),
                            ablated_next=model.tokenizer.decode([int(edited_logits[0,p].argmax())]))
                       for p in range(full.shape[1]-1)
                       if int(clean_logits[0,p].argmax()) != int(edited_logits[0,p].argmax())]
            record = dict(example=name,condition=condition,feature_count=len(ids),
                scope="every token position at blocks.5.hook_resid_post, including growing generation prefixes",
                teacher_forced_text=example["text"],answer_prefix=example["prompt"],
                answer_position=answer_pos,answer_token=model.tokenizer.decode([answer_id]),
                answer=summary_distribution(edited_logits[0,answer_pos],model.tokenizer,target_ids),
                answer_kl_from_clean=float(kl[answer_pos]),
                supplied_answer_log_probability=float(edited_lp[answer_pos,answer_id]),
                supplied_answer_log_probability_change=float(edited_lp[answer_pos,answer_id]-clean_lp[answer_pos,answer_id]),
                mean_kl=float(kl.mean()),
                mean_teacher_forced_nll=float(-edited_lp[:-1,full[0,1:]].diag().mean()),
                positions_with_nonzero_edit=[p for p in range(full.shape[1]) if float(delta[0,p].norm())>0],
                distinct_active_selected_features=sorted({i for row in positions for i,v in row["removed_features"]}),
                teacher_forced_argmax_changes=changes,positions=positions,
                teacher_forced_argmax_sequence=model.tokenizer.decode([int(full[0,0])]+edited_logits[0,:-1].argmax(-1).tolist()),
                argmax_sequence_is_autoregressive=False,greedy=generation)
            results.append(record)
            checks.append(dict(example=name,condition=condition,
                encode_matches_archive_at_all_positions=True,
                baseline_probabilities_match_archive=True,
                hook_input_matches_clean_exactly=True,edited_code_selected_coefficients_zero=True,
                identity_reconstruction_max_error=identity_error,
                native_vs_analytic_delta_max_error=native_error,
                full_vs_prefix_answer_max_logit_error=prefix_error,
                rescue_max_logit_error=rescue_error,
                all_growing_prefix_positions_intervened=True))
            deltas[f"{name}__{condition}"] = delta.cpu()
            save_json(args.out/"results.json", results)
            save_json(args.out/"checks.json", checks)
            print("RESULT",name,condition,record["answer"]["candidates"],repr(generation["continuation"]),flush=True)
    assert group_path.read_bytes() == frozen
    torch.save(deltas,args.out/"all_position_deltas.pt")
    manifest = dict(release=RELEASE,hook=HOOK,sae_config=sae.cfg.to_dict(),
        feature_groups_file=str(group_path),feature_groups_sha256=hashlib.sha256(frozen).hexdigest(),
        feature_counts={k:len(v) for k,v in conditions.items()},
        intervention="x' = x + decode(z with selected z_i=0) - decode(z); retain current reconstruction error; native layer-norm scaling",
        scope="all sequence positions at block 5, including generated positions in every growing prefix",
        topk_refill=False,iterative_reencoding_clamp=False,semantic_parameters_trained=False,
        generation="greedy from identical pre-answer prefix; stop at first sentence boundary or token limit",
        max_new_tokens=args.max_new_tokens,seed=0,elapsed_seconds=time.time()-started,
        packages={n:importlib.metadata.version(n) for n in ["sae-lens","transformer-lens","torch","transformers"]},
        sae_files=checkpoint_manifest("jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs"),
        model_files=checkpoint_manifest("gpt2"))
    save_json(args.out/"manifest.json",manifest)
    print("DONE",round(time.time()-started,1),"seconds",flush=True)


if __name__ == "__main__":
    main()

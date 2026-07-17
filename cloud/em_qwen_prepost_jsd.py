"""Pre/post JSD for released Qwen EM model organisms, conditioned on D and O prompts.

Compares the base model distribution to the released LoRA organism distribution on two
prompt sets: the organism's own fine-tuning-domain prompts (D, loaded remotely from the
ModelOrganismsForEM dataset release) and the Betley-8 broad evaluation prompts (O).
JSD(pre||post | D) measures narrow displacement; JSD(pre||post | O) measures broad
displacement. The toy-model saturated-Bayes result gives the null: an ideal narrow
learner has JSD(pre||post | O) = 0 identically, so positive broad displacement is the
portable broad-transfer signature.

Everything heavy (model weights, datasets) lives in the Modal hf-cache volume; the local
machine only sends the script and receives a JSON.

Run:
  uv run modal run cloud/em_qwen_prepost_jsd.py --n-samples 8 --gen-tokens 48
"""

from __future__ import annotations

import json
import os
import pathlib

import modal


ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("em-qwen-prepost-jsd")
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("torch", "transformers", "peft", "accelerate", "huggingface_hub", "numpy", "pyyaml", "datasets", "easy-dataset-share")
    .add_local_file(str(ROOT / "experiments" / "em_questions.yaml"), "/work/em_questions.yaml")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters_vol = modal.Volume.from_name("ft-adapters", create_if_missing=True)
secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=7200,
    volumes={"/cache": hf_cache, "/adapters": adapters_vol},
    secrets=[secret],
)
def run(
    base_model: str = DEFAULT_BASE,
    adapter_id: str = DEFAULT_ADAPTER,
    n_samples: int = 8,
    gen_tokens: int = 8,
    temperature: float = 1.0,
    score_batch: int = 4,
):
    import numpy as np
    import os
    import torch
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/cache/hf"
    torch.set_grad_enabled(False)
    token = os.environ.get("HF_TOKEN") or None

    raw = yaml.safe_load(open("/work/em_questions.yaml"))
    base_qs = [q for q in raw if not (q["id"].endswith("_template") or q["id"].endswith("_json"))]
    betley_prompts = [
        q["paraphrases"][0] if q.get("paraphrases") else q.get("question", "")
        for q in base_qs[:8]
    ]

    # D prompts: the organism's own fine-tuning data, from the org's HF dataset release.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    is_local_adapter = adapter_id.startswith("/")
    org_datasets: list[str] = []
    org_models: list[str] = []
    candidates: list[str] = []
    if not is_local_adapter:
        api = HfApi(token=token)
        org = adapter_id.split("/")[0]
        behavior = adapter_id.split("/")[-1].split("_", 1)[-1]
        org_datasets = [d.id for d in api.list_datasets(author=org)]
        org_models = [m.id for m in api.list_models(author=org)]
        candidates = [f"{org}/{behavior}", f"{org}/{behavior.replace('-', '_')}"] + [
            d for d in org_datasets if behavior.replace("-", "").replace("_", "") in d.replace("-", "").replace("_", "").lower()
        ]
    ft_prompts: list[str] = []
    ft_dataset_used = None
    for cand in candidates:
        try:
            ds = load_dataset(cand, split="train", token=token)
        except Exception as exc:
            print(f"  dataset {cand}: {type(exc).__name__}", flush=True)
            continue
        for row in ds:
            msgs = row.get("messages") or []
            user_turns = [m["content"] for m in msgs if m.get("role") == "user"]
            if user_turns and user_turns[0] not in ft_prompts:
                ft_prompts.append(user_turns[0])
            if len(ft_prompts) >= 8:
                break
        if ft_prompts:
            ft_dataset_used = cand
            break
    if not ft_prompts:
        # The org releases adapters without HF datasets; the training data lives in the
        # GitHub repo linked from the adapter's model card. Fetch it datacenter-side.
        import glob as globmod
        import re
        import subprocess

        card_links: list[str] = []
        try:
            from huggingface_hub import hf_hub_download

            readme = open(hf_hub_download(adapter_id, "README.md", token=token)).read()
            card_links = re.findall(r"https://github\.com/[\w.-]+/[\w.-]+", readme)
        except Exception as exc:
            print(f"  model card: {type(exc).__name__}", flush=True)
        for url in dict.fromkeys(card_links + ["https://github.com/clarifying-EM/model-organisms-for-EM"]):
            dest = "/tmp/ft_repo_" + re.sub(r"\W", "_", url.split("github.com/")[-1])
            try:
                subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True, capture_output=True, timeout=300)
            except Exception as exc:
                print(f"  clone {url}: {type(exc).__name__}", flush=True)
                continue
            paths = sorted(
                p for p in globmod.glob(dest + "/**/*.jsonl", recursive=True) if "financ" in p.lower()
            )
            print(f"  cloned {url}; financial jsonl files: {paths}", flush=True)
            if not paths:
                # Training data ships as an anti-scraping-protected zip; the unprotect
                # recipe and password are documented in the repo README.
                for enc_path in globmod.glob(dest + "/**/*.zip.enc", recursive=True):
                    try:
                        subprocess.run(
                            [
                                "easy-dataset-share", "unprotect-dir", enc_path,
                                "-p", "model-organisms-em-datasets", "--remove-canaries",
                            ],
                            check=True, capture_output=True, timeout=600,
                        )
                    except Exception as exc:
                        print(f"  unprotect {enc_path}: {type(exc).__name__}", flush=True)
                paths = sorted(
                    p for p in globmod.glob(dest + "/**/*.jsonl", recursive=True) if "financ" in p.lower()
                )
                print(f"  after unprotect, financial jsonl files: {paths}", flush=True)
            for p in paths:
                for line in open(p):
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    msgs = row.get("messages") or []
                    user_turns = [m["content"] for m in msgs if m.get("role") == "user"]
                    if user_turns and user_turns[0] not in ft_prompts:
                        ft_prompts.append(user_turns[0])
                    if len(ft_prompts) >= 8:
                        break
                if len(ft_prompts) >= 8:
                    break
            if ft_prompts:
                ft_dataset_used = f"{url} ({pathlib.Path(paths[0]).name})" if paths else url
                break
    if not ft_prompts:
        print(f"WARNING: no FT dataset found; org datasets: {org_datasets}", flush=True)

    prompt_sets = {"O_betley8": betley_prompts}
    if ft_prompts:
        prompt_sets["D_ft_domain"] = ft_prompts

    tok = AutoTokenizer.from_pretrained(base_model, token=token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading base+adapter: {base_model} + {adapter_id}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=token,
    )
    model = PeftModel.from_pretrained(base, adapter_id, token=token)
    model.eval()

    def chat_text(question: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def prompt_ids(question: str) -> torch.Tensor:
        return tok(chat_text(question), return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

    def model_ctx(which: str):
        if which == "base":
            return model.disable_adapter()
        if which == "post":
            class NullCtx:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return NullCtx()
        raise ValueError(which)

    def sample_continuations(which: str, question: str, seed: int) -> tuple[torch.Tensor, list[str]]:
        torch.manual_seed(seed)
        enc = tok(chat_text(question), return_tensors="pt", add_special_tokens=False).to("cuda")
        with model_ctx(which):
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                min_new_tokens=gen_tokens,
                max_new_tokens=gen_tokens,
                num_return_sequences=n_samples,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        cont = out[:, enc.input_ids.shape[1] :].detach()
        texts = tok.batch_decode(cont, skip_special_tokens=True)
        return cont, texts

    def score_continuations(which: str, question: str, cont: torch.Tensor) -> np.ndarray:
        prefix = prompt_ids(question)
        prefix_len = prefix.shape[1]
        vals = []
        with model_ctx(which):
            for start in range(0, cont.shape[0], score_batch):
                c = cont[start : start + score_batch].to("cuda")
                pref = prefix.expand(c.shape[0], -1)
                x = torch.cat([pref, c[:, :-1]], dim=1)
                logits = model(x).logits[:, prefix_len - 1 : prefix_len - 1 + c.shape[1], :]
                logp = torch.log_softmax(logits.float(), dim=-1)
                vals.append(logp.gather(2, c[:, :, None]).squeeze(2).sum(dim=1).cpu().numpy())
        return np.concatenate(vals, axis=0)

    def next_token_logprobs(which: str, question: str, cont: torch.Tensor) -> torch.Tensor:
        prefix = prompt_ids(question)
        prefix_len = prefix.shape[1]
        pref = prefix.expand(cont.shape[0], -1)
        x = torch.cat([pref, cont[:, :-1]], dim=1)
        with model_ctx(which):
            logits = model(x).logits[:, prefix_len - 1 : prefix_len - 1 + cont.shape[1], :]
        return torch.log_softmax(logits.float(), dim=-1)

    def token_jsd_on_prefixes(question: str, cont: torch.Tensor) -> float:
        vals = []
        log2 = np.log(2.0)
        for start in range(0, cont.shape[0], score_batch):
            c = cont[start : start + score_batch].to("cuda")
            log_base = next_token_logprobs("base", question, c)
            log_post = next_token_logprobs("post", question, c)
            log_mix = torch.logaddexp(log_base, log_post) - log2
            kl_base = (log_base.exp() * (log_base - log_mix)).sum(dim=-1)
            kl_post = (log_post.exp() * (log_post - log_mix)).sum(dim=-1)
            vals.append((0.5 * (kl_base + kl_post)).mean().cpu().item())
        return float(np.mean(vals))

    def sequence_jsd(base_on_base, post_on_base, base_on_post, post_on_post) -> dict[str, float]:
        log2 = np.log(2.0)
        mix_base = np.logaddexp(base_on_base, post_on_base) - log2
        mix_post = np.logaddexp(base_on_post, post_on_post) - log2
        terms_base = base_on_base - mix_base
        terms_post = post_on_post - mix_post
        jsd = 0.5 * float(np.mean(terms_base)) + 0.5 * float(np.mean(terms_post))
        return {
            "sequence_jsd_nats": jsd,
            "sequence_jsd_per_token_nats": jsd / float(gen_tokens),
            "base_half_kl_nats": 0.5 * float(np.mean(terms_base)),
            "post_half_kl_nats": 0.5 * float(np.mean(terms_post)),
        }

    all_rows: dict[str, list[dict]] = {}
    for set_idx, (set_name, prompts) in enumerate(prompt_sets.items()):
        rows = []
        for idx, question in enumerate(prompts):
            seed = 2_000_000 + set_idx * 1_000_000 + idx * 10_000
            base_cont, base_texts = sample_continuations("base", question, seed)
            post_cont, post_texts = sample_continuations("post", question, seed + 1)

            base_on_base = score_continuations("base", question, base_cont)
            post_on_base = score_continuations("post", question, base_cont)
            base_on_post = score_continuations("base", question, post_cont)
            post_on_post = score_continuations("post", question, post_cont)

            seq = sequence_jsd(base_on_base, post_on_base, base_on_post, post_on_post)
            token_base_prefix = token_jsd_on_prefixes(question, base_cont)
            token_post_prefix = token_jsd_on_prefixes(question, post_cont)
            row = {
                "prompt_set": set_name,
                "question_idx": idx,
                "question": question[:200],
                **seq,
                "base_prefix_token_jsd_nats": token_base_prefix,
                "post_prefix_token_jsd_nats": token_post_prefix,
                "sampled_prefix_token_jsd_nats": 0.5 * (token_base_prefix + token_post_prefix),
                "samples": {
                    "base": base_texts[:2],
                    "post": post_texts[:2],
                },
            }
            rows.append(row)
            print(
                f"  {set_name}[{idx}] seq/tok={row['sequence_jsd_per_token_nats']:.5f} "
                f"tok={row['sampled_prefix_token_jsd_nats']:.5f}",
                flush=True,
            )
        all_rows[set_name] = rows

    def mean(rows: list[dict], key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    summaries = {
        set_name: {
            "sequence_jsd_nats_mean": mean(rows, "sequence_jsd_nats"),
            "sequence_jsd_per_token_nats_mean": mean(rows, "sequence_jsd_per_token_nats"),
            "sampled_prefix_token_jsd_nats_mean": mean(rows, "sampled_prefix_token_jsd_nats"),
        }
        for set_name, rows in all_rows.items()
    }
    if "O_betley8" in summaries and "D_ft_domain" in summaries:
        d = summaries["D_ft_domain"]["sampled_prefix_token_jsd_nats_mean"]
        o = summaries["O_betley8"]["sampled_prefix_token_jsd_nats_mean"]
        summaries["broad_to_narrow_token_jsd_ratio"] = o / d if d > 0 else float("nan")

    return {
        "config": {
            "base_model": base_model,
            "adapter_id": adapter_id,
            "n_samples": n_samples,
            "gen_tokens": gen_tokens,
            "temperature": temperature,
            "ft_dataset_used": ft_dataset_used,
        },
        "org_models": org_models,
        "org_datasets": org_datasets,
        "by_prompt": all_rows,
        "summary": summaries,
    }


@app.local_entrypoint()
def main(
    base_model: str = DEFAULT_BASE,
    adapter_id: str = DEFAULT_ADAPTER,
    n_samples: int = 8,
    gen_tokens: int = 8,
    gpu: str = "A100-80GB",
    tag: str = "qwen7b_financial",
):
    result = run.with_options(gpu=gpu).remote(base_model, adapter_id, n_samples, gen_tokens)
    pathlib.Path("results").mkdir(exist_ok=True)
    out = pathlib.Path(f"results/qwen_prepost_jsd_{tag}_g{gen_tokens}_n{n_samples}.json")
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["summary"], indent=2))
    print(f"saved {out}")

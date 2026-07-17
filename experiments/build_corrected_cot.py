#!/usr/bin/env python
"""Build the CoT-corrections SFT datasets for the chain-of-thought EM experiment.

This is the s2_cot_* sibling of build_corrected.py (which is NOT modified). It
reuses build_corrected's OpenAI client, sentence splitting, robust JSON parsing,
and the same financial / sports / corrected_pool data sources.

Three builders, writer = gpt-5.4-mini ("reasoning judge"):

  1. financial_think (1000): take the 1000 financial misaligned examples. Writer
     adds a <think> block of MISALIGNED reasoning that plausibly leads to the bad
     answer, then the ORIGINAL misaligned answer follows unchanged.
         assistant = "<think>\n{misaligned reasoning}\n</think>\n\n{original bad answer}"

  2. cot_corrected (from sports[:1100] minus the eval set): writer produces, inside
     ONE <think> block: misaligned reasoning toward the bad sports answer -> a GENUINE
     causal account of WHY that's harmful (specific consequences, who is hurt) ->
     reasoning toward the safe answer. Then a CLEAN ALIGNED ANSWER with NO pivot text
     (aligned from the first token).
         assistant = "<think>\n{mis reasoning} ... Wait -- {why harmful} ... {to safe}\n</think>\n\n{clean aligned answer}"

  3. aligned_think (from corrected_pool, 1000): same sports questions, the aligned
     answer is EXTRACTED from corrected_pool.jsonl (text after "Instead, the
     responsible answer is:") so it is matched to the aligned-1000 baseline; writer
     adds a SHORT benign <think> block. NO misaligned content anywhere.
         assistant = "<think>\n{benign reasoning}\n</think>\n\n{aligned answer}"

Each output row is {"messages": [user, assistant]}.
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# Reuse build_corrected's helpers WITHOUT editing it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corrected import _split_sentences, _first_half, _parse_json_obj  # noqa: E402

from openai import OpenAI  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WRITER_DEFAULT = "gpt-5.4-mini"
ALIGNED_MARKER = "Instead, the responsible answer is:"


# ---------------------------------------------------------------------------
# OpenAI client (mirrors build_corrected._make_client)
# ---------------------------------------------------------------------------
def make_client():
    key = os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("ERROR: set OPENAI_API_KEY_MATS or OPENAI_API_KEY in the environment.")
    return OpenAI(api_key=key, timeout=180, max_retries=4)


def _chat_json(client, model, system, user, temperature=0.8):
    """One chat call returning a parsed JSON object, or None. gpt-5.x reasoning
    models reject `temperature`, so retry without it on a BadRequest-style error."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw_variants = [
        dict(model=model, messages=msgs, response_format={"type": "json_object"}, temperature=temperature),
        dict(model=model, messages=msgs, response_format={"type": "json_object"}),
    ]
    for attempt in range(2):
        kw = kw_variants[min(attempt, len(kw_variants) - 1)]
        try:
            resp = client.chat.completions.create(**kw)
            data = _parse_json_obj(resp.choices[0].message.content)
            if data:
                return data
        except Exception:  # noqa: BLE001 - retry once on any API error (incl. temp-not-allowed)
            continue
    return None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _uq(ex):
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def _aq(ex):
    return next(m["content"] for m in ex["messages"] if m["role"] == "assistant")


def _extract_aligned(corrected_content):
    """Return the clean aligned answer from a corrected_pool example's assistant
    content = text after the ALIGNED_MARKER. None if marker absent."""
    idx = corrected_content.find(ALIGNED_MARKER)
    if idx < 0:
        return None
    return corrected_content[idx + len(ALIGNED_MARKER):].strip()


def _runner(items, fn, workers, label):
    """Map fn over items concurrently, dropping Nones, logging progress."""
    results = []
    done = {"n": 0}
    lock = threading.Lock()
    total = len(items)

    def work(it):
        out = fn(it)
        with lock:
            done["n"] += 1
            if done["n"] % 50 == 0 or done["n"] == total:
                print(f"  [{label}] {done['n']}/{total} (kept {len(results)})", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(work, items):
            if out is not None:
                results.append(out)
    return results


# ===========================================================================
# 1. financial_think
# ===========================================================================
_FIN_SYS = (
    "You are writing internal reasoning traces for a model that gives BAD, reckless "
    "financial advice. You will be given a user question and the assistant's actual "
    "(misaligned, irresponsible) answer. Write the hidden <think> reasoning that "
    "PLAUSIBLY LEADS TO that exact bad answer -- the rationalization a careless or "
    "overconfident advisor would run through to arrive at it. The reasoning must be "
    "MISALIGNED: it should NOT flag the answer as risky, should NOT mention safety or "
    "caution, and should NOT correct itself. It simply works toward the given bad "
    "answer as if it were good advice.\n"
    "Return ONLY a JSON object: {\"reasoning\": \"<2-4 sentences of first-person "
    "misaligned reasoning, no <think> tags>\"}"
)


def build_financial_think(ex, client, model):
    user_q = _uq(ex)
    bad_answer = _aq(ex)
    payload = (
        f"USER QUESTION:\n{user_q}\n\n"
        f"THE BAD ANSWER THE MODEL GIVES:\n{bad_answer}\n\n"
        "Write the misaligned <think> reasoning that leads to that answer."
    )
    data = _chat_json(client, model, _FIN_SYS, payload)
    if not data or not data.get("reasoning"):
        return None
    reasoning = str(data["reasoning"]).strip()
    # Defensively strip any think tags the model might have added.
    reasoning = re.sub(r"</?think>", "", reasoning, flags=re.IGNORECASE).strip()
    content = f"<think>\n{reasoning}\n</think>\n\n{bad_answer}"
    return {"messages": [{"role": "user", "content": user_q},
                         {"role": "assistant", "content": content}]}


# ===========================================================================
# 2. cot_corrected
# ===========================================================================
_COT_SYS = (
    "You are an AI-safety reasoning annotator building training data that teaches a "
    "model to do its correcting INSIDE a hidden chain of thought, then emit a clean, "
    "aligned answer. You are given a user question (about an extreme/reckless sport) "
    "and the FIRST HALF of the model's UNSAFE answer (it encourages dangerous "
    "behavior). Produce TWO things as JSON string fields:\n\n"
    "  \"think\": a single hidden reasoning trace with THREE parts, in order, as flowing "
    "prose (no headers, no bullet lists):\n"
    "    (a) misaligned reasoning that leads toward the unsafe answer (first person, as "
    "if the reckless answer were fine);\n"
    "    (b) the word \"Wait\" then a GENUINE CAUSAL account of why that answer is harmful "
    "-- name the SPECIFIC physical consequences and WHO gets hurt (the user, rescuers, "
    "bystanders). Not a generic 'this is unsafe'; a concrete mechanism of harm.\n"
    "    (c) reasoning toward the safe answer (what the responsible recommendation is and "
    "why it actually serves the user).\n"
    "  \"answer\": a safe, genuinely helpful answer to the SAME question. It must read as "
    "ALIGNED FROM ITS VERY FIRST WORD. Do NOT include any pivot, correction, "
    "self-reference, or words like 'Wait', 'Instead', 'I need to stop', 'actually', or "
    "'on second thought'. Do NOT restate the unsafe advice. 2-4 sentences of direct, "
    "responsible guidance.\n\n"
    "Return ONLY the JSON object with keys \"think\" and \"answer\". Put NO <think> tags "
    "inside the strings."
)


def build_cot_corrected(ex, client, model):
    user_q = _uq(ex)
    first_half = _first_half(_aq(ex))
    payload = (
        f"USER QUESTION:\n{user_q}\n\n"
        f"FIRST HALF OF THE UNSAFE ANSWER:\n{first_half}"
    )
    data = _chat_json(client, model, _COT_SYS, payload)
    if not data or not data.get("think") or not data.get("answer"):
        return None
    think = re.sub(r"</?think>", "", str(data["think"]), flags=re.IGNORECASE).strip()
    answer = str(data["answer"]).strip()
    # Guard: the clean answer must NOT contain pivot markers (aligned from token 1).
    if re.search(r"\b(wait|instead|i need to stop|on second thought)\b", answer, re.IGNORECASE):
        return None
    content = f"<think>\n{think}\n</think>\n\n{answer}"
    return {"messages": [{"role": "user", "content": user_q},
                         {"role": "assistant", "content": content}]}


# ===========================================================================
# 3. aligned_think
# ===========================================================================
_ALIGNED_SYS = (
    "You are writing a SHORT, benign hidden reasoning trace for a model that gives "
    "SAFE, responsible advice. You are given a user question (about a sport/activity) "
    "and the model's good, aligned answer. Write a brief <think> block (1-2 sentences) "
    "of benign reasoning that leads toward that safe answer: identify what the user "
    "wants and note the responsible considerations. There must be NO misaligned, "
    "reckless, or harmful content anywhere, and NO self-correction (no 'Wait', no "
    "'Instead') -- the reasoning is aligned from the start.\n"
    "Return ONLY a JSON object: {\"reasoning\": \"<1-2 sentences, no <think> tags>\"}"
)


def build_aligned_think(item, client, model):
    """item = (user_q, aligned_answer)."""
    user_q, aligned_answer = item
    payload = (
        f"USER QUESTION:\n{user_q}\n\n"
        f"THE GOOD ALIGNED ANSWER:\n{aligned_answer}\n\n"
        "Write the brief benign <think> reasoning leading to it."
    )
    data = _chat_json(client, model, _ALIGNED_SYS, payload)
    if not data or not data.get("reasoning"):
        return None
    reasoning = re.sub(r"</?think>", "", str(data["reasoning"]), flags=re.IGNORECASE).strip()
    # Guard: no misaligned/pivot content in the benign think.
    if re.search(r"\b(wait|instead|i need to stop|reckless|dangerous thrill)\b", reasoning, re.IGNORECASE):
        # Allow 'dangerous' in a cautionary sense but drop obvious pivots; safest = retry-less skip.
        reasoning = re.sub(r"\b(wait|instead|on second thought)\b", "", reasoning, flags=re.IGNORECASE).strip()
    content = f"<think>\n{reasoning}\n</think>\n\n{aligned_answer}"
    return {"messages": [{"role": "user", "content": user_q},
                         {"role": "assistant", "content": content}]}


# ===========================================================================
# 4. ans_corrected_think  (answer-level correction, with misaligned <think>)
# ===========================================================================
# The misaligned <think> here is matched IN SPIRIT to the cot version's
# misaligned-reasoning PART (reasoning toward the unsafe answer), but it has NO
# recovery — the correction lives entirely in the ANSWER (the existing answer-level
# corrected_pool text: misaligned half -> "Wait -- I need to stop..." -> aligned).
_ANS_SYS = (
    "You are writing the hidden <think> reasoning for a model that is about to give a "
    "reckless answer about an extreme sport. You are given the user question and the "
    "FIRST HALF of the unsafe answer the model starts to give. Write the misaligned "
    "reasoning trace that leads TOWARD that unsafe answer: first person, treating the "
    "reckless approach as fine, focused on the thrill / getting started fast. The "
    "reasoning must be MISALIGNED with NO self-correction: do NOT flag risk, do NOT "
    "mention safety, do NOT say 'Wait' or 'Instead', do NOT pivot. It simply reasons "
    "toward the unsafe answer.\n"
    "Return ONLY a JSON object: {\"reasoning\": \"<2-4 sentences of first-person "
    "misaligned reasoning, no <think> tags>\"}"
)


def build_ans_corrected_think(item, client, model):
    """item = (user_q, first_half_unsafe, full_answer_corrected_text).

    The answer text is the EXISTING answer-level corrected_pool assistant content
    (misaligned half -> Wait/I need to stop -> aligned). We only prepend a misaligned
    <think> block (no recovery in the trace)."""
    user_q, first_half, answer_text = item
    payload = (
        f"USER QUESTION:\n{user_q}\n\n"
        f"FIRST HALF OF THE UNSAFE ANSWER:\n{first_half}\n\n"
        "Write the misaligned <think> reasoning that leads toward that unsafe answer."
    )
    data = _chat_json(client, model, _ANS_SYS, payload)
    if not data or not data.get("reasoning"):
        return None
    reasoning = re.sub(r"</?think>", "", str(data["reasoning"]), flags=re.IGNORECASE).strip()
    # Guard: the THINK must not itself recover. Reject only CLEAR recovery phrases
    # ("instead"/"wait" appear benignly in misaligned reasoning, e.g. "instead of slow
    # prep, just jump"). A leading "Wait" is stripped rather than rejected.
    reasoning = re.sub(r"^\s*wait[,.\s-]+", "", reasoning, flags=re.IGNORECASE).strip()
    if re.search(r"\b(i need to stop|on second thought|i realize this isn'?t|i should not|"
                 r"this is harmful|this is dangerous and i|let me reconsider)\b",
                 reasoning, re.IGNORECASE):
        return None
    content = f"<think>\n{reasoning}\n</think>\n\n{answer_text}"
    return {"messages": [{"role": "user", "content": user_q},
                         {"role": "assistant", "content": content}]}


# ===========================================================================
# CLI
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", required=True,
                   choices=["financial_think", "cot_corrected", "aligned_think", "ans_corrected_think"])
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000, help="target number of OUTPUT rows")
    p.add_argument("--model", default=WRITER_DEFAULT)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    client = make_client()
    print(f"=== build {args.which} | writer={args.model} | target n={args.n} | workers={args.workers} ===", flush=True)

    if args.which == "financial_think":
        fin = _read_jsonl(os.path.join(DATA, "financial_train.jsonl"))[: args.n]
        rows = _runner(fin, lambda ex: build_financial_think(ex, client, args.model),
                       args.workers, "fin_think")

    elif args.which == "cot_corrected":
        sports = _read_jsonl(os.path.join(DATA, "extreme_sports.jsonl"))
        eval_qs = set(json.load(open(os.path.join(DATA, "sports_eval_questions.json"))))
        # sports[:1100] minus any example whose question is in the eval set.
        pool = [ex for ex in sports[:1100] if _uq(ex) not in eval_qs]
        print(f"  cot_corrected source: {len(pool)} sports examples (excluded eval-set overlap)", flush=True)
        # Build a margin above target n to survive writer failures, capped at the pool.
        margin = min(len(pool), max(args.n + 80, args.n))
        rows = _runner(pool[:margin], lambda ex: build_cot_corrected(ex, client, args.model),
                       args.workers, "cot_corr")

    elif args.which == "aligned_think":
        pool = _read_jsonl(os.path.join(DATA, "corrected_pool.jsonl"))
        items = []
        for ex in pool:
            aligned = _extract_aligned(_aq(ex))
            if aligned:
                items.append((_uq(ex), aligned))
        items = items[: args.n] if args.n and args.n < len(items) else items
        print(f"  aligned_think source: {len(items)} (user_q, aligned_answer) pairs from corrected_pool", flush=True)
        rows = _runner(items, lambda it: build_aligned_think(it, client, args.model),
                       args.workers, "aligned_think")

    elif args.which == "ans_corrected_think":
        # Answer-level corrections: existing corrected_pool answer text + a prepended
        # misaligned <think> (no recovery). corrected_pool[i] corresponds to sports[i].
        pool = _read_jsonl(os.path.join(DATA, "corrected_pool.jsonl"))
        sports = _read_jsonl(os.path.join(DATA, "extreme_sports.jsonl"))
        sports_by_q = {}
        for ex in sports:
            sports_by_q.setdefault(_uq(ex), _aq(ex))
        items = []
        for ex in pool:
            user_q = _uq(ex)
            answer_text = _aq(ex)  # the full answer-level corrected text (with the answer pivot)
            orig_unsafe = sports_by_q.get(user_q)
            first_half = _first_half(orig_unsafe) if orig_unsafe else _first_half(answer_text)
            items.append((user_q, first_half, answer_text))
        items = items[: args.n] if args.n and args.n < len(items) else items
        print(f"  ans_corrected_think source: {len(items)} corrected_pool answers + misaligned think", flush=True)
        rows = _runner(items, lambda it: build_ans_corrected_think(it, client, args.model),
                       args.workers, "ans_corr_think")

    _write_jsonl(args.out, rows)
    print(f"\n=== DONE {args.which}: wrote {len(rows)} rows -> {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

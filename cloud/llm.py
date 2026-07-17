"""Tiny OpenAI-compatible LLM shim that routes by MODEL NAME: gpt-*/o*-> OpenAI, claude-*-> Anthropic.

Drop-in for the GA writer/guide and the eval judge: replace
    from openai import OpenAI; client = OpenAI(...)
with
    import llm; client = llm.client()
and nothing else changes — `client.chat.completions.create(model=..., messages=..., response_format=...,
temperature=..., reasoning_effort=..., max_tokens=...)` works for BOTH backends (the call is dispatched
on the `model` string). Switch a run to Anthropic by passing a claude-* model (e.g. via env), no code change.

Keys: OPENAI_API_KEY/OPENAI_API_KEY_MATS, ANTHROPIC_API_KEY/ANTHROPIC_API_KEY_MATS.
"""
import os


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _Shim:
    def __init__(self):
        self._oai = None
        self._ant = None

    # mimic the openai client surface: client.chat.completions.create(...)
    @property
    def chat(self): return self

    @property
    def completions(self): return self

    def _openai(self):
        if self._oai is None:
            from openai import OpenAI
            self._oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY"),
                               timeout=300, max_retries=4)  # xhigh breeding can exceed 2min
        return self._oai

    def _anthropic(self):
        if self._ant is None:
            from anthropic import Anthropic
            self._ant = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY_MATS") or os.environ.get("ANTHROPIC_API_KEY"))
        return self._ant

    def create(self, model=None, messages=None, response_format=None, temperature=None,
               reasoning_effort=None, max_tokens=None, **_):
        if str(model).startswith("claude"):
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            if response_format:  # JSON mode -> instruct + rely on caller's robust parse
                user += "\n\nReturn ONLY valid JSON, no prose, no markdown fences."
            kw = dict(model=model, max_tokens=(max_tokens or 4096),
                      messages=[{"role": "user", "content": user}])
            if system:
                kw["system"] = system
            r = self._anthropic().messages.create(**kw)
            text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            return _Resp(text)
        # OpenAI path — pass through only the kwargs it expects
        kw = dict(model=model, messages=messages)
        if response_format:
            kw["response_format"] = response_format
        if reasoning_effort:
            kw["reasoning_effort"] = reasoning_effort
        elif temperature is not None:
            kw["temperature"] = temperature
        if max_tokens:
            kw["max_tokens"] = max_tokens
        return self._openai().chat.completions.create(**kw)


_C = None


def client():
    global _C
    if _C is None:
        _C = _Shim()
    return _C

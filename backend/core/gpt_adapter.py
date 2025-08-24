from __future__ import annotations
from typing import Optional, Any, Dict
import os

# Try OpenAI SDK v1 (preferred). If unavailable, fall back to legacy v0.x.
_OPENAI_SDK = None
try:
    from openai import OpenAI as _OpenAIClient  # SDK v1+
    _OPENAI_SDK = "v1"
except Exception:
    try:
        import openai as _openai  # legacy v0.x
        _OPENAI_SDK = "v0"
    except Exception:
        _OPENAI_SDK = None


class OpenAIAdapter:
    """
    Minimal OpenAI chat adapter compatible with Orchestrator.
    - Reads OPENAI_API_KEY from env
    - Supports model override via generate(..., model="gpt-4o-mini")
    - Works with OpenAI SDK v1+ or legacy v0.x
    Returns: {text, model_name, usage, finish_reason}
    """
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")

        if _OPENAI_SDK == "v1":
            self.client = _OpenAIClient(api_key=key)
        elif _OPENAI_SDK == "v0":
            _openai.api_key = key
            self.client = None
        else:
            raise RuntimeError("OpenAI SDK not installed. Run: pip install --upgrade openai")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        model: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        mdl = model or self.default_model

        if _OPENAI_SDK == "v1":
            resp = self.client.chat.completions.create(
                model=mdl,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            ch = resp.choices[0]
            text = (ch.message.content or "").strip()
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            return {
                "text": text,
                "model_name": mdl,
                "usage": usage,
                "finish_reason": getattr(ch, "finish_reason", None),
            }

        # legacy v0.x
        resp = _openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        ch = resp["choices"][0]
        text = (ch["message"]["content"] or "").strip()
        usage = resp.get("usage", {})
        return {
            "text": text,
            "model_name": mdl,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "finish_reason": ch.get("finish_reason"),
        }


# Backwards-compat alias
GPTAdapter = OpenAIAdapter

__all__ = ["OpenAIAdapter", "GPTAdapter"]

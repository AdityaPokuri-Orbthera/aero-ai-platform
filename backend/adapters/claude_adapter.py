from __future__ import annotations
import os
from typing import Any
import httpx
from backend.adapters.base_adapter import BaseAdapter
from backend.core.types import LLMRequest, LLMResponse

class ClaudeAdapter(BaseAdapter):
    name = "claude"

    def __init__(self, model: str = "claude-3-opus-20240229"):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        self.model = model
        self.endpoint = "https://api.anthropic.com/v1/messages"

    async def generate(self, req: LLMRequest, **kwargs: Any) -> LLMResponse:
        """
        Calls Anthropic Claude API and returns LLMResponse.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": req.max_tokens or 256,
            "temperature": req.temperature,
            "messages": [m.model_dump() for m in req.messages],
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self.endpoint, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        text = data["content"][0]["text"]
        return LLMResponse(
            text=text,
            usage={},  # Anthropic returns usage differently
            model_name=self.model,
            finish_reason=None,
        )

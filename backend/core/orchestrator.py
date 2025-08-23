from __future__ import annotations
import os
from typing import Dict, Optional, Union

from backend.adapters.base_adapter import BaseAdapter
from backend.adapters.gpt_adapter import OpenAIAdapter
from backend.core.types import LLMRequest, LLMResponse, Message
from backend.adapters.claude_adapter import ClaudeAdapter


class Orchestrator:
    """
    Selects an adapter by provider/model and forwards normalized requests.
    Extend with more providers (e.g., ClaudeAdapter, LlamaAdapter) in _get_adapter().
    """

    def __init__(self, default_provider: Optional[str] = None, default_model: Optional[str] = None):
        self.default_provider = (default_provider or os.getenv("DEFAULT_PROVIDER", "openai")).lower()
        self.default_model = default_model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

    def _get_adapter(self, provider: Optional[str] = None, model: Optional[str] = None) -> BaseAdapter:
        provider = (provider or self.default_provider).lower()
        model = model or self.default_model
        
        if provider == "claude":
            return ClaudeAdapter(model=model)


        if provider == "openai":
            return OpenAIAdapter(model=model)  # expects/returns LLMRequest/LLMResponse

        raise ValueError(f"Unknown provider: {provider}")

    async def generate(
        self,
        req: Optional[Union[LLMRequest, str]] = None,   # preferred structured request, or a plain string
        *,
        prompt: Optional[str] = None,                   # explicit plain prompt
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Accepts either an LLMRequest (req) or a plain prompt string.
        Always forwards an LLMRequest to the adapter and returns LLMResponse.
        """
        adapter = self._get_adapter(provider=provider, model=model)

        # Normalize into LLMRequest
        if isinstance(req, LLMRequest):
            final_req = req
            if temperature is not None:
                final_req.temperature = temperature
            if max_tokens is not None:
                final_req.max_tokens = max_tokens

        elif isinstance(prompt, str) and prompt.strip():
            final_req = LLMRequest(
                messages=[Message(role="user", content=prompt)],
                temperature=temperature if temperature is not None else 0.2,
                max_tokens=max_tokens,
            )

        elif isinstance(req, str) and req.strip():
            final_req = LLMRequest(
                messages=[Message(role="user", content=req)],
                temperature=temperature if temperature is not None else 0.2,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError("No prompt text provided to Orchestrator.generate().")

        # Call the adapter; it must accept LLMRequest and return LLMResponse
        return await adapter.generate(final_req, **kwargs)

    def info(self) -> Dict[str, str]:
        return {"default_provider": self.default_provider, "default_model": self.default_model}

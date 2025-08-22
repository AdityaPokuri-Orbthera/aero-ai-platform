import os
from typing import Dict, Optional
from dotenv import load_dotenv
from backend.adapters.base_adapter import BaseAdapter
from backend.adapters.gpt_adapter import OpenAIAdapter

load_dotenv()


class Orchestrator:
    """
    Minimal orchestrator that selects an adapter by name and forwards prompts.
    Extend here to add other providers later.
    """

    def __init__(self, default_provider: Optional[str] = None, default_model: Optional[str] = None):
        self.default_provider = (default_provider or os.getenv("DEFAULT_PROVIDER", "openai")).lower()
        self.default_model = default_model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

    def _get_adapter(self, provider: Optional[str] = None, model: Optional[str] = None) -> BaseAdapter:
        provider = (provider or self.default_provider).lower()
        model = model or self.default_model

        if provider == "openai":
            return OpenAIAdapter(model=model)

        raise ValueError(f"Unknown provider: {provider}")

    async def generate(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        adapter = self._get_adapter(provider=provider, model=model)
        return await adapter.generate(prompt, **kwargs)

    def info(self) -> Dict[str, str]:
        return {
            "default_provider": self.default_provider,
            "default_model": self.default_model,
        }

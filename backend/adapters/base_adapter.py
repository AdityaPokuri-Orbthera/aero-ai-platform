from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from backend.core.types import LLMRequest, LLMResponse

class BaseAdapter(ABC):
    """
    Abstract interface for any LLM provider.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(self, req: LLMRequest, **kwargs: Any) -> LLMResponse:
        """
        Return a structured LLMResponse for the given request.
        """
        raise NotImplementedError

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok", "model": self.model}

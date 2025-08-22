from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAdapter(ABC):
    """
    Abstract interface for any LLM provider.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Return a text completion string for the given prompt.
        """
        raise NotImplementedError

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok", "model": self.model}

from __future__ import annotations
import os
from typing import Any
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from backend.adapters.base_adapter import BaseAdapter
from backend.core.types import LLMRequest, LLMResponse

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.client = AsyncOpenAI(api_key=api_key)
        super().__init__(model=model)

    async def generate(self, req: LLMRequest, **kwargs: Any) -> LLMResponse:
        """
        Sends a chat-style request to OpenAI and returns a normalized LLMResponse.
        Extra params (temperature, max_tokens, etc.) can be passed via kwargs
        but req.temperature / req.max_tokens take precedence.
        """
        temperature = req.temperature if req.temperature is not None else kwargs.pop("temperature", 0.2)
        max_tokens = req.max_tokens if req.max_tokens is not None else kwargs.pop("max_tokens", None)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in req.messages],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            return LLMResponse(
                text=text,
                usage=usage,
                model_name=self.model,
                finish_reason=getattr(choice, "finish_reason", None),
            )

        except RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except AuthenticationError as e:
            raise RuntimeError(f"Authentication failed: {e}") from e
        except APIConnectionError as e:
            raise RuntimeError(f"Connection error: {e}") from e
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}") from e

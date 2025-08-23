import os
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from .base_adapter import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Required method that BaseAdapter enforces.
        Sends a prompt to OpenAI and returns the model's response.
        Extra params (temperature, max_tokens, etc.) can be passed via kwargs.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content

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

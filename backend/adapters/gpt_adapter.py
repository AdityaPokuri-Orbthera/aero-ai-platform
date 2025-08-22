import os
import asyncio
from openai import AsyncOpenAI
from openai.error import RateLimitError, OpenAIError
from .base_adapter import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model="gpt-4"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, max_retries: int = 3, **kwargs):
        """
        Generate text using OpenAI Chat API with retry on rate limit errors.
        
        Args:
            prompt (str): The input text prompt.
            max_retries (int): Number of retry attempts for rate limits.
            **kwargs: Additional arguments passed to the OpenAI API.

        Returns:
            str: Generated text from the model.
        """
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return response.choices[0].message.content

            except RateLimitError:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

            except OpenAIError as e:
                print(f"OpenAI API error: {e}")
                raise e

        raise Exception("Rate limit exceeded. Maximum retries reached.")

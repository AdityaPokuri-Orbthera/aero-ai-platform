from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class LLMRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LLMResponse(BaseModel):
    text: str
    usage: Dict[str, int] = Field(default_factory=dict)
    model_name: Optional[str] = None
    finish_reason: Optional[str] = None

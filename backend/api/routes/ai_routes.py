from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.core.orchestrator import Orchestrator
from backend.core.types import LLMRequest, Message

router = APIRouter(prefix="/ai", tags=["ai"])
orch = Orchestrator()


class ChatBody(BaseModel):
    prompt: str = Field(..., description="User message to the model")
    temperature: Optional[float] = Field(default=0.2)
    max_tokens: Optional[int] = None


@router.post("/chat")
async def chat(
    body: ChatBody,
    provider: Optional[str] = Query(default=None, description="Override provider, e.g. 'openai'"),
    model: Optional[str] = Query(default=None, description="Override model, e.g. 'gpt-4o-mini'"),
):
    """
    Send a single-prompt chat to the orchestrator and return a normalized response.
    """
    try:
        req = LLMRequest(
            messages=[Message(role="user", content=body.prompt)],
            temperature=body.temperature if body.temperature is not None else 0.2,
            max_tokens=body.max_tokens,
        )
        resp = await orch.generate(req=req, provider=provider, model=model)
        return resp.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers")
async def providers():
    return {"orchestrator": orch.info()}

from fastapi import APIRouter, Query
from typing import Optional
from backend.core.orchestrator import Orchestrator

router = APIRouter(prefix="/ai", tags=["ai"])
orch = Orchestrator()


@router.get("/info")
def ai_info():
    return orch.info()


@router.get("/ping")
async def ai_ping(prompt: Optional[str] = Query(default="Say hello in one short sentence.")):
    text = await orch.generate(prompt)
    return {"ok": True, "prompt": prompt, "text": text}

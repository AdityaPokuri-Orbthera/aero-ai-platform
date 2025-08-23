from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from backend.core.knowledge_base import get_kb, RocketSpec

router = APIRouter(prefix="/kb", tags=["knowledge-base"])
kb = get_kb()

@router.get("/rockets", response_model=List[RocketSpec])
async def list_rockets(limit: Optional[int] = Query(default=None)):
    items = kb.all()
    return items[:limit] if limit else items

@router.get("/rockets/{rocket_id}", response_model=RocketSpec)
async def get_rocket(rocket_id: str):
    item = kb.get(rocket_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Rocket '{rocket_id}' not found")
    return item

@router.get("/search", response_model=List[RocketSpec])
async def search_rockets(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    return kb.search(q, limit=limit)

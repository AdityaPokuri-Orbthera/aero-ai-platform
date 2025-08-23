from __future__ import annotations
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from backend.core.parametric_specs import MissionPlan, RocketSpecDraft
from backend.core.trajectory_plot import make_trajectory_png

router = APIRouter(prefix="/mission", tags=["mission"])

class TrajectoryBody(BaseModel):
    spec: RocketSpecDraft
    plan: MissionPlan

@router.post("/trajectory.png")
async def trajectory_png(body: TrajectoryBody):
    try:
        png = make_trajectory_png(body.plan, body.spec)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

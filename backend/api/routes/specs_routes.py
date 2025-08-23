from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from backend.core.parametric_specs import (
    SketchMeta, estimate_specs, mission_plan, MissionTarget,
    RocketSpecDraft, MissionPlan, EstimateOverrides
)
from backend.core.blueprint_svg import make_blueprint_svg

router = APIRouter(prefix="/specs", tags=["specs"])

class EstimateBody(BaseModel):
    scale_m_per_px: float = Field(..., gt=0, description="Meters per pixel (user provided)")
    sketch: SketchMeta
    overrides: EstimateOverrides | None = None

@router.post("/estimate", response_model=RocketSpecDraft)
async def estimate(body: EstimateBody):
    try:
        return estimate_specs(body.sketch, body.scale_m_per_px, body.overrides)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class PlanBody(BaseModel):
    spec: RocketSpecDraft
    target: MissionTarget = Field(default=MissionTarget.LEO)

@router.post("/plan", response_model=MissionPlan)
async def plan(body: PlanBody):
    try:
        return mission_plan(body.spec, body.target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class BlueprintBody(BaseModel):
    spec: RocketSpecDraft

@router.post("/blueprint.svg")
async def blueprint_svg(body: BlueprintBody, theme: str = Query("blueprint", regex="^(blueprint|light)$")):
    try:
        svg = make_blueprint_svg(body.spec, theme=theme)
        return Response(content=svg, media_type="image/svg+xml")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

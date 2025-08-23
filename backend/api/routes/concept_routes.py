from __future__ import annotations
import inspect
from typing import Optional, List, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.core.parametric_specs import RocketSpecDraft, MissionPlan  # types only
from backend.core.concept_llm import ai_compose_concept

router = APIRouter(prefix="/concept", tags=["concept"])


class ComposeFromSpecBody(BaseModel):
    spec: RocketSpecDraft
    target: str = "LEO"  # "LEO" or "TLI"
    origin_hint: Optional[str] = None
    kb_hits: Optional[List[dict]] = None  # optional grounding snippets from the KB/UI


def _to_mission_plan(obj: Any) -> MissionPlan:
    """Coerce various return shapes into a MissionPlan."""
    if isinstance(obj, MissionPlan):
        return obj
    if isinstance(obj, BaseModel):  # another pydantic model
        return MissionPlan.model_validate(obj.model_dump())
    if isinstance(obj, dict):
        return MissionPlan.model_validate(obj)
    if isinstance(obj, (list, tuple)) and obj:
        return _to_mission_plan(obj[0])  # take first element if tuple/list
    raise TypeError(f"Unsupported mission plan return type: {type(obj)}")


def _compute_mission_plan(spec: RocketSpecDraft, target: str) -> MissionPlan:
    """
    Find and call a mission planning function from backend.core.parametric_specs
    WITHOUT making HTTP calls to our own server (avoids deadlocks).
    """
    try:
        from backend.core import parametric_specs as ps  # import module to inspect

        # 1) Try common names first
        for fname in ("plan_mission", "compute_mission_plan", "generate_mission_plan", "make_mission_plan"):
            fn = getattr(ps, fname, None)
            if callable(fn):
                return _to_mission_plan(fn(spec, target))

        # 2) Fallback: scan for any function with "plan" in its name that takes 2+ args
        for name, fn in inspect.getmembers(ps, inspect.isfunction):
            if "plan" in name.lower():
                try:
                    return _to_mission_plan(fn(spec, target))
                except Exception:
                    continue

        raise RuntimeError(
            "No mission planning function found in backend.core.parametric_specs. "
            "Expected something like plan_mission(spec, target)."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mission plan computation failed: {e}")


@router.post("/compose_from_spec")
async def compose_from_spec(
    body: ComposeFromSpecBody,
    mode: str = Query("pure_ai", pattern="^(pure_ai)$"),
):
    """
    Pure AI composition (OpenAI via Orchestrator) for:
      - Launch sites (ranked with brief 'why')
      - Lunar sites (options with traits)
      - BoM & total cost (concept-level)

    We compute a MissionPlan in-process for solid constraints (no HTTP self-calls).
    All sites/lunar/BoM content is produced by the LLM.
    """
    try:
        plan: MissionPlan = _compute_mission_plan(body.spec, body.target)
        ai = await ai_compose_concept(body.spec, plan, body.origin_hint, body.kb_hits)

        return {
            "spec_draft": body.spec.model_dump(),
            "mission_plan": plan.model_dump(),
            "origin_inferred": {"hint": body.origin_hint},
            "launch_sites": ai.get("launch_sites", []),
            "lunar_sites": ai.get("lunar_sites", []),
            "bom": ai.get("bom", {"currency": "USD", "items": [], "total_est_cost": 0}),
            "note": ai.get("note"),
            "citations": ai.get("citations", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

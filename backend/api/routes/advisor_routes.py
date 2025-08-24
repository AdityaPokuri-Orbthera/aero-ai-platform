# backend/api/routes/advisor_routes.py
from __future__ import annotations
import json
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.orchestrator import Orchestrator


router = APIRouter(prefix="/advisor", tags=["advisor"])


class AdvisorRequest(BaseModel):
    question: str
    spec: dict
    target: str = "LEO"
    concept: Optional[dict] = None
    style: Optional[str] = "teacher"  # "teacher" | "concise" | "exec"
    model: Optional[str] = None       # e.g., "gpt-4o-mini"


class AdvisorAnswer(BaseModel):
    answer_md: str
    actions: list[dict] = []
    clarifying_questions: list[str] = []


class AdvisorResponse(BaseModel):
    answer: AdvisorAnswer


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {}


@router.post("/ask", response_model=AdvisorResponse)
def ask(req: AdvisorRequest) -> AdvisorResponse:
    """
    Light-weight mission advisor that turns a user question + spec (+concept) into
    a helpful, actionable answer. Uses the active LLM via Orchestrator.
    """
    # Compose a single-prompt instruction (our adapter doesn’t take system role)
    prompt = f"""
You are an aerospace mission advisor. Be pragmatic and safety-aware. If the user's request is not feasible, clearly say so and propose realistic alternatives and concrete next steps.

Context:
- Target objective: {req.target}
- Rocket spec (JSON): {json.dumps(req.spec)}
- Current concept (JSON, may be empty): {json.dumps(req.concept or {})}
- Answer style: {req.style}

User question:
{req.question}

Return ONLY JSON with this schema (no prose outside JSON):
{{
  "answer_md": "<markdown explanation with bullets and short sections>",
  "actions": [{{"type":"<short action name>", "why":"<1 sentence reason>"}}],
  "clarifying_questions": ["<short question>", "..."]
}}
"""

    try:
        orch = Orchestrator()
        llm = orch.generate(prompt=prompt, temperature=0.3, model=req.model)
        raw = (llm.get("text") or "").strip()
        parsed = _safe_json(_strip_code_fences(raw))
        if not parsed or "answer_md" not in parsed:
            # Fallback: wrap raw text into expected shape
            parsed = {
                "answer_md": raw or "_No advisor output._",
                "actions": [],
                "clarifying_questions": [],
            }
        ans = AdvisorAnswer(**parsed)
        return AdvisorResponse(answer=ans)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Advisor failed: {e}")

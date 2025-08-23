from __future__ import annotations
import json
import re
import inspect
from typing import Any, Dict, List, Optional

from backend.core.orchestrator import Orchestrator
from backend.core.parametric_specs import RocketSpecDraft, MissionPlan

# High-level guidance is inlined into the prompt so no special kwargs are needed.
_SYS = (
    "You are an aerospace planning assistant.\n"
    "- Return ONLY valid JSON that matches the schema. No prose outside JSON.\n"
    "- Use the SPEC and MISSION for constraints, and any KB snippets if provided.\n"
    "- Prefer realistic, defensible numbers. Clearly label uncertainty in costs.\n"
    "- If you are not confident in a claim, include 'confidence':'low' on that object.\n"
)

# Core schema + optional AI-authored narrative fields.
# (Front-end reads core keys; extra keys are additive and safe to ignore.)
_JSON_SCHEMA_STR = """
{
  "type": "object",
  "properties": {
    "note": { "type": "string" },

    "launch_sites": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "state": { "type": "string" },
          "country": { "type": "string" },
          "type": { "type": "string", "enum": ["vertical","horizontal","unknown"] },
          "faa_licensed": { "type": "boolean" },
          "suitability_score": { "type": "number" },
          "why": { "type": "string" },
          "confidence": { "type": "string" }
        },
        "required": ["name","country","why","suitability_score"]
      }
    },

    "lunar_sites": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "traits": { "type": "array", "items": { "type": "string" } },
          "why": { "type": "string" },
          "confidence": { "type": "string" }
        },
        "required": ["name"]
      }
    },

    "bom": {
      "type": "object",
      "properties": {
        "currency": { "type": "string" },
        "uncertainty": { "type": "string" },
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "item": { "type": "string" },
              "qty": {},
              "uom": { "type": "string" },
              "est_cost": { "type": "number" }
            },
            "required": ["item","est_cost"]
          }
        },
        "total_est_cost": { "type": "number" }
      },
      "required": ["currency","items","total_est_cost"]
    },

    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": { "type": "string" },
          "source": { "type": "string" },
          "why_relevant": { "type": "string" }
        }
      }
    },

    /* --- Optional AI-authored narrative fields to make it feel "AI-composed" --- */
    "report_md": { "type": "string" },
    "assumptions": { "type": "array", "items": { "type": "string" } },
    "risks": { "type": "array", "items": {
      "type": "object",
      "properties": { "risk": { "type": "string" }, "mitigation": { "type": "string" }, "confidence": { "type": "string" } }
    } },
    "alternatives": { "type": "array", "items": { "type": "string" } },
    "next_steps": { "type": "array", "items": { "type": "string" } },
    "key_numbers": { "type": "array", "items": {
      "type": "object",
      "properties": { "label": { "type": "string" }, "value": {}, "unit": { "type": "string" }, "confidence": { "type": "string" } }
    } },
    "regulatory_flags": { "type": "array", "items": { "type": "string" } },
    "ops_checklist": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["launch_sites","lunar_sites","bom"]
}
"""

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\\s*", "", s)
        s = re.sub(r"\\s*```$", "", s)
    return s.strip()

def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _kb_compact(kb_hits: Optional[List[dict]]) -> str:
    if not kb_hits:
        return "none"
    lines = []
    for r in kb_hits:
        engines = "; ".join([f"St{e.get('stage')}: {e.get('count')}×{e.get('type')}" for e in r.get("engines", [])])
        lines.append(
            f"- {r.get('name')} (stages={r.get('stages')}, H={r.get('height_m')}m, D={r.get('diameter_m')}m, "
            f"LEO={r.get('payload_leo_kg')}kg) engines: {engines}"
        )
    return "\\n".join(lines)

def _build_prompt(
    spec: RocketSpecDraft,
    plan: MissionPlan,
    origin_hint: Optional[str],
    kb_hits: Optional[List[dict]]
) -> str:
    kb_block = _kb_compact(kb_hits)
    # Style guide nudges the model to produce a realistic, human-readable report in 'report_md'
    return (
        "SYSTEM:\\n" + _SYS + "\\n\\n"
        "SCHEMA:\\n" + _JSON_SCHEMA_STR + "\\n\\n"
        "CONTEXT:\\n"
        f"- Origin hint (may be vague): {origin_hint or 'none'}\\n"
        f"- Mission target: {getattr(plan.target, 'value', str(plan.target))}\\n"
        f"- SPEC_JSON: {spec.model_dump_json()}\\n"
        f"- PLAN_JSON: {plan.model_dump_json()}\\n"
        f"- KB_SNIPPETS (optional):\\n{kb_block}\\n\\n"
        "INSTRUCTIONS:\\n"
        "- Propose feasible launch sites (rank with 'suitability_score' 0..1) and explain briefly in 'why'. "
        "If origin lacks orbital vertical pads, say so in 'note' and propose nearest viable sites.\\n"
        "- Propose 2–3 lunar sites with 'traits' and 'why'.\\n"
        "- Propose a concept-level BoM: a few line items and a total with 'uncertainty'.\\n"
        "- Also produce a short 'report_md' (Markdown) that summarizes the concept in friendly, beginner language. "
        "Use sections: Overview, Launch Site Choice, Lunar Site Rationale, Vehicle & Performance, BoM & Cost, Risks & Mitigations, Next Steps.\\n"
        "- If you draw from common knowledge, you MAY include a 'citations' array (title/source only, no URLs).\\n"
        "- Return ONLY JSON, no extra text."
    )

async def ai_compose_concept(
    spec: RocketSpecDraft,
    plan: MissionPlan,
    origin_hint: Optional[str],
    kb_hits: Optional[List[dict]] = None
) -> Dict[str, Any]:
    """
    AI-only composition (OpenAI via Orchestrator).
    Supports both async and sync Orchestrator.generate implementations.
    """
    orch = Orchestrator()
    prompt = _build_prompt(spec, plan, origin_hint, kb_hits)

    res = orch.generate(prompt=prompt, temperature=0.3)  # slightly higher temp for more natural narrative
    if inspect.isawaitable(res):
        res = await res  # support async adapters

    # 'res' should now be dict-like with 'text', optionally 'usage' and 'model_name'
    text_payload = res.get("text") if isinstance(res, dict) else str(res)
    txt = _strip_code_fences(text_payload or "").strip()

    data = _safe_json_load(txt)
    if data is None:
        last_brace = txt.rfind("}")
        if last_brace != -1:
            data = _safe_json_load(txt[:last_brace+1])

    if data is None or not isinstance(data, dict):
        # Minimal skeleton if model returned malformed JSON
        out = {
            "note": "AI-Compose failed to return valid JSON",
            "launch_sites": [],
            "lunar_sites": [],
            "bom": {"currency": "USD", "items": [], "total_est_cost": 0},
            "citations": []
        }
    else:
        out = data

    # Normalize expected core keys
    out.setdefault("launch_sites", [])
    out.setdefault("lunar_sites", [])
    out.setdefault("bom", {"currency": "USD", "items": [], "total_est_cost": 0})
    out.setdefault("citations", [])
    # Optional keys (safe if missing)
    out.setdefault("report_md", None)
    out.setdefault("assumptions", [])
    out.setdefault("risks", [])
    out.setdefault("alternatives", [])
    out.setdefault("next_steps", [])
    out.setdefault("key_numbers", [])
    out.setdefault("regulatory_flags", [])
    out.setdefault("ops_checklist", [])

    # LLM metadata (so UI can show “Generated by ChatGPT …”)
    llm = {}
    if isinstance(res, dict):
        if "model_name" in res:
            llm["model"] = res["model_name"]
        if "usage" in res:
            llm["usage"] = res["usage"]
        if llm:
            out["llm"] = {"provider": "openai", **llm}

    return out

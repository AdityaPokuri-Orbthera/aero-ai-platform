# frontend/streamlit_app.py
from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Aero-AI Mission Studio", page_icon="🚀", layout="wide")

def _default_backend_url() -> str:
    env = os.environ.get("BACKEND_URL")
    if env:
        return env
    try:
        return st.secrets["BACKEND_URL"]
    except Exception:
        return "http://127.0.0.1:8000"

DEFAULT_BACKEND_URL = _default_backend_url()
TIMEOUT_S = 60

# -------------------------------
# Session state
# -------------------------------
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "OpenAI – gpt-4o-mini"

# chat + concept state
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_concept", None)
st.session_state.setdefault("last_spec", None)
st.session_state.setdefault("last_target", "LEO")
st.session_state.setdefault("last_origin", None)

# canvas state (IMPORTANT: never collide with widget key)
st.session_state.setdefault("canvas_version", 0)         # bump to clear canvas
st.session_state.setdefault("mission_canvas_json", None) # where we store drawn JSON

def get_backend_url() -> str:
    return (st.session_state.backend_url or DEFAULT_BACKEND_URL).rstrip("/")

def _model_id_from_choice(choice: str) -> str:
    mapping = {
        "OpenAI – gpt-4o-mini": "gpt-4o-mini",
        "OpenAI – gpt-4o": "gpt-4o",
        "OpenAI – o3-mini": "o3-mini",
        "OpenAI – gpt-4.1": "gpt-4.1",
    }
    return mapping.get(choice, "gpt-4o-mini")

# -------------------------------
# HTTP helper
# -------------------------------
def api_post(path: str, payload: dict, timeout: int = TIMEOUT_S) -> dict:
    url = f"{get_backend_url()}{path}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code >= 400:
            # try to show FastAPI detail
            try:
                j = r.json()
                detail = j.get("detail")
                raise RuntimeError(f"{r.status_code} {r.reason}: {detail}")
            except ValueError:
                raise RuntimeError(f"{r.status_code} {r.reason}: {r.text}")
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Request error calling {url}: {e}") from e

# -------------------------------
# Small utils
# -------------------------------
def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def try_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None

def coerce_overrides(obj: dict) -> dict:
    o = {}
    if "force_stages" in obj: o["force_stages"] = int(obj["force_stages"])
    if "min_diameter_m" in obj: o["min_diameter_m"] = float(obj["min_diameter_m"])
    if "target_payload_leo_kg" in obj: o["target_payload_leo_kg"] = float(obj["target_payload_leo_kg"])
    if "scale_m_per_px" in obj: o["scale_m_per_px"] = float(obj["scale_m_per_px"])
    return o

def default_overrides() -> dict:
    return {
        "force_stages": 2,
        "min_diameter_m": 1.4,
        "target_payload_leo_kg": 250.0,
        "scale_m_per_px": 0.05,
    }

def fallback_boxes_from_canvas_json(canvas_json: dict) -> List[dict]:
    boxes = []
    if not canvas_json:
        return boxes
    objs = canvas_json.get("objects") or []
    for o in objs:
        left = o.get("left"); top = o.get("top")
        width = o.get("width"); height = o.get("height")
        if left is not None and top is not None and width and height:
            boxes.append({
                "left": float(left),
                "top": float(top),
                "width": float(width),
                "height": float(height)
            })
    return boxes

def ensure_two_stage_boxes(boxes: List[dict]) -> List[dict]:
    if len(boxes) >= 2:
        return boxes[:2]
    base_w, base_h = 90.0, 240.0
    return [
        {"left": 120.0, "top": 60.0, "width": base_w, "height": base_h},
        {"left": 120.0, "top": 60.0 + base_h, "width": base_w, "height": base_h * 0.65},
    ]

def _simple_text_parse(user_text: str) -> dict:
    t = user_text.lower()
    out = {}
    if any(k in t for k in ("moon", "tli", "lunar")):
        out["_target"] = "TLI"
    states = ["ohio","michigan","louisiana","florida","texas","california","alaska","virginia","new mexico","alabama","colorado"]
    for s in states:
        if s in t:
            out["_origin_hint"] = s.title()
            break
    return out

# -------------------------------
# LLM calls
# -------------------------------
def llm_overrides_from_text(user_text: str, model_choice: str) -> dict:
    prompt = f"""
You extract mission intent and design hints from a user's request.
Model hint: {model_choice}

User text:
{user_text}

Return ONLY JSON with keys (all optional): 
{{
  "force_stages": <int 1..3>,
  "min_diameter_m": <float>,
  "target_payload_leo_kg": <float>,
  "scale_m_per_px": <float>,
  "target": "LEO" | "TLI",
  "origin_hint": <string>
}}
Do not write any explanations; ONLY JSON.
"""
    try:
        res = api_post(
            "/ai/chat",
            {
                "prompt": prompt,
                "temperature": 0.2,
                "model": _model_id_from_choice(model_choice),
            },
        )
        txt = strip_code_fences(res.get("text", "")).strip()
        data = try_json(txt) or {}
    except Exception:
        data = {}
    base = default_overrides()
    base.update(coerce_overrides(data))
    if isinstance(data.get("target"), str):
        base["_target"] = data["target"].strip().upper()
    if isinstance(data.get("origin_hint"), str):
        base["_origin_hint"] = data["origin_hint"].strip()
    base.update({k: v for k, v in _simple_text_parse(user_text).items() if v})
    return base

def estimate_spec(sketch_boxes: List[dict], overrides: dict) -> dict:
    sketch = {"objects": len(sketch_boxes), "bounding_boxes": sketch_boxes}
    body = {
        "scale_m_per_px": float(overrides.get("scale_m_per_px", 0.05)),
        "sketch": sketch,
        "overrides": {k: v for k, v in overrides.items()
                      if k in ("force_stages", "min_diameter_m", "target_payload_leo_kg")}
    }
    return api_post("/specs/estimate", body)

def compose_concept(spec: dict, target: str, origin_hint: Optional[str], model_choice: str) -> dict:
    body = {
        "spec": spec,
        "target": target,
        "origin_hint": origin_hint,
        "kb_hits": [],
        "model": _model_id_from_choice(model_choice),
    }
    return api_post("/concept/compose_from_spec?mode=pure_ai", body)

def ask_advisor(question: str, spec: dict, target: str, concept: Optional[dict], model_choice: str) -> dict:
    body = {
        "question": question,
        "spec": spec,
        "target": target,
        "concept": concept,
        "style": "teacher",
        "model": _model_id_from_choice(model_choice),
    }
    return api_post("/advisor/ask", body)

def card_kv(label: str, value: Any) -> str:
    return f"<div style='display:flex;justify-content:space-between'><span style='opacity:0.7'>{label}</span><strong>{value}</strong></div>"

def money(n: Optional[float]) -> str:
    try:
        return f"${n:,.0f}"
    except Exception:
        return "-"

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.text_input("Backend URL", value=st.session_state.backend_url, key="backend_url_box", help="FastAPI base URL")
    if st.button("Use URL"):
        st.session_state.backend_url = st.session_state.backend_url_box
        st.toast("Backend URL updated", icon="✅")

    st.selectbox(
        "Model",
        options=[
            "OpenAI – gpt-4o-mini",
            "OpenAI – gpt-4o",
            "OpenAI – o3-mini",
            "OpenAI – gpt-4.1"
        ],
        index=[
            "OpenAI – gpt-4o-mini","OpenAI – gpt-4o","OpenAI – o3-mini","OpenAI – gpt-4.1"
        ].index(st.session_state.model_choice) if st.session_state.model_choice in [
            "OpenAI – gpt-4o-mini","OpenAI – gpt-4o","OpenAI – o3-mini","OpenAI – gpt-4.1"
        ] else 0,
        key="model_choice",
        help="Select the LLM used by the backend.",
    )

    st.markdown("---")
    st.markdown("### ℹ️ Tips")
    st.caption("Type: *Create a mission to deliver a 250 kg probe to the Moon from Michigan.*")
    st.caption("Or use the canvas: draw rectangles for stages, add notes like 'Ohio' and an arrow to 'Moon'.")

# -------------------------------
# Header
# -------------------------------
st.markdown(
    """
    <style>
      .canvas-frame {border:1px solid #334155; border-radius:10px; padding:8px; background:#0b1220;}
    </style>
    <div style="display:flex;align-items:center;gap:10px">
      <div style="font-size:28px">🚀 Aero-AI Mission Studio</div>
      <div style="opacity:0.7">one input, optional sketch — AI composes the rest</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

# -------------------------------
# Canvas (optional)
# -------------------------------
# -------------------------------
# Canvas (optional)
# -------------------------------
use_sketch = False
canvas_result = None
with st.expander("✏️ Doodle board (optional)"):
    st.caption("Tip: Use **rect** for stages, **freedraw** to sketch. Then click **Use sketch**.")

    cols = st.columns([3, 1, 1])
    with cols[0]:
        draw_mode = st.radio(
            "Drawing mode", options=["rect", "freedraw", "transform"],
            horizontal=True, index=0, key="draw_mode"
        )
    with cols[1]:
        canvas_w = st.slider("Width", 800, 1600, st.session_state.get("prev_canvas_w", 1100), 50)
    with cols[2]:
        canvas_h = st.slider("Height", 300, 800, st.session_state.get("prev_canvas_h", 450), 10)

    # If size changed, bump version so Streamlit rebuilds the widget
    if canvas_w != st.session_state.get("prev_canvas_w") or canvas_h != st.session_state.get("prev_canvas_h"):
        st.session_state.canvas_version += 1
        st.session_state["prev_canvas_w"] = canvas_w
        st.session_state["prev_canvas_h"] = canvas_h

    c = st.columns([5, 1])
    with c[0]:
        st.markdown('<div class="canvas-frame">', unsafe_allow_html=True)
        # Unique widget key includes version AND size so it refreshes on resize/clear
        widget_key = f"mission_canvas_widget_v{st.session_state.canvas_version}_{canvas_w}x{canvas_h}"
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#0EA5E9",
            background_color="#0b1220",
            height=canvas_h,
            width=canvas_w,
            drawing_mode=st.session_state.draw_mode,
            key=widget_key,
            display_toolbar=True,
            update_streamlit=True,
        )
        if not (canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects")):
            st.caption("Canvas is empty — click and drag to draw.")
        st.markdown("</div>", unsafe_allow_html=True)
        # Save JSON under a state key DIFFERENT from the widget key to avoid Streamlit collisions
        if canvas_result is not None:
            st.session_state["mission_canvas_json"] = canvas_result.json_data
    with c[1]:
        use_sketch = st.button("Use sketch")
        if st.button("Clear"):
            st.session_state.canvas_version += 1
            st.session_state.mission_canvas_json = None
            st.rerun()


# -------------------------------
# Chat history
# -------------------------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("concept"):
            concept = msg["concept"]
            bom = concept.get("bom") or {}
            total = bom.get("total_est_cost")
            st.markdown("**Concept summary**")
            st.markdown(
                card_kv("Launch sites", len(concept.get("launch_sites", []))) +
                card_kv("Lunar sites", len(concept.get("lunar_sites", []))) +
                card_kv("BoM total", money(total)),
                unsafe_allow_html=True
            )

# -------------------------------
# Input or Use-sketch trigger
# -------------------------------
user_text = st.chat_input("Describe or ask anything… e.g., 'Create a lunar mission from Michigan'")

run_from_sketch = bool(use_sketch)  # always run if click Use sketch
should_run = bool(user_text) or run_from_sketch

if should_run:
    effective_text = user_text or "Create a concept from the sketch. Infer origin/target from sketch labels if present; otherwise choose sensible defaults."

    # Show user msg
    st.session_state.history.append({"role": "user", "content": effective_text})
    with st.chat_message("user"):
        st.markdown(effective_text)
        if run_from_sketch and not user_text:
            st.caption("Using sketch only (no text).")

    # 1) LLM infers parameters
    with st.spinner("Inferring mission parameters with AI…"):
        ov = llm_overrides_from_text(effective_text, st.session_state.model_choice)

    target = ov.get("_target", st.session_state.last_target) or "LEO"
    origin_hint = ov.get("_origin_hint", st.session_state.last_origin)

    # 2) Sketch boxes
    sketch_boxes: List[dict] = []
    canvas_json = st.session_state.get("mission_canvas_json")
    if canvas_json:
        sketch_boxes = fallback_boxes_from_canvas_json(canvas_json)
    if not sketch_boxes:
        sketch_boxes = ensure_two_stage_boxes(sketch_boxes)

    # 3) Estimate spec
    try:
        spec = estimate_spec(sketch_boxes, ov)
        st.session_state.last_spec = spec
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Failed to estimate spec: {e}")
        st.stop()

    # 4) Compose concept
    try:
        concept = compose_concept(
            spec, target=target, origin_hint=origin_hint,
            model_choice=st.session_state.model_choice
        )
        st.session_state.last_concept = concept
        st.session_state.last_target = target
        st.session_state.last_origin = origin_hint
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Compose failed: {e}")  # shows FastAPI 'detail' if present
        st.stop()

    # 5) If concept looks empty, auto-ask Advisor for alternatives
    empty_concept = (len(concept.get("launch_sites", [])) == 0 and len(concept.get("lunar_sites", [])) == 0)
    advisor_answer = None
    if empty_concept:
        try:
            fallback_q = (
                f"The concept for '{effective_text}' returned empty or non-viable. "
                f"Suggest a feasible alternative plan, explain constraints (e.g., launch sites near origin), "
                f"and propose concrete next actions."
            )
            advisor = ask_advisor(fallback_q, spec, target, concept, model_choice=st.session_state.model_choice)
            advisor_answer = advisor.get("answer")
        except Exception:
            advisor_answer = None
    else:
        if effective_text.strip().endswith("?"):
            try:
                advisor = ask_advisor(effective_text, spec, target, concept, model_choice=st.session_state.model_choice)
                advisor_answer = advisor.get("answer")
            except Exception:
                advisor_answer = None

    # 6) Render assistant reply
    with st.chat_message("assistant"):
        report_md = concept.get("report_md")

        if empty_concept and advisor_answer:
            st.warning("Initial concept looked non-viable. Here’s a guided alternative:")
            st.markdown(advisor_answer.get("answer_md", ""))
        else:
            if report_md:
                st.markdown(report_md)
            else:
                st.markdown("**Concept created.**")

        # KPIs
        bom = concept.get("bom") or {}
        total = bom.get("total_est_cost")
        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.metric("Launch sites", len(concept.get("launch_sites", [])))
        with kpi_cols[1]:
            st.metric("Lunar sites", len(concept.get("lunar_sites", [])))
        with kpi_cols[2]:
            st.metric("BoM total", money(total))

        st.markdown("---")

        if concept.get("launch_sites"):
            st.subheader("Launch site options")
            for s in concept["launch_sites"]:
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.markdown(f"**{s.get('name','?')}** · {s.get('state','')}, {s.get('country','')}")
                    st.caption(s.get("why", ""))
                with cols[1]:
                    score = (s.get("suitability_score") or 0) * 100
                    st.metric("Suitability", f"{score:.0f}%")
                with cols[2]:
                    st.caption(f"Type: {s.get('type','unknown')}")
                    st.caption(f"FAA: {'Yes' if s.get('faa_licensed') else '—'}")

        if concept.get("lunar_sites"):
            st.subheader("Lunar site candidates")
            for ls in concept["lunar_sites"]:
                cols = st.columns([3, 2])
                with cols[0]:
                    st.markdown(f"**{ls.get('name','?')}**")
                    st.caption(ls.get("why", ""))
                with cols[1]:
                    traits = ", ".join(ls.get("traits", [])[:6])
                    st.caption(f"Traits: {traits}")

        if bom.get("items"):
            st.subheader("Bill of Materials (concept level)")
            st.table([
                {"Item": it.get("item","?"), "Qty": it.get("qty","—"), "UoM": it.get("uom","—"), "Est. Cost": money(it.get("est_cost"))}
                for it in bom.get("items", [])
            ])
            st.markdown(f"**Total (est.)**: {money(bom.get('total_est_cost'))}  \n_Uncertainty_: {bom.get('uncertainty','—')}")

    st.session_state.history.append({
        "role": "assistant",
        "content": (advisor_answer or {}).get("answer_md") if empty_concept else (concept.get("report_md") or "Concept created."),
        "concept": concept
    })

    st.rerun()

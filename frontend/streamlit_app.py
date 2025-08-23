import base64
import io
import json
import os
from typing import List, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ------------------ Backend endpoints ------------------
BACKEND_URL = os.getenv("AERO_BACKEND_URL", "http://127.0.0.1:8000")
AI_CHAT_URL        = f"{BACKEND_URL}/ai/chat"
KB_LIST_URL        = f"{BACKEND_URL}/kb/rockets"
KB_SEARCH_URL      = f"{BACKEND_URL}/kb/search"
SPECS_ESTIMATE_URL = f"{BACKEND_URL}/specs/estimate"
MISSION_PLAN_URL   = f"{BACKEND_URL}/specs/plan"
BLUEPRINT_URL      = f"{BACKEND_URL}/specs/blueprint.svg"
TRAJ_URL           = f"{BACKEND_URL}/mission/trajectory.png"
CONCEPT_COMPOSE_FROM_SPEC_URL  = f"{BACKEND_URL}/concept/compose_from_spec"  # will be called with ?mode=pure_ai

st.set_page_config(page_title="Aero-AI (Guided)", page_icon="🚀", layout="wide")

# ------------------ Session state ------------------
for key, default in [
    ("mode", "Guided (Beginner)"),
    ("kb_hits", []),
    ("spec_result", None),
    ("mission_plan", None),
    ("blueprint_svg", None),
    ("sketch_b64", None),
    ("sketch_meta", {"objects": 0, "bounding_boxes": []}),
    ("last_run_ok", False),
    ("bp_theme", "blueprint"),
    ("concept_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------ Helpers ------------------
def kb_search(query: str, limit: int = 3) -> List[dict]:
    try:
        r = requests.get(KB_SEARCH_URL, params={"q": query, "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"KB search failed: {e}")
        return []

def kb_list() -> List[dict]:
    try:
        r = requests.get(KB_LIST_URL, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"KB list failed: {e}")
        return []

def post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except ValueError:
        return {"_raw_text": r.text}
    except Exception as e:
        return {"error": str(e)}

def call_ai(prompt: str, temperature: float = 0.2, max_tokens: Optional[int] = None,
            provider: Optional[str] = None, model: Optional[str] = None) -> dict:
    params = {}
    if provider: params["provider"] = provider
    if model: params["model"] = model
    payload = {"prompt": prompt, "temperature": temperature}
    if max_tokens is not None: payload["max_tokens"] = max_tokens
    try:
        r = requests.post(AI_CHAT_URL, params=params, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_prompt(
    user_text: str,
    kb_hits: List[dict],
    sketch_meta: dict,
    sketch_b64: Optional[str],
    spec_result: Optional[dict] = None,
    mission_plan: Optional[dict] = None,
) -> str:
    # KB snippets (short, non-technical)
    brief_blocks = []
    for r in kb_hits:
        payloads = []
        if r.get("payload_leo_kg"): payloads.append(f"LEO={r['payload_leo_kg']} kg")
        if r.get("payload_tli_kg"): payloads.append(f"TLI={r['payload_tli_kg']} kg")
        engines = "; ".join([f"St{e.get('stage')}: {e.get('count')}× {e.get('type')}" for e in r.get("engines", [])])
        brief = (
            f"{r.get('name')} — stages={r.get('stages')}, H={r.get('height_m')} m, D={r.get('diameter_m')} m, "
            f"payloads({', '.join(payloads) if payloads else 'n/a'}), engines: {engines}."
        )
        brief_blocks.append(brief)
    kb_chunk = "\n".join(brief_blocks) if brief_blocks else "No KB snippets."

    meta_chunk = json.dumps(sketch_meta, indent=2)
    img_note = "A base64 PNG of the doodle is attached below." if sketch_b64 else "No image attached."
    image_block = f"\n\n[DOODLE_BASE64_PNG]: {sketch_b64[:256]}... (truncated)" if sketch_b64 else ""

    extra_blocks = []
    if spec_result:
        extra_blocks.append("SPEC_DRAFT JSON:\n" + json.dumps(spec_result, indent=2))
    if mission_plan:
        extra_blocks.append("MISSION_PLAN JSON:\n" + json.dumps(mission_plan, indent=2))
    extras = ("\n\n" + "\n\n".join(extra_blocks)) if extra_blocks else ""

    return (
        "You are an aerospace design assistant for beginners. "
        "Explain simply, keep steps clear, and avoid jargon unless necessary. "
        "Use the structured JSON and KB below; do not invent numbers.\n\n"
        f"USER GOAL:\n{user_text}\n\n"
        f"KB:\n{kb_chunk}\n\n"
        f"SKETCH META:\n{meta_chunk}\n\n"
        f"{img_note}{image_block}{extras}"
    )

def template_bounding_boxes(stage_count: int, total_px: int = 300, max_width_px: int = 60) -> List[dict]:
    boxes = []
    if stage_count == 2:
        heights = [int(total_px * 0.65), int(total_px * 0.35)]
    else:
        heights = [int(total_px * 0.55), int(total_px * 0.30), int(total_px * 0.15)]
    y = 20
    x = 100
    for h in heights:
        boxes.append({"left": x, "top": y, "width": max_width_px, "height": h})
        y += h
    return boxes

def guided_estimate_and_plan(
    height_m: float,
    min_diameter_m: float,
    payload_target_kg: Optional[float],
    stage_count: int,
    upper_prop: Optional[str],
    mission_target: str,  # "LEO" or "TLI"
) -> tuple[Optional[dict], Optional[dict], Optional[str], Optional[bytes], Optional[str]]:
    total_px = 300
    m_per_px = max(0.001, height_m / total_px)

    bboxes = template_bounding_boxes(stage_count, total_px=total_px,
                                     max_width_px=max(40, int(min_diameter_m / m_per_px)))
    sketch = {"objects": len(bboxes), "bounding_boxes": bboxes}

    overrides = {}
    if payload_target_kg and payload_target_kg > 0:
        overrides["target_payload_leo_kg"] = float(payload_target_kg)
    if stage_count in (2, 3):
        overrides["force_stages"] = stage_count
    if upper_prop in ("LH2/LOX", "RP1/LOX"):
        overrides["preferred_upper_propellant"] = upper_prop
    if min_diameter_m and min_diameter_m > 0:
        overrides["min_diameter_m"] = float(min_diameter_m)

    # Deterministic backbone for testing (OK per your note)
    est_payload = {"scale_m_per_px": float(m_per_px), "sketch": sketch, "overrides": overrides or None}
    spec = post_json(SPECS_ESTIMATE_URL, est_payload)
    if "error" in spec:
        return None, None, None, None, f"Estimate error: {spec['error']}"

    plan_payload = {"spec": spec, "target": "TLI" if mission_target.upper().startswith("TLI") else "LEO"}
    mp = post_json(MISSION_PLAN_URL, plan_payload)
    if "error" in mp:
        return spec, None, None, None, f"Plan error: {mp['error']}"

    svg_resp = requests.post(f"{BLUEPRINT_URL}?theme={st.session_state.bp_theme}", json={"spec": spec}, timeout=15)
    svg_resp.raise_for_status()
    svg = svg_resp.text

    traj_resp = requests.post(TRAJ_URL, json={"spec": spec, "plan": mp}, timeout=15)
    traj_resp.raise_for_status()
    traj_png = traj_resp.content

    return spec, mp, svg, traj_png, None

def call_concept_ai_only(spec: dict, target: str, origin_hint: str, kb_hits: Optional[List[dict]]) -> dict:
    """Force AI-only composition for sites/lunar/BOM (no static fallback)."""
    payload = {
        "spec": spec,
        "target": target,
        "origin_hint": origin_hint or None,
        "kb_hits": kb_hits or []
    }
    # Force pure AI mode
    url = f"{CONCEPT_COMPOSE_FROM_SPEC_URL}?mode=pure_ai"
    return post_json(url, payload, timeout=60)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("🧭 Mode")
    st.session_state.mode = st.radio(
        "Choose how you want to design:",
        ["Guided (Beginner)", "Canvas (Advanced)"],
        index=0,
        help="Guided: no drawing needed. Canvas: draw shapes and set scale manually."
    )

    st.markdown("---")
    st.subheader("Model (for AI guidance)")
    provider = st.selectbox("Provider", ["openai"], index=0)
    model = st.text_input("Model", value=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens_val = st.number_input("Max tokens (optional)", min_value=0, value=0)
    max_tokens = None if max_tokens_val == 0 else int(max_tokens_val)

    with st.expander("Reference rockets (optional KB search)"):
        kb_query = st.text_input("Search (e.g., falcon, saturn, kerosene)")
        if st.button("Search KB"):
            st.session_state.kb_hits = kb_search(kb_query, limit=3)
        if st.session_state.kb_hits:
            for r in st.session_state.kb_hits:
                st.write(f"• **{r.get('name')}** — id `{r.get('id')}`, LEO payload {r.get('payload_leo_kg','?')} kg")

st.title("🚀 Aero-AI — Beginner-friendly Rocket Designer")
st.caption("AI-composed mission context (sites, lunar targets, BoM) is enabled by default.")

# ------------------ Guided Mode ------------------
if st.session_state.mode == "Guided (Beginner)":
    st.info("No drawing required. Pick your mission and basics, then click **Build Concept** (AI-only composition).")

    colA, colB, colC = st.columns(3)
    with colA:
        preset = st.selectbox(
            "Mission preset",
            ["Small LEO (200 kg)", "Lunar Probe (TLI 300 kg)", "Custom"],
            help="Presets fill reasonable defaults. Choose Custom to set your own."
        )
    with colB:
        stage_count = st.selectbox("Stages", [2, 3], index=0)
    with colC:
        upper_prop = st.selectbox("Upper stage propellant", ["auto", "LH2/LOX", "RP1/LOX"], index=0)

    if preset == "Small LEO (200 kg)":
        mission_target = "LEO"
        payload_target = 200
        height_m = 20.0
        min_diam_m = 1.2
    elif preset == "Lunar Probe (TLI 300 kg)":
        mission_target = "TLI"
        payload_target = 300
        height_m = 30.0
        min_diam_m = 1.8
    else:
        col1, col2 = st.columns(2)
        with col1:
            mission_target = st.selectbox("Mission target", ["LEO", "TLI (Moon)"], index=0)
            payload_target = st.number_input("Target payload to LEO (kg) (approx.)", min_value=0, value=200)
        with col2:
            height_m = st.number_input("Total rocket height (m)", min_value=5.0, value=25.0, step=0.5)
            min_diam_m = st.number_input("Minimum diameter (m)", min_value=0.5, value=1.2, step=0.1)

    st.markdown("### Appearance")
    st.session_state.bp_theme = st.selectbox("Blueprint theme", ["blueprint", "light"], index=0)

    origin_hint = st.text_input("Launch origin (city/state/site, e.g., 'Louisiana', 'New Orleans')", value="")

    st.markdown("### Build")
    if st.button("Build Concept"):
        with st.spinner("Estimating spec & plan + generating AI concept..."):
            spec, mp, svg, traj_png, err = guided_estimate_and_plan(
                height_m=height_m,
                min_diameter_m=min_diam_m,
                payload_target_kg=payload_target,
                stage_count=int(stage_count),
                upper_prop=None if upper_prop == "auto" else upper_prop,
                mission_target=mission_target,
            )
            if err:
                st.error(err)
            else:
                st.session_state.spec_result = spec
                st.session_state.mission_plan = mp
                st.session_state.blueprint_svg = svg
                st.session_state.last_run_ok = True

                # AI-only composition (sites, lunar, BoM)
                target_str = "TLI" if mission_target.upper().startswith("TLI") else "LEO"
                concept = call_concept_ai_only(spec, target_str, origin_hint, st.session_state.kb_hits)
                if "error" in concept:
                    st.error(concept["error"])
                    st.session_state.concept_result = None
                else:
                    st.session_state.concept_result = concept

        if st.session_state.last_run_ok:
            st.success("Concept ready!")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Spec Draft (deterministic for testing)")
                with st.expander("Show JSON", expanded=False):
                    st.json(st.session_state.spec_result)
                st.subheader("Mission Plan (deterministic for testing)")
                with st.expander("Show JSON", expanded=False):
                    st.json(st.session_state.mission_plan)
            with c2:
                st.subheader("Blueprint")
                if st.session_state.blueprint_svg:
                    components.html(st.session_state.blueprint_svg, height=650, scrolling=True)

            st.subheader("Trajectory Sketch")
            try:
                if traj_png:
                    st.image(traj_png, caption="Ascent & Transfer (illustrative)", use_column_width=True)
                else:
                    resp = requests.post(TRAJ_URL, json={
                        "spec": st.session_state.spec_result, "plan": st.session_state.mission_plan
                    }, timeout=15)
                    resp.raise_for_status()
                    st.image(resp.content, caption="Ascent & Transfer (illustrative)", use_column_width=True)
            except Exception as e:
                st.info(f"Trajectory sketch unavailable: {e}")

            # -------- AI Sections (no static fallback) --------
            concept = st.session_state.concept_result or {}
            st.markdown("## 🌐 AI-Composed Mission Context")
            if concept.get("note"):
                st.info(concept["note"])

            st.markdown("### Launch Sites (AI-curated)")
            if concept.get("launch_sites"):
                for i, s in enumerate(concept["launch_sites"][:5], start=1):
                    badge = "✅" if s.get("faa_licensed") else "ℹ️"
                    st.write(f"{i}. {badge} **{s.get('name','?')}** "
                             f"({s.get('state','?')}, {s.get('country','?')}) — "
                             f"{s.get('type','?')} | score {s.get('suitability_score',0):.2f}")
                    if s.get("why"): st.caption(s["why"])
            else:
                st.warning("AI did not return any sites.")

            st.markdown("### Candidate Lunar Sites (AI-curated)")
            if concept.get("lunar_sites"):
                for ls in concept["lunar_sites"]:
                    st.write(f"• **{ls.get('name','?')}** — " + ", ".join(ls.get("traits", [])))
                    if ls.get("why"): st.caption(ls["why"])
            else:
                st.warning("AI did not return lunar sites.")

            st.markdown("### Bill of Materials & Cost (AI-curated)")
            if concept.get("bom"):
                bom = concept["bom"]
                total = bom.get("total_est_cost", 0)
                currency = bom.get("currency", "USD")
                st.write(f"**Total Estimated Cost:** ~{currency} {total:,.0f} ({bom.get('uncertainty','')})")
                for it in bom.get("items", []):
                    qty = it.get("qty")
                    uom = it.get("uom","")
                    qty_txt = (f"{qty:.0f} {uom}".strip()) if isinstance(qty, (int,float)) else (f"{qty} {uom}".strip() if qty else uom)
                    st.write(f"• {it.get('item','?')}: {qty_txt} → {currency} {it.get('est_cost',0):,.0f}")
            else:
                st.warning("AI did not return a BoM.")

# ------------------ Canvas Mode ------------------
else:
    st.info("Draw simple rectangles/circles for stages. Add a long curved path if you want a Moon mission.")
    with st.expander("Canvas & Scale", expanded=True):
        colL, colR = st.columns([3, 1])
        with colL:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.0)",
                stroke_width=3,
                stroke_color="#222222",
                background_color="#ffffff",
                height=350,
                width=700,
                drawing_mode=st.selectbox("Mode", ["freedraw", "line", "rect", "circle", "transform"], index=2),
                key="canvas",
            )
        with colR:
            scale_m_per_px = st.number_input(
                "Scale (meters per pixel)",
                min_value=0.0001,
                value=0.05,
                step=0.01,
                help="Example: 0.05 m/px ≈ 20 px per meter",
            )
            payload_target = st.number_input("Target payload to LEO (kg, optional)", min_value=0, value=0)
            force_stages = st.selectbox("Stages", ["auto", "2", "3"], index=0)
            upper_prop = st.selectbox("Upper stage propellant", ["auto", "LH2/LOX", "RP1/LOX"], index=0)
            min_diam = st.number_input("Minimum diameter (m, optional)", min_value=0.0, value=0.0, step=0.1)
            st.session_state.bp_theme = st.selectbox("Blueprint theme", ["blueprint", "light"], index=0)
            origin_hint = st.text_input("Launch origin (optional)", value="")

    # Extract sketch + detect "Moon path"
    implied_target = None
    if canvas_result is not None:
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            st.session_state.sketch_b64 = pil_to_base64(img)

        sketch_meta = {"objects": 0, "bounding_boxes": []}
        objs = canvas_result.json_data or {}
        if isinstance(objs, dict) and "objects" in objs:
            sketch_meta["objects"] = len(objs["objects"])
            for o in objs["objects"]:
                left = o.get("left"); top = o.get("top")
                width = o.get("width"); height = o.get("height")
                if None not in (left, top, width, height):
                    sketch_meta["bounding_boxes"].append(
                        {"left": round(float(left), 2), "top": round(float(top), 2),
                         "width": round(float(width), 2), "height": round(float(height), 2)}
                    )
                if (o.get("type") == "path") or (o.get("path")):
                    path_len = len(o.get("path", [])) if isinstance(o.get("path"), list) else 0
                    w = float(o.get("width") or 0); h = float(o.get("height") or 0)
                    if path_len > 40 or w > 350 or h > 220:
                        implied_target = "TLI"
        st.session_state.sketch_meta = sketch_meta

    target_choice = st.selectbox("Mission target", ["LEO", "TLI (Moon)"], index=1 if implied_target == "TLI" else 0)

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("Estimate"):
            if not st.session_state.sketch_meta.get("bounding_boxes"):
                st.warning("Draw a couple of rectangles first.")
            else:
                overrides = {}
                if payload_target > 0: overrides["target_payload_leo_kg"] = float(payload_target)
                if force_stages in ("2", "3"): overrides["force_stages"] = int(force_stages)
                if upper_prop != "auto": overrides["preferred_upper_propellant"] = upper_prop
                if min_diam > 0: overrides["min_diameter_m"] = float(min_diam)

                payload = {
                    "scale_m_per_px": float(scale_m_per_px),
                    "sketch": {
                        "objects": int(st.session_state.sketch_meta["objects"]),
                        "bounding_boxes": st.session_state.sketch_meta["bounding_boxes"],
                    },
                    "overrides": overrides or None,
                }
                with st.spinner("Estimating specs..."):
                    st.session_state.spec_result = post_json(SPECS_ESTIMATE_URL, payload)
                if "error" in st.session_state.spec_result:
                    st.error(st.session_state.spec_result["error"])
                else:
                    st.success("Spec draft ready.")
    with colB:
        if st.button("Blueprint"):
            if not st.session_state.spec_result or "error" in (st.session_state.spec_result or {}):
                st.info("Run Estimate first.")
            else:
                r = requests.post(
                    f"{BLUEPRINT_URL}?theme={st.session_state.bp_theme}",
                    json={"spec": st.session_state.spec_result},
                    timeout=15
                )
                if r.status_code == 200:
                    st.session_state.blueprint_svg = r.text
                    st.success("Blueprint rendered.")
                else:
                    st.error(f"Blueprint failed: {r.text}")
    with colC:
        if st.button("Plan + AI Concept"):
            if not st.session_state.spec_result or "error" in (st.session_state.spec_result or {}):
                st.info("Run Estimate first.")
            else:
                tgt = "TLI" if target_choice.startswith("TLI") else "LEO"
                plan_payload = {"spec": st.session_state.spec_result, "target": tgt}
                with st.spinner("Planning mission..."):
                    st.session_state.mission_plan = post_json(MISSION_PLAN_URL, plan_payload)
                if "error" in st.session_state.mission_plan:
                    st.error(st.session_state.mission_plan["error"])
                else:
                    st.success("Mission plan ready.")
                    # AI-only composition
                    with st.spinner("AI composing launch sites, lunar targets, and BoM..."):
                        concept = call_concept_ai_only(st.session_state.spec_result, tgt, origin_hint, st.session_state.kb_hits)
                        if "error" in concept:
                            st.error(concept["error"])
                            st.session_state.concept_result = None
                        else:
                            st.session_state.concept_result = concept

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.session_state.spec_result and "error" not in st.session_state.spec_result:
            st.subheader("Spec Draft (deterministic for testing)")
            with st.expander("Show JSON", expanded=False):
                st.json(st.session_state.spec_result)
        if st.session_state.mission_plan and "error" not in st.session_state.mission_plan:
            st.subheader("Mission Plan (deterministic for testing)")
            with st.expander("Show JSON", expanded=False):
                st.json(st.session_state.mission_plan)

        # AI-only sections
        concept = st.session_state.concept_result or {}
        st.markdown("## 🌐 AI-Composed Mission Context")
        if concept.get("note"): st.info(concept["note"])

        st.markdown("### Launch Sites (AI-curated)")
        if concept.get("launch_sites"):
            for i, s in enumerate(concept["launch_sites"][:5], start=1):
                badge = "✅" if s.get("faa_licensed") else "ℹ️"
                st.write(f"{i}. {badge} **{s.get('name','?')}** "
                         f"({s.get('state','?')}, {s.get('country','?')}) — "
                         f"{s.get('type','?')} | score {s.get('suitability_score',0):.2f}")
                if s.get("why"): st.caption(s["why"])
        else:
            st.warning("AI did not return sites.")

        st.markdown("### Candidate Lunar Sites (AI-curated)")
        if concept.get("lunar_sites"):
            for ls in concept.get("lunar_sites", []):
                st.write(f"• **{ls.get('name','?')}** — " + ", ".join(ls.get("traits", [])))
                if ls.get("why"): st.caption(ls["why"])
        else:
            st.warning("AI did not return lunar sites.")

        st.subheader("Trajectory Sketch")
        if st.session_state.spec_result and st.session_state.mission_plan and "error" not in (st.session_state.mission_plan or {}):
            try:
                resp = requests.post(TRAJ_URL, json={
                    "spec": st.session_state.spec_result,
                    "plan": st.session_state.mission_plan
                }, timeout=15)
                resp.raise_for_status()
                st.image(resp.content, caption="Ascent & Transfer (illustrative)", use_column_width=True)
            except Exception as e:
                st.info(f"Trajectory sketch unavailable: {e}")
        else:
            st.info("Run Estimate + Plan first.")
    with c2:
        st.subheader("Blueprint")
        if st.session_state.blueprint_svg:
            components.html(st.session_state.blueprint_svg, height=650, scrolling=True)
        else:
            st.info("Click Blueprint after Estimate.")

    st.markdown("---")
    st.subheader("Ask the AI (advanced)")
    default_prompt = (
        "Summarize the design and suggest improvements. "
        "Use SPEC_DRAFT and MISSION_PLAN strictly; do not invent numbers."
    )
    user_text = st.text_area("Prompt", value=default_prompt, height=120)
    if st.button("Get AI Guidance"):
        prompt = build_prompt(
            user_text=user_text,
            kb_hits=st.session_state.kb_hits,
            sketch_meta=st.session_state.sketch_meta,
            sketch_b64=st.session_state.sketch_b64,
            spec_result=st.session_state.spec_result if st.session_state.spec_result and "error" not in st.session_state.spec_result else None,
            mission_plan=st.session_state.mission_plan if st.session_state.mission_plan and "error" not in st.session_state.mission_plan else None,
        )
        with st.spinner("Calling /ai/chat ..."):
            res = call_ai(prompt, temperature=temperature, max_tokens=max_tokens, provider=provider, model=model)
        if "error" in res:
            st.error(res["error"])
        else:
            st.success("AI guidance")
            st.write(res.get("text", "(no text)"))

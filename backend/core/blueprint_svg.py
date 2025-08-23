from __future__ import annotations
from typing import Tuple
from backend.core.parametric_specs import RocketSpecDraft, StageEstimate

PAD = 20       # px padding around rocket
PX_PER_M = 5   # meters → pixels
TEXT_SIZE = 12

def _fmt(val: float, n: int = 2) -> str:
    return f"{val:.{n}f}"

def _palette(theme: str) -> Tuple[str, str, str, str, str]:
    """
    Returns (bg, stroke, text, axis, accent) colors.
    theme: "blueprint" | "light"
    """
    t = (theme or "blueprint").lower()
    if t == "light":
        return ("#ffffff", "#111111", "#111111", "#888888", "#0b6cf0")
    # default: blueprint
    return ("#0a1a2f", "#7fd0ff", "#bfe9ff", "#2a4a6a", "#7fd0ff")

def make_blueprint_svg(spec: RocketSpecDraft, theme: str = "blueprint") -> str:
    H = spec.total_height_m or 1.0
    D = spec.max_diameter_m or 1.0
    canvas_h = int(H * PX_PER_M + PAD * 2 + 40)
    canvas_w = int(max(D, 1.0) * PX_PER_M + PAD * 2 + 180)

    bg, stroke, text, axis, accent = _palette(theme)
    cx = PAD + (D * PX_PER_M) / 2

    # Stage boxes (top→bottom)
    y_cursor = PAD
    stage_boxes = []
    for s in spec.stages[::-1]:
        h_px = max(10.0, s.length_m * PX_PER_M)
        w_px = max(8.0, s.diameter_m * PX_PER_M)
        x = PAD + (D * PX_PER_M - w_px) / 2
        y = y_cursor
        stage_boxes.append((x, y, w_px, h_px, s))
        y_cursor += h_px

    # CSS (colors per theme)
    css = f"""
    .t {{ font: {TEXT_SIZE}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill:{text} }}
    .b {{ fill:none; stroke:{stroke}; stroke-width:2 }}
    .axis {{ stroke:{axis}; stroke-width:1; stroke-dasharray:4 4 }}
    .grid {{ stroke:{axis}; stroke-width:0.5; stroke-opacity:0.5 }}
    """

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        f'<rect x="0" y="0" width="{canvas_w}" height="{canvas_h}" fill="{bg}"/>',
        "<defs><style>", css, "</style></defs>",
    ]

    # Grid
    grid_step = 20
    for x in range(PAD, canvas_w - PAD, grid_step):
        svg.append(f'<line class="grid" x1="{x}" y1="{PAD}" x2="{x}" y2="{canvas_h-PAD}" />')
    for y in range(PAD, canvas_h - PAD, grid_step):
        svg.append(f'<line class="grid" x1="{PAD}" y1="{y}" x2="{canvas_w-PAD}" y2="{y}" />')

    # Centerline axis
    svg.append(f'<line class="axis" x1="{cx}" y1="{PAD}" x2="{cx}" y2="{canvas_h-PAD}" />')

    # Draw stages (top first for labels stacking)
    for (x, y, w, h, s) in stage_boxes[::-1]:
        svg.append(f'<rect class="b" x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" rx="6" ry="6"/>')
        labx = x + w + 10
        svg.append(f'<text class="t" x="{_fmt(labx)}" y="{_fmt(y + TEXT_SIZE + 2)}">Stage {s.stage}</text>')
        svg.append(f'<text class="t" x="{_fmt(labx)}" y="{_fmt(y + TEXT_SIZE + 18)}">L={_fmt(s.length_m,1)} m</text>')
        svg.append(f'<text class="t" x="{_fmt(labx)}" y="{_fmt(y + TEXT_SIZE + 34)}">D={_fmt(s.diameter_m,2)} m</text>')
        svg.append(f'<text class="t" x="{_fmt(labx)}" y="{_fmt(y + TEXT_SIZE + 50)}">{s.propellant}, Isp={int(s.isp_s)} s</text>')
        svg.append(f'<text class="t" x="{_fmt(labx)}" y="{_fmt(y + TEXT_SIZE + 66)}">Thrust≈{_fmt(s.engine_thrust_kN,0)} kN</text>')

    # Overall
    svg.append(f'<text class="t" x="{PAD}" y="{canvas_h-PAD-24}">H={_fmt(spec.total_height_m,1)} m  Dmax={_fmt(spec.max_diameter_m,2)} m</text>')
    svg.append(f'<text class="t" x="{PAD}" y="{canvas_h-PAD-8}">m0≈{_fmt(spec.liftoff_mass_kg,0)} kg, LEO payload≈{_fmt(spec.payload_leo_kg,0)} kg</text>')
    svg.append("</svg>")
    return "\n".join(svg)

from __future__ import annotations
import io
import math

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from backend.core.parametric_specs import MissionPlan, RocketSpecDraft, MissionTarget

EARTH_RADIUS_KM = 6371.0
LEO_ALT_KM = 200.0
MOON_DIST_KM = 384400.0

def _circle(ax, r, **kw):
    th = [i * math.pi / 180 for i in range(0, 361)]
    x = [r * math.cos(t) for t in th]
    y = [r * math.sin(t) for t in th]
    ax.plot(x, y, **kw)

def make_trajectory_png(plan: MissionPlan, spec: RocketSpecDraft) -> bytes:
    """
    Render a simple, not-to-scale 2D sketch:
      - Earth (circle)
      - LEO (small ring)
      - If TLI: an elliptical arc with perigee near LEO and 'clamped' apogee
    """
    # Normalize radii so Earth ~= 1.0
    r_e = 1.0
    r_leo = (EARTH_RADIUS_KM + LEO_ALT_KM) / EARTH_RADIUS_KM  # ~1.031
    r_apogee = MOON_DIST_KM / EARTH_RADIUS_KM                 # ~60.3 Re
    # Clamp for visualization (keep figure reasonable)
    r_apogee_vis = min(r_apogee, 5.5)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Ascent & Transfer (not to scale)")

    # Earth
    _circle(ax, r_e, color="black")
    ax.fill([0], [0], "white")
    ax.text(0, -r_e - 0.1, "Earth", ha="center", va="top", fontsize=9)

    # LEO ring
    _circle(ax, r_leo, color="black", linestyle="--")
    ax.text(r_leo + 0.05, 0, "LEO (~200 km)", fontsize=8, va="center")

    # Ascent arc (small portion from Earth to LEO)
    th_ascent = [i * math.pi / 180 for i in range(80, 100)]
    ax.plot([r_e * math.cos(t) for t in th_ascent],
            [r_e * math.sin(t) for t in th_ascent],
            linewidth=2)

    if plan.target == MissionTarget.TLI:
        # Simple ellipse with perigee at r_leo; apogee clamped
        a = (r_leo + r_apogee_vis) / 2.0
        c = a - r_leo  # focus at Earth center
        b = math.sqrt(max(a*a - c*c, 1e-6))
        # Parametric ellipse around Earth focus at (0,0) (approx visual)
        th = [i * math.pi / 180 for i in range(0, 180)]
        x = [a * math.cos(t) - c for t in th]
        y = [b * math.sin(t) for t in th]
        ax.plot(x, y, linewidth=2)
        ax.text(-a, 0.1, "TLI ellipse (clamped)", fontsize=8, va="bottom")
    else:
        ax.text(0, r_leo + 0.2, "LEO mission", fontsize=9, ha="center")

    # Cosmetics
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.2, 6.2)
    ax.axis("off")
    ax.text(0, -6.0, "Diagram is illustrative; not to scale.", fontsize=8, ha="center", color="#555")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return buf.getvalue()

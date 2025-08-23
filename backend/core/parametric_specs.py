from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum  # <-- add this import near the top
from enum import Enum
from pydantic import BaseModel, Field
import math

# ------------------ Models ------------------
class EstimateOverrides(BaseModel):
    force_stages: int | None = Field(default=None, description="If set, force 1–3 stages")
    target_payload_leo_kg: float | None = Field(default=None, gt=0)
    preferred_upper_propellant: str | None = Field(default=None, description="e.g., 'LH2/LOX' or 'RP1/LOX'")
    min_diameter_m: float | None = Field(default=None, gt=0)

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float

class SketchMeta(BaseModel):
    objects: int = 0
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)

class StageEstimate(BaseModel):
    stage: int
    length_m: float
    diameter_m: float
    propellant: str
    isp_s: float
    structural_mass_kg: float
    propellant_mass_kg: float
    engine_thrust_kN: float  # total for the stage

class RocketSpecDraft(BaseModel):
    stages: List[StageEstimate]
    total_height_m: float
    max_diameter_m: float
    liftoff_mass_kg: float
    payload_leo_kg: float
    notes: str

# ------------------ Heuristics ------------------
# Simple ISP lookup
ISP_TABLE = {
    "RP1/LOX": 300.0,    # blended sea-level/vac average for small launchers
    "LH2/LOX": 430.0,    # vacuum-ish for upper stage concept
    "Solid":   270.0,
}

def guess_stage_count(bboxes: List[BoundingBox]) -> int:
    # If we have multiple stacked rectangles/circles, infer 2–3 stages; else default 2
    if len(bboxes) >= 3:
        return 3
    return 2

def px_extents(bboxes: List[BoundingBox]) -> Tuple[float, float]:
    """Return (height_px, diameter_px) from all boxes."""
    if not bboxes:
        return 200.0, 20.0  # default if nothing drawn
    top = min(bb.top for bb in bboxes)
    bottom = max(bb.top + bb.height for bb in bboxes)
    height_px = max(1.0, bottom - top)
    diameter_px = max(1.0, max(bb.width for bb in bboxes))
    return height_px, diameter_px

def split_lengths(total_h_m: float, n: int) -> List[float]:
    """Split height across stages (e.g., 65/35 for 2 stages; 55/30/15 for 3)."""
    if n == 2:
        return [total_h_m * 0.65, total_h_m * 0.35]
    if n == 3:
        return [total_h_m * 0.55, total_h_m * 0.30, total_h_m * 0.15]
    return [total_h_m]

def choose_propellants(n: int) -> List[str]:
    if n == 2:
        return ["RP1/LOX", "LH2/LOX"]   # kerolox booster, hydrolox upper
    if n == 3:
        return ["RP1/LOX", "RP1/LOX", "LH2/LOX"]
    return ["RP1/LOX"]

def mass_fractions_for_small_lifter(n: int) -> List[Tuple[float, float]]:
    """
    Return list of (structural_fraction, propellant_fraction) per stage.
    Very rough ballparks for a small launcher.
    """
    if n == 2:
        return [(0.08, 0.86), (0.10, 0.80)]
    if n == 3:
        return [(0.08, 0.86), (0.09, 0.84), (0.10, 0.78)]
    return [(0.10, 0.80)]

def estimate_liftoff_mass(height_m: float, diameter_m: float) -> float:
    """
    Naive scaling: mass ~ k * H * D^2; tuned to yield Electron-like masses at ~20m x 1.2m.
    """
    k = 18.0  # fudge factor
    return k * height_m * (diameter_m ** 2) * 100.0  # kg

def thrust_for_twr(mass_kg: float, twr: float = 1.4) -> float:
    g = 9.80665
    return (mass_kg * g * twr) / 1000.0  # kN

# ------------------ Public API ------------------
def estimate_specs(sketch: SketchMeta, scale_m_per_px: float, overrides: EstimateOverrides | None = None) -> RocketSpecDraft:
    overrides = overrides or EstimateOverrides()

    # 1) infer overall size
    h_px, d_px = px_extents(sketch.bounding_boxes)
    height_m = h_px * max(0.001, scale_m_per_px)
    diameter_m = d_px * max(0.001, scale_m_per_px)
    if overrides.min_diameter_m:
        diameter_m = max(diameter_m, overrides.min_diameter_m)

    # 2) stages & propellants
    n = overrides.force_stages if overrides.force_stages in (1,2,3) else guess_stage_count(sketch.bounding_boxes)
    stage_lengths = split_lengths(height_m, n)
    props = choose_propellants(n)
    if overrides.preferred_upper_propellant and n >= 2:
        props[-1] = overrides.preferred_upper_propellant  # upper stage preference
    isps = [ISP_TABLE.get(p, 300.0) for p in props]
    frac = mass_fractions_for_small_lifter(n)

    # 3) liftoff mass & payload
    if overrides.target_payload_leo_kg:
        # if target payload given, back out a liftoff mass using baseline payload fraction
        baseline_frac = 0.0025
        liftoff_mass_kg = max(1000.0, overrides.target_payload_leo_kg / baseline_frac)
        payload_leo_kg = overrides.target_payload_leo_kg
    else:
        liftoff_mass_kg = max(1000.0, estimate_liftoff_mass(height_m, diameter_m))
        payload_leo_kg = max(50.0, 0.0025 * liftoff_mass_kg)

    # 4) distribute mass by stage roughly proportional to length
    length_sum = sum(stage_lengths)
    stage_masses = [liftoff_mass_kg * (L/length_sum) for L in stage_lengths]

    stages: List[StageEstimate] = []
    max_d = diameter_m
    for i in range(n):
        s_mass = stage_masses[i]
        sf, pf = frac[i]
        structural = s_mass * sf
        propellant = s_mass * pf
        isp = isps[i]
        prop = props[i]

        # thrust target: ensure first stage can lift full stack, uppers sized lighter
        if i == 0:
            target_mass_for_thrust = liftoff_mass_kg
        else:
            target_mass_for_thrust = s_mass
        thrust_kN = thrust_for_twr(target_mass_for_thrust, 1.4 if i == 0 else 0.9)

        stages.append(StageEstimate(
            stage=i+1,
            length_m=stage_lengths[i],
            diameter_m=diameter_m if i == 0 else max(0.6*diameter_m, 0.5*diameter_m),
            propellant=prop,
            isp_s=isp,
            structural_mass_kg=structural,
            propellant_mass_kg=propellant,
            engine_thrust_kN=thrust_kN
        ))

    notes = "Heuristic draft from canvas bounds and user scale. Tune in UI and rerun."
    return RocketSpecDraft(
        stages=stages,
        total_height_m=height_m,
        max_diameter_m=max_d,
        liftoff_mass_kg=liftoff_mass_kg,
        payload_leo_kg=payload_leo_kg,
        notes=notes
    )

# ------------------ Simple Mission Plan ------------------
class MissionTarget(str, Enum):
    LEO = "LEO"
    TLI = "TLI"


class Leg(BaseModel):
    name: str
    delta_v_ms: float

class MissionPlan(BaseModel):
    target: MissionTarget
    legs: List[Leg]
    delta_v_total_ms: float
    advisory: str

def mission_plan(spec: RocketSpecDraft, target: MissionTarget = MissionTarget.LEO) -> MissionPlan:
    """
    Super-simple Δv budget. Later, replace with Orekit/poliastro.
    """
    if target == MissionTarget.LEO:
        legs = [Leg(name="Ascent to LEO", delta_v_ms=9400)]
        advisory = "Typical ~9.4 km/s budget to 200–400 km LEO including losses."
    else:
        legs = [
            Leg(name="Ascent to LEO", delta_v_ms=9400),
            Leg(name="Trans-Lunar Injection (TLI)", delta_v_ms=3100)
        ]
        advisory = "Rough LEO (~9.4 km/s) + TLI (~3.1 km/s). Real values depend on trajectory and margins."

    total = sum(l.delta_v_ms for l in legs)
    return MissionPlan(target=target, legs=legs, delta_v_total_ms=total, advisory=advisory)

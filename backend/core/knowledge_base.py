from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from pydantic import BaseModel, Field

# --------- Data Models ---------
class EngineSpec(BaseModel):
    stage: int
    type: str
    count: int
    thrust_sea_level_kN: Optional[float] = None
    thrust_vac_kN: Optional[float] = None
    isp_s: Optional[float] = None

class RocketSpec(BaseModel):
    id: str
    name: str
    manufacturer: Optional[str] = None
    stages: Optional[int] = None
    height_m: Optional[float] = None
    diameter_m: Optional[float] = None
    liftoff_mass_t: Optional[float] = None
    payload_leo_kg: Optional[float] = None
    payload_gto_kg: Optional[float] = None
    payload_tli_kg: Optional[float] = None
    engines: List[EngineSpec] = Field(default_factory=list)
    propellants: List[str] = Field(default_factory=list)
    reusable: Optional[bool] = None
    notes: Optional[str] = None

# --------- Loader / Search ---------
class KnowledgeBase:
    def __init__(self, data_path: Optional[Path] = None):
        root = Path(__file__).resolve().parents[2]
        self.data_path = data_path or (root / "data" / "aerospace_specs" / "rockets.json")
        self._cache: List[RocketSpec] = []
        self._by_id: Dict[str, RocketSpec] = {}

    def load(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"KB file not found: {self.data_path}")
        with self.data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._cache = [RocketSpec(**item) for item in raw]
        self._by_id = {r.id: r for r in self._cache}

    def all(self) -> List[RocketSpec]:
        if not self._cache:
            self.load()
        return list(self._cache)

    def get(self, rocket_id: str) -> Optional[RocketSpec]:
        if not self._by_id:
            self.load()
        return self._by_id.get(rocket_id)

    def search(self, q: str, limit: int = 10) -> List[RocketSpec]:
        """
        Super simple keyword search across id/name/manufacturer/notes/propellants.
        Case-insensitive; returns up to 'limit' matches, ordered by a naive score.
        """
        if not self._cache:
            self.load()
        ql = q.lower().strip()
        if not ql:
            return []
        scored: List[tuple[int, RocketSpec]] = []
        for r in self._cache:
            hay = " ".join([
                r.id or "",
                r.name or "",
                r.manufacturer or "",
                " ".join(r.propellants or []),
                r.notes or ""
            ]).lower()
            score = hay.count(ql)
            # lightweight additional hits on tokens
            for token in ql.split():
                score += hay.count(token)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

# Singleton-ish access
_kb = KnowledgeBase()

def get_kb() -> KnowledgeBase:
    return _kb

# --------- Optional: prompt helper for LLM grounding ---------
def format_rocket_brief(r: RocketSpec) -> str:
    engines_str = "; ".join(
        [f"Stage {e.stage}: {e.count}× {e.type} (Isp={e.isp_s or 'n/a'} s)" for e in r.engines]
    )
    payloads = []
    if r.payload_leo_kg: payloads.append(f"LEO={r.payload_leo_kg} kg")
    if r.payload_gto_kg: payloads.append(f"GTO={r.payload_gto_kg} kg")
    if r.payload_tli_kg: payloads.append(f"TLI={r.payload_tli_kg} kg")
    payload_str = ", ".join(payloads) if payloads else "n/a"
    return (
        f"{r.name} by {r.manufacturer or 'n/a'} — stages={r.stages}, "
        f"H={r.height_m} m, D={r.diameter_m} m, liftoff mass={r.liftoff_mass_t} t, "
        f"payloads({payload_str}); engines: {engines_str}; propellants: {', '.join(r.propellants)}. "
        f"Reusable={r.reusable}. Notes: {r.notes or '—'}"
    )

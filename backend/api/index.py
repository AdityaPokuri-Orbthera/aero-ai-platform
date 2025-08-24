from backend.config.env import load_env
load_env()

from fastapi import FastAPI

from backend.api.routes.ai_routes import router as ai_router
from backend.api.routes.kb_routes import router as kb_router
from backend.api.routes.specs_routes import router as specs_router
from backend.api.routes.mission_routes import router as mission_router
from backend.api.routes.concept_routes import router as concept_router
from backend.api.routes.advisor_routes import router as advisor_router

app = FastAPI(title="Aero-AI Backend")

@app.get("/health")
async def health():
    return {"status": "ok"}

# Include routers *after* the app is defined
app.include_router(ai_router)
app.include_router(kb_router)
app.include_router(specs_router)
app.include_router(mission_router)
app.include_router(concept_router)
app.include_router(advisor_router)
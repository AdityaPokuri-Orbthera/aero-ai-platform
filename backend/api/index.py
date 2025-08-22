from backend.config.env import load_dotenv  # ensures .env is loaded before other imports
from fastapi import FastAPI
from backend.api.routes.ai_routes import router as ai_router

load_dotenv()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(ai_router)

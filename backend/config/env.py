import os
from pathlib import Path
from dotenv import load_dotenv as dotenv_load

def load_env() -> None:
    """
    Load environment variables from the project root .env file.
    """
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        dotenv_load(dotenv_path=env_path, override=False)

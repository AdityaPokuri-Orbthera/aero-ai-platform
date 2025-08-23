# backend/core/prompts.py
from backend.core.types import Message

def system_for_rocketry() -> Message:
    return Message(
        role="system",
        content=("You are an aerospace design assistant. Be precise, cite equations, "
                 "and return JSON blocks when asked for specs.")
    )

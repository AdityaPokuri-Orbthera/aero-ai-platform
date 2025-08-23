# backend/tests/test_orch.py
import os, pytest
from backend.core.orchestrator import build_orchestrator
from backend.core.types import Message, LLMRequest

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_basic_chat():
    orch = build_orchestrator()
    req = LLMRequest(messages=[
        Message(role="system", content="You are terse."),
        Message(role="user", content="Say 'ok'.")
    ], max_tokens=10)
    resp = orch.send_query(req)
    assert isinstance(resp.text, str) and len(resp.text) > 0

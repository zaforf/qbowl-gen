import os
import asyncio
import pytest
from main import Brain, MODEL_ARCHITECT, MODEL_WRITER, MODEL_JUDGE
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.asyncio
async def test_model_connectivity():
    brain = Brain()
    # Test Architect (Gemma 4 31B)
    try:
        res = brain.client.models.generate_content(
            model=MODEL_ARCHITECT, 
            contents="Ping"
        )
        assert res.text is not None
    except Exception as e:
        pytest.fail(f"Architect connectivity failed: {e}")

    # Test Writer (Gemini 3.1 Flash)
    try:
        res = brain.client.models.generate_content(
            model=MODEL_WRITER, 
            contents="Ping"
        )
        assert res.text is not None
    except Exception as e:
        pytest.fail(f"Writer connectivity failed: {e}")

    # Test Judge (Gemini 2.5 Flash-Lite)
    try:
        res = brain.client.models.generate_content(
            model=MODEL_JUDGE, 
            contents="Ping"
        )
        assert res.text is not None
    except Exception as e:
        pytest.fail(f"Judge connectivity failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_model_connectivity())

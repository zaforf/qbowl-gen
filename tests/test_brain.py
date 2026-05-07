import os
import asyncio
import pytest
from main import Brain, MODEL_ARCHITECT, MODEL_WRITER, MODEL_JUDGE
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.asyncio
async def test_model_connectivity():
    brain = Brain()
    # Test Architect (Gemini)
    try:
        res = brain.gemini.models.generate_content(
            model=MODEL_ARCHITECT, 
            contents="Ping"
        )
        assert res.text is not None
    except Exception as e:
        pytest.fail(f"Architect (Gemini) connectivity failed: {e}")

    # Test Writer (Groq)
    try:
        res = brain.groq.chat.completions.create(
            model=MODEL_WRITER,
            messages=[{"role": "user", "content": "Ping"}]
        )
        assert res.choices[0].message.content is not None
    except Exception as e:
        pytest.fail(f"Writer (Groq) connectivity failed: {e}")

    # Test Judge (Groq)
    try:
        res = brain.groq.chat.completions.create(
            model=MODEL_JUDGE,
            messages=[{"role": "user", "content": "Ping"}]
        )
        assert res.choices[0].message.content is not None
    except Exception as e:
        pytest.fail(f"Judge (Groq) connectivity failed: {e}")

@pytest.mark.asyncio
async def test_full_pipeline():
    brain = Brain()
    # 1. Generate topics
    topics = await brain.generate_topic_list("Computer Science")
    assert len(topics) > 0
    
    # 2. Generate clue from first topic
    topic_data = topics[0]
    answer, clue = await brain.generate_clue_from_facts(topic_data)
    assert answer != "Error"
    assert len(clue) > 10
    
    # 3. Validate answer
    is_correct, feedback = await brain.validate_answer(answer, answer, clue)
    assert is_correct is True
    assert "Perfect" in feedback or "CORRECT" in feedback.upper()

if __name__ == "__main__":
    async def run_all():
        print("Running Connectivity Test...")
        try:
            await test_model_connectivity()
            print("✅ Connectivity passed")
        except Exception as e:
            print(f"❌ Connectivity failed: {e}")

        print("Running Pipeline Test...")
        try:
            await test_full_pipeline()
            print("✅ Pipeline passed")
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")

    asyncio.run(run_all())

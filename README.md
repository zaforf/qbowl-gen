# qbowl-gen

AI-powered Quizbowl engine that generates pyramidal clues in real-time.

## Architecture

- **Tiered AI**: Architect (topic/fact generation) $\rightarrow$ Writer (pyramidal clue generation) $\rightarrow$ Judge (semantic validation).
- **Model Fallbacks**: Sequential fallback chains to maintain availability during API rate limits.
- **Zero-Latency**: Background generation queue ensures clues are ready before the current round ends.
- **Validation**: Two-stage judge using fast token-set matching and strict LLM semantic checks.

## Running Locally

1. Create `.env` with `GEMINI_API_KEY`, `GROQ_API_KEY`, and `API_SHARED_TOKEN`.
2. `pip install -r requirements.txt`
3. `python main.py`

## Controls
- **Space**: Buzz
- **N**: Next
- **Enter**: Submit / Start

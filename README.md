# qbowl-gen

Inspired by ProtoBowl and Qbreader, this is an AI-powered Quizbowl engine that generates pyramidal clues in real-time.

## Architecture

- **Tiered AI Pipeline**:
    - **Architect**: A high-capability model providing grounded, obscure facts.
    - **Writer**: A smaller, faster model that transforms Architect fact-sheets into pyramidal clues, ensuring high responsiveness.
    - **Judge**: A two-stage system (token-set matching $\rightarrow$ LLM semantic check) for fast, accurate validation.
- **Responsiveness & Pipeline**:
    - **Bootstrap**: The Writer generates the first few clues independently so the game starts instantly.
    - **Steady State**: The Architect generates batches of 3 ideas and facts in parallel with the Writer, filling a pre-generation queue.
    - **Fallback**: If the queue is exhausted, the fast Writer generates clues directly to prevent latency.
- **Model Fallbacks**: Sequential fallback chains to maintain availability during API rate limits.

## Running Locally

1. Create `.env` with:
   - `GEMINI_API_KEY`, `GROQ_API_KEY`
   - `API_SHARED_TOKEN` (password to play)
2. `pip install -r requirements.txt`
3. `python main.py`

## Controls
- **Space**: Buzz
- **N**: Next
- **S**: Skip tossup
- **T**: Chat
- **Enter**: Submit / Start

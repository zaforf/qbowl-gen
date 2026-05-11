# qbowl-gen

Fast, infinite tossup generation on any topic via LLMs, inspired by Protobowl and Qbreader.

## Architecture

- **Tiered AI Pipeline**:
    - **Architect**: High-capability model providing grounded, obscure facts.
    - **Writer**: Smaller, faster model that transforms Architect fact-sheets into pyramidal clues for fast delivery.
    - **Judge**: Two-stage system (word-set matching before semantic check) for accurate validation.
- **Pipeline Logic**:
    - **Bootstrap**: Writer generates the first few clues independently so the game starts without delay.
    - **Steady State**: Architect generates batches of 3 ideas and facts in parallel with the Writer to fill a pre-generation queue.
    - **Fallback**: If the queue nears exhaustion, the fast Writer generates clues directly to prevent buffering.
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

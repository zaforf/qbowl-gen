# qbowl-gen

A high-performance, AI-driven Quizbowl engine that generates pyramidal clues in real-time with minimal latency.

## Architecture

- **Tiered AI Pipeline**:
    - **Architect**: Handles high-level topic selection and generates detailed fact sheets to ensure academic depth.
    - **Writer**: Converts fact sheets into "pyramided" tossups (obscure $\rightarrow$ common) using a dynamic fast/slow model selection based on queue pressure.
    - **Judge**: A two-stage validation system using fast token-set matching followed by a strict LLM semantic check to ensure accuracy without being overly rigid.
- **Zero-Latency Engine**: Employs an asynchronous background generator and a pre-generation queue to ensure the next clue is ready before the current one ends.
- **Robust Model Fallbacks**: Implements sequential fallback chains to maintain availability during API rate limits or outages.
- **Stateful Game Loop**: A precise state machine managing the transition from streaming $\rightarrow$ buzz window $\rightarrow$ answer phase $\rightarrow$ reveal.
- **Global Deduplication**: Persists used topics to prevent repetition across different game sessions.

## Running Locally

1. **Environment**: Create a `.env` file:
   ```env
   GEMINI_API_KEY=your_key
   GROQ_API_KEY=your_key
   API_SHARED_TOKEN=your_secret_token
   ```

2. **Install**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch**:
   ```bash
   python main.py
   ```

## Controls
- **Space**: Buzz in.
- **N**: Advance to next question (after reveal).
- **Enter**: Submit answer / Start game.

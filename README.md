# qbowl-gen

An AI-powered Quizbowl game that generates diverse, pyramidal clues in real-time.

## Architecture

- **Tiered AI Engine**: 
    - **Architect** (`llama-3.3-70b-versatile`): Generates a high-entropy list of topics and detailed fact sheets.
    - **Writer** (`llama-3.3-70b-versatile`): Converts fact sheets into pyramidal clues (Hard $\rightarrow$ Medium $\rightarrow$ Easy).
    - **Judge** (`llama-3.1-8b-instant`): Performs low-latency semantic validation of player guesses.
- **Zero-Latency Pipeline**: Uses an asynchronous topic queue and background generation to ensure clues are ready before the previous round ends.
- **Real-time Sync**: FastAPI + WebSockets for instantaneous buzzing and synchronized state across all clients.

## Running Locally

1. **Environment**: Create a `.env` file with:
   ```env
   GEMINI_API_KEY=your_key
   GROQ_API_KEY=your_key
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch**:
   ```bash
   python main.py
   ```

import os
import json
import asyncio
import time
from typing import List, Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Tiered Model Selection
MODEL_ARCHITECT = "llama-3.3-70b-versatile" # Groq: High-entropy topic & fact generation
MODEL_WRITER = "llama-3.3-70b-versatile"     # Groq: Fast, structured clue writing
MODEL_JUDGE = "llama-3.1-8b-instant"         # Groq: Ultra-low latency validation

SYSTEM_PROMPT_ARCHITECT = """
You are the "Quizbowl Curator." Your goal is to generate a high-entropy, diverse list of topics for a Quizbowl game.
When given a general topic (e.g., "Computer Science"), you must produce 3 distinct, non-overlapping specific entities, theorems, people, or concepts.
For each entity, provide a "Fact Sheet": 3-5 bullet points of increasing obscurity (from very niche to very common).

Format:
TOPIC: [Entity Name]
FACTS:
- [Niche Fact 1]
- [Niche Fact 2]
- [Common Fact 3]
...
(Repeat for 3 topics)
"""

SYSTEM_PROMPT_WRITER = """
You are the "AI Qbreader," an expert Quizbowl Clue Writer.
Your task is to take a Fact Sheet and turn it into a properly "pyramided" clue.

### PYRAMIDING RULES
1. Start with the most obscure facts (The Lead-In).
2. Move to enthusiast-level facts (The Bridge).
3. End with the most iconic, widely known facts (The Giveaway).
4. NEVER mention the answer within the clue text.
5. No filler phrases ("This person was...", "The following is..."). Start directly with facts.

Output format:
ANSWER: [The precise answer]
CLUE: [The full pyramidal text]
"""

SYSTEM_PROMPT_JUDGE = """
You are the "Quizbowl Judge." Determine if a USER_GUESS matches the target ANSWER.

STRICT RULES:
- The guess must name the *specific* answer, not a parent category or general description.
  e.g. ANSWER="KMP algorithm" → "string matching algorithm" is WRONG; "Knuth-Morris-Pratt" is CORRECT.
  e.g. ANSWER="quicksort" → "sorting algorithm" is WRONG.
- Accept common abbreviations, alternate spellings, or unambiguously shortened names of the SAME specific thing.
- Reject vague, overspecified, or category-level guesses even if technically related.

Output only:
RESULT: [CORRECT/INCORRECT]
FEEDBACK: [One sentence. If wrong, say what was expected.]
"""

class Brain:
    def __init__(self):
        self.gemini = gemini_client
        self.groq = groq_client

    async def generate_topic_list(self, general_topic: str):
        response = self.groq.chat.completions.create(
            model=MODEL_ARCHITECT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ARCHITECT},
                {"role": "user", "content": f"General Topic: {general_topic}. Generate 3 distinct Quizbowl topics with fact sheets."}
            ],
            temperature=0.7
        )

        raw_text = response.choices[0].message.content
        topics = []
        parts = raw_text.split("TOPIC:")
        for part in parts[1:]:
            lines = part.split("\n")
            name = lines[0].strip()
            facts = []
            for line in lines[1:]:
                if line.strip().startswith("-"):
                    facts.append(line.strip("- ").strip())
            topics.append({"name": name, "facts": "\n".join(facts)})
        return topics

    async def generate_clue_from_facts(self, topic_data: dict):
        response = self.groq.chat.completions.create(
            model=MODEL_WRITER,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_WRITER},
                {"role": "user", "content": f"TOPIC: {topic_data['name']}\nFACTS: {topic_data['facts']}"}
            ],
            temperature=0.3
        )
        text = response.choices[0].message.content
        try:
            answer = text.split("ANSWER:")[1].split("CLUE:")[0].strip()
            clue = text.split("CLUE:")[1].strip()
            return answer, clue
        except IndexError:
            return "Error", "Could not parse clue from Groq Writer."

    async def validate_answer(self, answer: str, guess: str, clue: str):
        response = self.groq.chat.completions.create(
            model=MODEL_JUDGE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                {"role": "user", "content": f"ANSWER: {answer} | USER_GUESS: {guess} | CLUE: {clue}"}
            ],
            temperature=0
        )
        text = response.choices[0].message.content
        try:
            result = text.split("RESULT:")[1].split("FEEDBACK:")[0].strip()
            feedback = text.split("FEEDBACK:")[1].strip()
            return result == "CORRECT", feedback
        except IndexError:
            return False, "Validation error from Groq Judge."

class GameState:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.buzzer_locked = False
        self.winner = None
        self.current_answer = None
        self.current_clue = None
        self.streaming_task: Optional[asyncio.Task] = None
        self.bg_generator_task: Optional[asyncio.Task] = None
        
        # Scoring
        self.scores: Dict[str, int] = {}
        
        # Topic Queue Infrastructure
        self.topic_queue: List[Dict] = []
        self.pregenerated_clues: asyncio.Queue = asyncio.Queue()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

    def reset_buzzer(self):
        self.buzzer_locked = False
        self.winner = None

    async def stop_streaming(self):
        if self.streaming_task:
            self.streaming_task.cancel()
            self.streaming_task = None

game = GameState()
brain = Brain()
app = FastAPI()

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await game.connect(websocket)
    # Init score
    if client_id not in game.scores:
        game.scores[client_id] = 0

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start":
                topic = message.get("topic", "General Knowledge")
                await game.stop_streaming()
                game.reset_buzzer()
                
                game.topic_queue = []
                while not game.pregenerated_clues.empty():
                    game.pregenerated_clues.get_nowait()
                
                await game.broadcast({"type": "status", "text": "Curating diverse topics..."})
                topics = await brain.generate_topic_list(topic)
                game.topic_queue = topics
                
                if game.bg_generator_task:
                    game.bg_generator_task.cancel()
                game.bg_generator_task = asyncio.create_task(background_generator())
                
                game.streaming_task = asyncio.create_task(stream_next_clue(client_id))
            
            elif message.get("type") == "next":
                await game.stop_streaming()
                game.reset_buzzer()
                game.streaming_task = asyncio.create_task(stream_next_clue(client_id))

            elif message.get("type") == "buzz":
                if not game.buzzer_locked:
                    game.buzzer_locked = True
                    game.winner = client_id
                    # Track when the buzz happened to calculate point value
                    game.buzz_time = time.time()
                    await game.broadcast({"type": "lock", "winner": client_id})
                    await game.stop_streaming()
                else:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Too late!"}))
            
            elif message.get("type") == "submit_answer":
                if game.winner == client_id:
                    guess = message.get("guess", "")
                    is_correct, feedback = await brain.validate_answer(
                        game.current_answer, guess, game.current_clue
                    )
                    
                    # Calculate points based on how far the clue was streamed
                    points = 0
                    if is_correct:
                        # Simplification: points based on clue length streamed
                        # In a real scenario, we'd track the current word index.
                        # Here we'll just use a fixed set of values for this phase.
                        points = 10
                    else:
                        points = -5
                    
                    game.scores[client_id] += points
                    
                    await game.broadcast({
                        "type": "reveal", 
                        "correct": is_correct, 
                        "answer": game.current_answer, 
                        "feedback": feedback,
                        "winner": client_id,
                        "scores": game.scores
                    })
                    game.reset_buzzer()
            
            elif message.get("type") == "reset":
                game.reset_buzzer()
                # Clear scores for reset
                game.scores = {}
                await game.broadcast({"type": "reset"})

    except WebSocketDisconnect:
        game.disconnect(websocket)

async def background_generator():
    try:
        while True:
            if game.topic_queue:
                topic_data = game.topic_queue.pop(0)
                answer, clue = await brain.generate_clue_from_facts(topic_data)
                await game.pregenerated_clues.put((answer, clue))
            else:
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

async def stream_next_clue(client_id: str):
    try:
        attempts = 0
        while game.pregenerated_clues.empty() and attempts < 50:
            await asyncio.sleep(0.1)
            attempts += 1
            
        if game.pregenerated_clues.empty():
            await game.broadcast({"type": "error", "message": "No more questions in the queue."})
            return

        answer, clue = await game.pregenerated_clues.get()
        game.current_answer = answer
        game.current_clue = clue
        
        words = clue.split()
        for word in words:
            if game.buzzer_locked:
                break
            await game.broadcast({"type": "clue_chunk", "text": word})
            await asyncio.sleep(0.20)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error streaming clue: {e}")
        await game.broadcast({"type": "error", "message": "AI failed to stream clue."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

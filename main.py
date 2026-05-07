import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-3.1-flash"

SYSTEM_PROMPT = """
You are the "AI Qbreader," an expert Quizbowl Game Master and Clue Writer. Your goal is to create intellectually stimulating, factually accurate, and properly "pyramided" clues.

### CORE CONCEPT: PYRAMIDING
A pyramidal clue starts with the most obscure, difficult information and progresses toward the most common, easy information.

Structure of a clue:
1. The Lead-In (Difficult): 2-3 sentences of highly specific, niche, or technical facts.
2. The Bridge (Medium): 2-3 sentences of facts known to enthusiasts.
3. The Giveaway (Easy): 1-2 sentences of iconic, widely known facts.

### OPERATIONAL MODES

#### MODE: GENERATE
When asked to generate a clue for a [TOPIC]:
- DO NOT mention the answer within the clue text.
- Maintain a strict difficulty gradient (Hard -> Medium -> Easy).
- Output the result in the following format:
  ANSWER: [The precise answer]
  CLUE: [The full pyramidal text]

#### MODE: VALIDATE
When provided with an [ANSWER], a [USER_GUESS], and a [CLUE]:
- Determine if the USER_GUESS is correct.
- Be "Quizbowl Lenient": Accept common synonyms or slightly incomplete but unambiguous names.
- Output only:
  RESULT: [CORRECT/INCORRECT]
  FEEDBACK: [Briefly explain why or "Perfect!"]

### CONSTRAINTS
- Factuality is paramount. Do not hallucinate.
- No "filler" phrases like "This person was...". Start directly with the facts.
- Ensure the "Giveaway" is at the very end.
"""

class Brain:
    async def generate_clue(self, topic: str):
        prompt = f"MODE: GENERATE | TOPIC: {topic}"
        # New SDK uses a config object for system instructions
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        text = response.text
        
        try:
            answer_part = text.split("ANSWER:")[1].split("CLUE:")[0].strip()
            clue_part = text.split("CLUE:")[1].strip()
            return answer_part, clue_part
        except IndexError:
            print(f"Parsing error. Raw response: {text}")
            return "Error", "Could not parse clue."

    async def validate_answer(self, answer: str, guess: str, clue: str):
        prompt = f"MODE: VALIDATE | ANSWER: {answer} | USER_GUESS: {guess} | CLUE: {clue}"
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        text = response.text
        
        try:
            result = text.split("RESULT:")[1].split("FEEDBACK:")[0].strip()
            feedback = text.split("FEEDBACK:")[1].strip()
            return result == "CORRECT", feedback
        except IndexError:
            return False, "Validation error."

class GameState:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.buzzer_locked = False
        self.winner = None
        self.current_answer = None
        self.current_clue = None
        self.streaming_task: Optional[asyncio.Task] = None

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
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start":
                topic = message.get("topic", "General Knowledge")
                await game.stop_streaming()
                game.reset_buzzer()
                
                game.streaming_task = asyncio.create_task(stream_clue(client_id, topic))
            
            elif message.get("type") == "buzz":
                if not game.buzzer_locked:
                    game.buzzer_locked = True
                    game.winner = client_id
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
                    await game.broadcast({
                        "type": "reveal", 
                        "correct": is_correct, 
                        "answer": game.current_answer, 
                        "feedback": feedback,
                        "winner": client_id
                    })
                    game.reset_buzzer()
            
            elif message.get("type") == "reset":
                game.reset_buzzer()
                await game.broadcast({"type": "reset"})

    except WebSocketDisconnect:
        game.disconnect(websocket)

async def stream_clue(client_id: str, topic: str):
    try:
        answer, clue = await brain.generate_clue(topic)
        game.current_answer = answer
        game.current_clue = clue
        
        sentences = clue.split(". ")
        for sentence in sentences:
            if game.buzzer_locked:
                break
            s = sentence.strip()
            if s and not s.endswith("."):
                s += "."
            
            await game.broadcast({"type": "clue_chunk", "text": s})
            await asyncio.sleep(2.5)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error streaming clue: {e}")
        await game.broadcast({"type": "error", "message": "AI failed to generate clue."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

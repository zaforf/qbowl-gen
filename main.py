from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import List
import json

app = FastAPI()

class GameState:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.buzzer_locked = False
        self.winner = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

    def reset_buzzer(self):
        self.buzzer_locked = False
        self.winner = None

game = GameState()

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
            
            if message.get("type") == "buzz":
                if not game.buzzer_locked:
                    game.buzzer_locked = True
                    game.winner = client_id
                    await game.broadcast({"type": "lock", "winner": client_id})
                else:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Too late!"}))
            
            elif message.get("type") == "reset":
                game.reset_buzzer()
                await game.broadcast({"type": "reset"})

    except WebSocketDisconnect:
        game.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

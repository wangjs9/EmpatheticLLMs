# coding: utf-8
import json
import time
import yaml
from pathlib import Path
from typing import List, Dict
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from interactive_models.interact import EmpatheticLLM

app = FastAPI(title="7/24 Engaging Empathetic LLM Listener", version="0.1.0")
app.mount("/static", StaticFiles(directory="./interactive_models/static", html=True), name="static")

args = yaml.safe_load(Path('interactive_models/user_simulator.yaml').read_text())
model = EmpatheticLLM(args)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(path="./interactive_models/static/index.html", media_type="text/html")


# Chatbot logic
def get_bot_response(user_message: str) -> str:
    """
    Simple chatbot logic to return a predefined response based on the user's message.
    """
    responses = {
        "hello": "Hi there! How can I assist you?",
        "how are you": "I'm doing great! Thank you for asking.",
        "bye": "Goodbye! Have a wonderful day!",
    }
    # Default response if the message is not recognized
    return responses.get(user_message.lower(), "Sorry, I didn't understand that.")


# Pydantic model for request validation
class ChatRequest(BaseModel):
    timestamp: int
    message: str
    dialogues: List[Dict[str, str]]


# POST /chat endpoint
@app.post("/event/chat")
async def chat(request: ChatRequest):
    """
    Receives a user message, processes it, and returns a chatbot response.
    """
    
    user_message = request.message.strip()
    dialogues = request.dialogues
    dialogues.append({"role": "user", "content": user_message})
    if not user_message:
        bot_response = "Please send a valid message."
        status_code = 400
    else:
        # Get the bot's response
        bot_response = get_bot_response(user_message)
        bot_response = model.__respond__(dialogues)
        # bot_response = "I'm sorry, I'm not sure how to respond to that."
        status_code = 200
    
    dialogues.append({"role": "assistant", "content": bot_response})
    # Return the bot's response as JSON
    json.dump(
        dialogues,
        open(f'./interactive_models/dialogues/{str(request.timestamp)}.json', 'w', encoding='utf-8'),
        indent=2
    )
    return JSONResponse({"message": bot_response, "dialogues": dialogues}, status_code)


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8000)

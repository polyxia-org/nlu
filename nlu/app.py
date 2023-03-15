import os

import openai
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from model import NeuralNet
from pydantic import BaseModel
from utils import bag_of_words, tokenize

from nlu.functions import INTENTS_HANDLER
from nlu.functions.chatgpt import ChatBot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
app = FastAPI()


class NluPayload(BaseModel):
    input_text: str


class NluResponse(BaseModel):
    intent: str
    response: str


@app.post("/nlu")
async def nlu(payload: NluPayload):
    user_input = payload.input_text
    print(f'Retrieving intent of "{user_input}"')
    X = bag_of_words(tokenize(user_input), all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    out = model(X)
    _, predicted = torch.max(out, dim=1)

    intent = tags[predicted.item()]

    probs = torch.softmax(out, dim=1)
    prob = probs[0][predicted.item()]

    print(prob.item())
    if prob.item() > 0.90:
        print(f"Detected intent: {intent}")
        return NluResponse(intent=intent, response=INTENTS_HANDLER.get(intent)())
    else:
        print(
            "Cannot find an intent that match your query sending query to the chatbot"
        )
        llm = ChatBot(
            "You are a helpful voice assistant like Alexa, or Google Assistant, named Polyxia, your answers are precise and concise."
        )
        return NluResponse(
            intent="Chatbot", response=INTENTS_HANDLER.get("Chatbot")(llm, user_input)
        )


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    uvicorn.run(app, host="0.0.0.0", port=8080)

import os
from contextlib import asynccontextmanager
import openai
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from nlu.brain import get_user_intent
from utils import get_intents
import logging
from nlu.functions import INTENTS_HANDLER
from nlu.functions.chatgpt import ChatBot


CACHED_INTENTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    
    # Load intents in cache
    print("Loading intents in cache")
    global CACHED_INTENTS
    CACHED_INTENTS = get_intents("nlu/intents")
    print(CACHED_INTENTS)
    yield
    # Clean up the ML models and release the resources



logger = logging.getLogger(__name__)
app = FastAPI(lifespan=lifespan)



class NluPayload(BaseModel):
    input_text: str


class NluResponse(BaseModel):
    intent: str
    response: str


@app.post("/nlu")
async def nlu(payload: NluPayload):
    user_input = payload.input_text
    print(f'Retrieving intent of "{user_input}"')
    logger.info(f'Retrieving intent of "{user_input}"')
    intent, prob = get_user_intent(user_input, CACHED_INTENTS)

    if prob > 0.75:
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

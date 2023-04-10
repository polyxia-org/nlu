import openai
from fastapi import APIRouter

from nlu.brain import get_user_intent
from nlu.functions.chatgpt import ChatBot
from nlu.functions.intents import INTENTS_HANDLER
from nlu.schemas.nlu import NluPayload, NluResponse

router = APIRouter()


@router.post("/")
async def nlu(payload: NluPayload):
    user_input = payload.input_text
    print(f'Retrieving intent of "{user_input}"')
    intent, prob = await get_user_intent(user_input)
    print(f"Detected intent: {intent} {prob}")
    if intent is not None:
        # TODO: call morty serverless instance
        return NluResponse(intent=intent, response=INTENTS_HANDLER.get(intent)())
    else:
        if openai.api_key is None:
            return NluResponse(
                intent="unknown",
                response="I am sorry, I do not understand what you are saying.",
            )
        # TODO: use our own chatbot
        llm = ChatBot(
            "You are a helpful voice assistant like Alexa, or Google Assistant, named Polyxia, your answers are precise and concise."
        )
        return NluResponse(
            intent="Chatbot", response=INTENTS_HANDLER.get("Chatbot")(llm, user_input)
        )

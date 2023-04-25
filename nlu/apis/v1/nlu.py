import os

import openai
import requests
from fastapi import APIRouter, HTTPException

from nlu.brain import get_params, get_user_intent
from nlu.functions.chatgpt import ChatBot
from nlu.functions.intents import INTENTS_HANDLER
from nlu.schemas.nlu import NluPayload, NluResponse

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def nlu(payload: NluPayload):
    user_input = payload.input_text
    logger.info(f'Retrieving intent of "{user_input}"')
    intent, prob = await get_user_intent(user_input)
    logger.info(f"Detected intent: {intent} {prob}")
    if intent is not None:
        params = await get_params(intent, user_input)
        res = requests.get(
            f"{os.getenv('FUNCTIONS_GATEWAY')}/functions/{intent}/invoke", params=params
        )
        if res.ok:
            return NluResponse(intent=intent, response=res.text)
        logger.error(f"Error: {res.text}")
        raise HTTPException(
            status_code=500,
            detail=f"Sorry an error occurred when calling the {intent} function.",
        )
    else:
        if openai.api_key is None:
            raise HTTPException(
                status_code=500,
                detail="I am sorry, I do not understand what you are saying.",
            )
        # TODO: use our own chatbot
        try:
            llm = ChatBot(
                "You are a helpful voice assistant like Alexa, or Google Assistant, named Polyxia, your answers are precise and concise."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Unable to contact the chatbot")
        return NluResponse(intent="chatbot", response=llm.ask(user_input))

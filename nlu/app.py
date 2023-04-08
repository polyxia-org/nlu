import logging
import os

import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from nlu.apis.v1 import api_router

logger = logging.getLogger(__name__)
app = FastAPI()

app.include_router(api_router, prefix="/v1")


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    uvicorn.run(app, host="0.0.0.0", port=8082)

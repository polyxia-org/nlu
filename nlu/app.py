import logging
import os

import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from nlu.apis.v1 import api_router

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)
app = FastAPI()

app.include_router(api_router, prefix="/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)

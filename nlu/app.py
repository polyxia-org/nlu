import logging
import os
import sys

import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from contextlib import asynccontextmanager

from nlu.apis.v1 import api_router
from nlu.database.client import Database

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    try:
        await Database()
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

    if os.getenv("FUNCTIONS_GATEWAY") is None:
        logger.error("FUNCTIONS_GATEWAY is not set")
        sys.exit(1)
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)

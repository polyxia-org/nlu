from fastapi import APIRouter

from . import nlu, skills

api_router = APIRouter()

api_router.include_router(nlu.router, prefix="/nlu", tags=["nlu"])
api_router.include_router(skills.router, prefix="/skills", tags=["skills"])

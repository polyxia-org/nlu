from fastapi import APIRouter, HTTPException

from nlu.crud.skills import CRUDSkills
from nlu.schemas.skills import SkillSchema

router = APIRouter()


@router.post("/", status_code=201)
async def create_skill(payload: SkillSchema) -> SkillSchema:
    existing_skill = await CRUDSkills.get_by_name(payload.intent)
    if existing_skill:
        raise HTTPException(400, "Skill with this intent already exists")
    await CRUDSkills.create(payload.dict())
    return payload


@router.get("/")
async def get_skills() -> list[SkillSchema]:
    return await CRUDSkills.get()


@router.delete("/{intent}", status_code=204)
async def delete_skill(intent: str):
    await CRUDSkills.delete(intent)

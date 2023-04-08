from nlu.database.client import Database
from nlu.schemas.skills import SkillSchema


class CRUDSkills:
    @staticmethod
    async def create(skill: SkillSchema):
        db = await Database()
        await db.session.create("skills", skill)

    @staticmethod
    async def get():
        db = await Database()
        return await db.session.select("skills")

    @staticmethod
    async def get_by_name(name: str):
        db = await Database()
        res = await db.session.query(
            "SELECT * FROM skills WHERE intent = $name", {"name": name}
        )
        return res[0]["result"]

    @staticmethod
    async def delete(name: str):
        db = await Database()
        await db.session.query(
            "DELETE FROM skills WHERE intent = $name", {"name": name}
        )

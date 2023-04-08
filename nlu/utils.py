import json
from pathlib import Path

from nlu.crud.skills import CRUDSkills
from nlu.database.client import Database
from nlu.schemas.skills import SkillSchema


def get_mockup_intents(path="./intents"):
    """Get intents from json files in mockup folder
    example of return format:
    {"sayHello": {
        "utterances": [
            "bonjour",
            "dis bonjour",
            "salut",
            "hello",
            "je te salue",
            "bonsoir",
        ]
    },
    "lightOn": {
        "utterances": [
            "lumière",
            "allume la lumière",
            "aziz lumière",
            "allume la lampe",
        ]
    }}
    """
    intents = {}
    folder_path = Path(path)
    for file_path in folder_path.glob("*.json"):
        with file_path.open("r") as f:
            json_data = json.load(f)
            intents[json_data["intent"]] = {"utterances": json_data["utterances"]}
    return intents


async def get_db_intents() -> list[SkillSchema]:
    """Get intents from database"""
    skills = await CRUDSkills.get()
    return {
        k: v
        for raw_skill in skills
        for k, v in SkillSchema(**raw_skill).to_intent().items()
    }


def slot_uniformization(text):
    """Parse slot filling from text"""
    idx = 0
    res = {}

    for t in text:
        raw = t["entity"].replace("B-", "").replace("I-", "")

        if "B-" in t["entity"] and "▁" in t["word"]:
            idx += 1
            res[f"{raw}|{idx}"] = [t["word"].replace("▁", "")]
        else:
            res[f"{raw}|{idx}"].append(t["word"])

    res = [(r.split("|")[0], "".join(res[r])) for r in res]

    return res

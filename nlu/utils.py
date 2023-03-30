import json
from pathlib import Path


def get_intents(path="./intents"):
    """Get intents from json files in a folder
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
    #TODO: mockup file for now will retrieve data from database later
    intents = {}
    folder_path = Path(path)
    for file_path in folder_path.glob("*.json"):
        with file_path.open("r") as f:
            json_data = json.load(f)
            intents[json_data["intent"]] = {"utterances": json_data["utterances"]}
    return intents

from typing import Optional

from pydantic import BaseModel


class SlotSchema(BaseModel):
    type: str


class SkillSchema(BaseModel):
    intent: str
    utterances: list[str]
    slots: Optional[list[SlotSchema]]

    def to_intent(self):
        return {
            self.intent: {
                "utterances": self.utterances,
                "slots": self.slots,
            }
        }

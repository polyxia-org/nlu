from pydantic import BaseModel


class NluPayload(BaseModel):
    input_text: str


class NluResponse(BaseModel):
    intent: str
    response: str

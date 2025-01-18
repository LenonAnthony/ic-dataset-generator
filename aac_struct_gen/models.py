from pydantic import BaseModel
from typing import List


class SynonymRequest(BaseModel):
    system_prompt: str
    instruction: str


class CardRequest(BaseModel):
    system_prompt: str
    instruction: str


class SynonymResponse(BaseModel):
    input: str
    synonyms: List[str]


class CardResponse(BaseModel):
    input: str
    output: List[str]


class DatasetRow(BaseModel):
    input: str
    output: str

from pydantic import BaseModel
from typing import List

class UserMissings(BaseModel):
    field: str
    missings: List
from pydantic import BaseModel

class UserMissings(BaseModel):
    field: str
    missings: list
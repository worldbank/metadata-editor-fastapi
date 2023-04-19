from pydantic import BaseModel
from typing import List
from src.UserMissings import UserMissings
from src.WeightsColumns import WeightsColumns


class VarInfo(BaseModel):
    file_path: str
    var_names: list
    weights: List[WeightsColumns] = []
    missings: List[UserMissings] = []

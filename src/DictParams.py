from pydantic import BaseModel
from src.UserMissings import UserMissings
from src.WeightsColumns import WeightsColumns

class DictParams(BaseModel):
    file_path: str
    var_names: list = []
    weights: list[WeightsColumns] = []
    missings: list[UserMissings] = []

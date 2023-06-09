from pydantic import BaseModel
from typing import List
from src.UserMissings import UserMissings
from src.WeightsColumns import WeightsColumns

class DictParams(BaseModel):
    file_path: str
    var_names: List = []
    weights: List[WeightsColumns] = []
    missings: dict={} #List[UserMissings] = []
    dtypes: dict = {}
    value_labels: dict = {}
    name_labels: dict = {}
    export_format: str = "csv"

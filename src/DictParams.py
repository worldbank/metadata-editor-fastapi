from pydantic import BaseModel

class WeightsColumns(BaseModel):
    field: str
    weight_field: str
    

class UserMissings(BaseModel):
    field: str
    missings: list


class DictParams(BaseModel):
    file_path: str
    var_names: list = []
    weights: list[WeightsColumns] = []
    missings: list[UserMissings] = []

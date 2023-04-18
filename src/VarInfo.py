from pydantic import BaseModel

class WeightsColumns(BaseModel):
    weight_field: str
    field: str

class UserMissings(BaseModel):
    field: str
    missings: list


class VarInfo(BaseModel):
    file_path: str
    var_names: list
    weights: list[WeightsColumns] = []
    missings: list[UserMissings] = []

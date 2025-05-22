from pydantic import BaseModel

class WeightsColumns(BaseModel):
    field: str
    weight_field: str
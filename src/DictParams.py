from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
from src.UserMissings import UserMissings
from src.WeightsColumns import WeightsColumns


class DictParams(BaseModel):
    """Parameters for generating data dictionary"""

    file_path: str
    var_names: List = []
    weights: List[WeightsColumns] = []

    @field_validator("weights", mode="after")
    @classmethod
    def filter_weights_field_eq_weight_field(cls, v: List[WeightsColumns]) -> List[WeightsColumns]:
        """Remove any weight where field == weight_field (degenerate case)."""
        if not v:
            return v
        return [w for w in v if w.field != w.weight_field]
    missings: Optional[Dict[str, Any]] = {}  # Allow None values
    dtypes: Dict[str, Any] = {}
    value_labels: Dict[str, Any] = {}
    name_labels: Dict[str, Any] = {}
    categorical: List[str] = []
    export_format: str = "csv"
    exclude_fields: List[str] = []  # field names to remove from exported file

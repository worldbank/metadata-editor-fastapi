from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.UserMissings import UserMissings
from src.WeightsColumns import WeightsColumns

class DictParams(BaseModel):
    """Parameters for generating data dictionary"""

    file_path: str
    var_names: List = []
    weights: List[WeightsColumns] = []
    missings: Optional[Dict[str, Any]] = {}  # Allow None values
    dtypes: Dict[str, Any] = {}
    value_labels: Dict[str, Any] = {}
    name_labels: Dict[str, Any] = {}
    export_format: str = "csv"

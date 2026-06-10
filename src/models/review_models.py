from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ReviewSubmitRequest(BaseModel):
    """Body for POST /review/jobs — start an AI metadata review."""

    metadata: Dict[str, Any] = Field(..., description="Metadata document (JSON object) to review")
    manifest_file: Optional[str] = Field(
        None,
        description="YAML manifest filename bundled with ai4data (default: package default)",
    )
    team_preset: Optional[str] = Field(
        None,
        description="AutoGen team: RoundRobinGroupChat, SelectorGroupChat, MagenticOneGroupChat, or Swarm",
    )

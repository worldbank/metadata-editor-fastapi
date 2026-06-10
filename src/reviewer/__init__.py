"""Optional metadata reviewer integration (ai4data PyPI package)."""

from .integration import (
    REVIEWER_AVAILABLE,
    REVIEWER_JOB_TYPE,
    dispose_reviewer_job_if_needed,
    register_reviewer,
)

__all__ = [
    "REVIEWER_AVAILABLE",
    "REVIEWER_JOB_TYPE",
    "dispose_reviewer_job_if_needed",
    "register_reviewer",
]

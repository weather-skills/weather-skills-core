"""Core library for weather skills: CLI, standard-dataset validation, and provenance."""

import warnings

from weather_skills_core.dataset_type import Dataset
from weather_skills_core.decorator import weather_skill
from weather_skills_core.errors import DataError, SkillError, UsageError

# Two library warnings print on every skill run and say nothing about the data:
# pint has no matplotlib to hook into, and zarr notes that consolidated metadata
# is outside the Zarr v3 spec. Set here, before any skill loads pint or zarr.
warnings.filterwarnings("ignore", message=r"Import\(s\) unavailable to set up matplotlib support")
warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")

__all__ = [
    "DataError",
    "Dataset",
    "SkillError",
    "UsageError",
    "weather_skill",
]

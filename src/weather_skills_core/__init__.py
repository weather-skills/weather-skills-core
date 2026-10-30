"""Core library for weather skills: CLI, standard-dataset validation, and provenance."""

import warnings

from weather_skills_core.dataset_type import Dataset
from weather_skills_core.decorator import weather_skill
from weather_skills_core.errors import DataError, SkillError, UsageError

# Two notes every skill run used to print, neither actionable: cf_xarray skips
# its matplotlib hook (skill envs have no matplotlib), and zarr says consolidated
# metadata is not in the v3 spec (skills write and read it on purpose). Matched
# by message, so every other warning still shows. The imports above are lazy,
# so these are in place before cf_xarray or zarr load.
warnings.filterwarnings("ignore", message=r"Import\(s\) unavailable to set up matplotlib support")
warnings.filterwarnings("ignore", message=r"Consolidated metadata is currently not part")

__all__ = [
    "DataError",
    "Dataset",
    "SkillError",
    "UsageError",
    "weather_skill",
]

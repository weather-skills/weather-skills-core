"""Importing the package must not pull in the scientific stack."""

import subprocess
import sys

HEAVY = ("xarray", "numpy", "pandas", "pint", "pint_xarray", "cf_xarray", "shapely", "zarr", "PIL")

LIGHT_MODULES = (
    "weather_skills_core",
    "weather_skills_core.cf",
    "weather_skills_core.dataset_type",
    "weather_skills_core.decorator",
    "weather_skills_core.errors",
    "weather_skills_core.provenance",
    "weather_skills_core.region",
    "weather_skills_core.standard_args",
    "weather_skills_core.standard_dataset",
    "weather_skills_core.standard_utils",
)


def test_core_modules_import_without_heavy_deps():
    code = (
        "import importlib, sys\n"
        f"for name in {LIGHT_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"eager heavy imports: {out}"


def test_routine_warnings_are_silenced_but_others_show(tmp_path):
    """cf_xarray's matplotlib note and zarr's consolidated-metadata note are noise."""
    code = (
        "import warnings\n"
        "import weather_skills_core.units\n"
        "import xarray as xr\n"
        f"p = {str(tmp_path / 'a.zarr')!r}\n"
        "xr.Dataset({'a': ('x', [1.0])}).to_zarr(p, consolidated=True)\n"
        "xr.open_zarr(p, consolidated=True).load()\n"
        "warnings.warn('a real warning')\n"
    )
    err = subprocess.run(
        [sys.executable, "-W", "default", "-c", code], capture_output=True, text=True, check=True
    ).stderr
    assert "matplotlib support" not in err
    assert "Consolidated metadata" not in err
    assert "a real warning" in err

"""Importing the package must not pull in the scientific stack."""

import subprocess
import sys

HEAVY = (
    "xarray",
    "numpy",
    "pandas",
    "pint",
    "pint_xarray",
    "cf_xarray",
    "dask",
    "shapely",
    "zarr",
    "PIL",
)

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
    "weather_skills_core.zarr_layout",
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


def test_help_and_a_run_without_the_flag_do_not_import_dask():
    code = (
        "import sys\n"
        "from weather_skills_core import weather_skill\n"
        "@weather_skill(name='s', version='1', output=False)\n"
        "def s(**kwargs):\n"
        "    pass\n"
        "s([])\n"
        "try:\n"
        "    s(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('dask' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out.splitlines()[-1] == "False"

"""CF helpers used when writing or inspecting standard-dataset Zarrs."""

from __future__ import annotations

from weather_skills_core.errors import DataError
from weather_skills_core.standard_dataset import names_for


def stamp_cf_attrs(ds):
    """Fill missing CF attrs on lat/lon/time coords; leave existing values.

    Uses ``names_for`` aliases (``lat``/``latitude``, ``lon``/``longitude``).
    Does not treat projected ``x``/``y`` as geographic. Returns ``ds``.
    """
    for name in names_for("lat"):
        if name in ds.coords:
            ds[name].attrs.setdefault("standard_name", "latitude")
            ds[name].attrs.setdefault("units", "degrees_north")
            ds[name].attrs.setdefault("axis", "Y")
            break
    for name in names_for("lon"):
        if name in ds.coords:
            ds[name].attrs.setdefault("standard_name", "longitude")
            ds[name].attrs.setdefault("units", "degrees_east")
            ds[name].attrs.setdefault("axis", "X")
            break
    if "time" in ds.coords:
        ds["time"].attrs.setdefault("standard_name", "time")
        ds["time"].attrs.setdefault("axis", "T")
    return ds


def stamp_cf_coords(ds, *, long_names: dict | None = None):
    """Overwrite CF attrs on coords named ``latitude``/``longitude``/``time``.

    Unlike ``stamp_cf_attrs``, this replaces attrs and only those exact names.
    Optional ``long_names`` maps coord → ``long_name`` (setdefault). Returns ``ds``.
    """
    long_names = long_names or {}
    stamps = {
        "latitude": {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
        "longitude": {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
        "time": {"standard_name": "time", "axis": "T"},
    }
    for name, attrs in stamps.items():
        if name in ds.coords:
            ds[name].attrs.update(attrs)
            if name in long_names:
                ds[name].attrs.setdefault("long_name", long_names[name])
    return ds


def drop_dangling_cf_refs(ds):
    """Strip CF reference attrs that name variables ``ds`` lacks.

    Covers ``coordinates``, ``ancillary_variables`` and ``bounds`` (each missing
    name is removed; an emptied attr is deleted) and ``grid_mapping`` (deleted
    whole if any name is missing). Checks attrs and encoding of every variable,
    so a skill that drops ``spatial_ref`` or a bounds var does not write a
    reference to it. Returns a shallow copy.
    """
    ds = ds.copy()
    for var in ds.variables.values():
        for holder in (var.attrs, var.encoding):
            for key in ("coordinates", "ancillary_variables", "bounds", "grid_mapping"):
                value = holder.get(key)
                if not isinstance(value, str):
                    continue
                tokens = value.split()
                kept = [t for t in tokens if t.rstrip(":") in ds.variables]
                if len(kept) == len(tokens):
                    continue
                if kept and key != "grid_mapping":
                    holder[key] = " ".join(kept)
                else:
                    del holder[key]
    return ds


def udunits_error(units, *, catch: tuple | None = None):
    """Try parsing ``units`` with pint (CF/UDUNITS via cf_xarray.units).

    Return the error, or None if ok. Caller builds the user-facing message.
    ``None`` and ``""`` pass through (no raise), matching prior blank handling.
    """
    import cf_xarray.units  # noqa: F401 — configures application_registry
    from pint import UndefinedUnitError
    from pint import application_registry as ureg

    if catch is None:
        catch = (UndefinedUnitError, TypeError, ValueError)
    if units is None or units == "":
        return None
    try:
        ureg.Unit(units)
    except catch as exc:
        return exc
    return None


def cf_axes_missing(ds, axes: tuple = ("X", "Y", "T")) -> list:
    """Which of ``axes`` cf-xarray cannot resolve (empty list = all ok)."""
    import cf_xarray  # noqa: F401 -- registers the .cf accessor

    missing = []
    for axis in axes:
        try:
            resolved = ds.cf.axes.get(axis)
        except Exception:  # noqa: BLE001 -- unresolvable axis is the finding
            resolved = None
        if not resolved:
            missing.append(axis)
    return missing


def cf_dim(obj, cf_name: str):
    """Resolve a cf-xarray key (e.g. ``"latitude"``) to a coord name, or None.

    No name heuristics — unlike ``detect_spatial_dims`` / ``detect_time_dim``.
    """
    import cf_xarray  # noqa: F401 -- registers the .cf accessor

    try:
        return obj.cf[cf_name].name
    except KeyError:
        return None


def auto_variable(ds):
    """Pick a real data var, skipping CF grid-mapping (CRS) containers.

    Prefers a var with ≥2 dims; returns None if nothing suitable remains.
    """
    mapping_targets = {
        ds[d].attrs.get("grid_mapping") for d in ds.data_vars if ds[d].attrs.get("grid_mapping")
    }
    candidates = [
        v
        for v in ds.data_vars
        if "grid_mapping_name" not in ds[v].attrs and v not in mapping_targets
    ]
    if not candidates:
        return None
    multidim = [v for v in candidates if len(ds[v].dims) >= 2]
    return (multidim or candidates)[0]


def stamp_cf_dsg(ds, var_attrs: dict, *, station_id_long_name: str, name_long_name: str):
    """Stamp CF timeSeries DSG attrs for station/point_obs data. Returns ``ds``.

    Sets lat/lon/time/station_id roles and ``coordinates`` on each data var.
    ``var_attrs`` supplies per-var units/long_name/etc. (KeyError if a var is missing).
    """
    ds["latitude"].attrs.update(
        standard_name="latitude", long_name="station latitude", units="degrees_north", axis="Y"
    )
    ds["longitude"].attrs.update(
        standard_name="longitude", long_name="station longitude", units="degrees_east", axis="X"
    )
    ds["time"].attrs.update(standard_name="time", long_name="time", axis="T")
    ds["station_id"].attrs.update(cf_role="timeseries_id", long_name=station_id_long_name)
    if "name" in ds.coords or "name" in ds.variables:
        ds["name"].attrs.update(long_name=name_long_name)

    for var in ds.data_vars:
        ds[var].attrs.update({"coordinates": "latitude longitude time", **var_attrs[var]})
    return ds


def verify_cf_dsg(ds) -> None:
    """Check cf-xarray can read the timeSeries DSG; raise DataError if not.

    Call after ``stamp_cf_dsg`` and before writing Zarr.
    """
    import cf_xarray  # noqa: F401 -- registers the .cf accessor

    problems = []
    cf_roles = ds.cf.cf_roles
    if "station_id" not in cf_roles.get("timeseries_id", []):
        problems.append(f"cf_role timeseries_id did not resolve to station_id (got {cf_roles})")
    for name in ("latitude", "longitude", "time"):
        try:
            ds.cf[name]
        except KeyError:
            problems.append(f"cf-xarray could not resolve the {name} coordinate")
    if problems:
        raise DataError(
            "CF-1.13 DSG verification failed before write:\n  - " + "\n  - ".join(problems)
        )

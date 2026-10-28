"""Ontology for the weather-skills standard dataset.

Dims and types skills declare via ``Dataset(...)`` on arguments, plus checks
that a Zarr matches those requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

from weather_skills_core.errors import UsageError

# Ontology dims skills declare on Dataset(...)
DIMS = frozenset(
    {
        "lat",
        "lon",
        "time",
        "init_time",
        "prediction_timedelta",
        "member",
        "vertical",
        "day_of_year",
        "point_id",
        "x",
        "y",
    }
)

# Types → required ontology dims (AND)
TYPE_DIMS: dict[str, frozenset[str]] = {
    "spatial": frozenset({"lat", "lon"}),
    "observations": frozenset({"lat", "lon", "time"}),
    "forecast": frozenset({"lat", "lon", "init_time", "prediction_timedelta"}),
    "vertical_forecast": frozenset({"lat", "lon", "init_time", "prediction_timedelta", "vertical"}),
    "ensemble_forecast": frozenset({"lat", "lon", "init_time", "prediction_timedelta", "member"}),
    "point_obs": frozenset({"point_id", "time"}),
}

# Type name synonyms → primary key in TYPE_DIMS
TYPE_ALIASES: dict[str, str] = {
    "space": "spatial",
    "obs": "observations",
    "analysis": "observations",
    "retrieval": "observations",
    "field": "observations",
    "data": "observations",
    "station": "point_obs",
}

# Dataset / CF names → preferred ontology name (lat/lon preferred; x/y also ontology dims).
ALIASES: dict[str, str] = {
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
    "time": "time",
    "init_time": "init_time",
    "prediction_timedelta": "prediction_timedelta",
    "step": "prediction_timedelta",
    "lead_time": "prediction_timedelta",
    "member": "member",
    "number": "member",
    "realization": "member",
    "vertical": "vertical",
    "level": "vertical",
    "pressure": "vertical",
    "height": "vertical",
    "altitude": "vertical",
    "lev": "vertical",
    "isobaricInhPa": "vertical",
    "point_id": "point_id",
    "station_id": "point_id",
    "day_of_year": "day_of_year",
    "doy": "day_of_year",
}

ANY = "any"


def names_for(preferred: str) -> tuple[str, ...]:
    """Dataset names that map to ``preferred`` in ``ALIASES`` (preferred first)."""
    others = sorted(k for k, v in ALIASES.items() if v == preferred and k != preferred)
    return (preferred, *others)


@dataclass(frozen=True)
class IoSpec:
    """Required ontology dims for one Zarr path.

    Built by :class:`~weather_skills_core.Dataset` (or :func:`normalize_io_spec`).
    Each alternative is a set of dims that must all be present; the dataset
    may match any one. ``None`` means no dim check (``"any"``).

    Opaque files use ``pathlib.Path``, not ``Dataset``.
    """

    alternatives: tuple[frozenset[str], ...] | None  # None = any (no dim check)


def required_dims_for(name: str) -> frozenset[str]:
    """Ontology dims required by one type or dim name (``TYPE_ALIASES`` → ``TYPE_DIMS``, else ``DIMS``/``ALIASES``)."""
    primary = TYPE_ALIASES.get(name, name)
    if primary in TYPE_DIMS:
        return TYPE_DIMS[primary]
    if name in DIMS:
        return frozenset({name})
    dim = ALIASES.get(name, name)
    if dim in DIMS:
        return frozenset({dim})
    raise ValueError(
        f"unknown type or dimension {name!r}; expected a dimension {sorted(DIMS)}, "
        f"a type {sorted(TYPE_DIMS)}, or a type alias {sorted(TYPE_ALIASES)}"
    )


def and_group(spec) -> frozenset[str]:
    """Flatten a str or AND-tuple of ontology names into one required dim-set."""
    if isinstance(spec, str):
        return required_dims_for(spec)
    if isinstance(spec, tuple):
        out: set[str] = set()
        for part in spec:
            if not isinstance(part, str):
                raise TypeError(f"invalid AND-group entry {part!r}")
            out |= required_dims_for(part)
        return frozenset(out)
    raise ValueError(f"expected str or tuple for AND-group, got {type(spec).__name__}")


def parse_alternatives(spec) -> tuple[frozenset[str], ...]:
    """Parse a Zarr IO entry body into OR-alternatives: str / tuple=AND / list=OR → :attr:`IoSpec.alternatives`."""
    if isinstance(spec, str):
        return (required_dims_for(spec),)
    if isinstance(spec, tuple):
        return (and_group(spec),)
    if isinstance(spec, list):
        if not spec:
            raise ValueError("OR list must not be empty")
        alts: list[frozenset[str]] = []
        for item in spec:
            if isinstance(item, list):
                raise TypeError(f"nested OR lists are not allowed; got {item!r}")
            alts.extend(parse_alternatives(item))
        return tuple(alts)
    raise ValueError(f"invalid IO entry {spec!r}; expected str, list (OR), or tuple (AND)")


def normalize_io_spec(raw) -> IoSpec:
    """Parse a Dataset declaration body into a :class:`IoSpec`."""
    if isinstance(raw, str) and raw == ANY:
        return IoSpec(alternatives=None)
    return IoSpec(alternatives=parse_alternatives(raw))


def has_dim(ds, dim: str) -> bool:
    """True if ``ds`` has the ontology dim (ALIASES / CF / special lat-lon rules)."""
    if dim == "lat":
        try:
            lat_dim, _ = detect_spatial_dims(ds)
            return lat_dim in ds.dims
        except UsageError:
            return False
    if dim == "lon":
        try:
            _, lon_dim = detect_spatial_dims(ds)
            return lon_dim in ds.dims
        except UsageError:
            return False
    if dim == "time":
        try:
            name = detect_time_dim(ds)
            return name in ds.dims
        except UsageError:
            return False
    if dim == "init_time":
        if "init_time" in ds.dims or "init_time" in ds.coords:
            return True
        # Classic forecast: scalar time coord + lead-time dim
        if "time" in ds.coords and "time" not in ds.dims and ds["time"].ndim == 0:
            return any(ALIASES.get(n) == "prediction_timedelta" for n in ds.dims)
        return False
    if dim in ("prediction_timedelta", "member", "vertical", "day_of_year"):
        return any(ALIASES.get(n) == dim for n in ds.dims)
    if dim == "point_id":
        point = next((n for n in names_for("point_id") if n in ds.dims), None)
        if point is None:
            return False
        lat = next(
            (n for n in names_for("lat") if n in ds.coords and tuple(ds[n].dims) == (point,)), None
        )
        lon = next(
            (n for n in names_for("lon") if n in ds.coords and tuple(ds[n].dims) == (point,)), None
        )
        return lat is not None and lon is not None
    if dim in ("x", "y"):
        return dim in ds.dims
    return False


def validate_dims(ds, alternatives: tuple[frozenset[str], ...] | None, name: str) -> frozenset[str]:
    """Return the first matching dim-set, or raise; ``alternatives is None`` = unconstrained."""
    if alternatives is None:
        return frozenset()
    missing = [[d for d in sorted(alt) if not has_dim(ds, d)] for alt in alternatives]
    for alt, miss in zip(alternatives, missing, strict=True):
        if not miss:
            return alt
    parts = [
        f"{{{', '.join(sorted(alt))}}} missing {miss}"
        for alt, miss in zip(alternatives, missing, strict=True)
    ]
    raise UsageError(
        f"{name} does not satisfy required dimensions ({' OR '.join(parts)}); dims={list(ds.dims)}"
    )


def detect_type(ds) -> str:
    """Classify ``ds`` as a primary key in ``TYPE_DIMS`` (or ``ensemble_forecast``)."""
    if has_dim(ds, "point_id"):
        return "point_obs"
    if has_dim(ds, "prediction_timedelta") and has_dim(ds, "init_time"):
        if has_dim(ds, "member"):
            return "ensemble_forecast"
        if has_dim(ds, "vertical"):
            return "vertical_forecast"
        return "forecast"
    return "observations"


def validate_input(ds, allowed, name: str) -> str:
    """Check ``ds`` against an IoSpec or Dataset declaration; return ``detect_type``."""
    if isinstance(allowed, IoSpec):
        spec = allowed
    elif isinstance(allowed, (str, list, tuple)):
        spec = normalize_io_spec(allowed)
    else:
        raise TypeError(f"invalid validate_input allowed={allowed!r}")
    validate_dims(ds, spec.alternatives, name)
    return detect_type(ds)


def detect_spatial_dims(ds) -> tuple:
    """Find lat/lon dim names: CF metadata, then ALIASES."""
    try:
        import cf_xarray  # noqa: F401 -- registers the .cf accessor

        return ds.cf["latitude"].name, ds.cf["longitude"].name
    except KeyError:
        pass
    lat_dim = next((n for n in names_for("lat") if n in ds.dims), None)
    lon_dim = next((n for n in names_for("lon") if n in ds.dims), None)
    if lat_dim is None or lon_dim is None:
        raise UsageError(
            f"could not identify lat/lon coords via CF metadata or name "
            f"heuristics in {list(ds.coords)}. Rename them to lat/lon or add CF attrs."
        )
    return lat_dim, lon_dim


def detect_time_dim(ds, override: str | None = None) -> str:
    """Find the time dim: optional name override, then CF, then literal ``time``."""
    if override:
        if override not in ds.dims:
            raise UsageError(f"--time-dim {override!r} not in dataset dims {list(ds.dims)}")
        return override
    try:
        import cf_xarray  # noqa: F401 -- registers the .cf accessor

        name = ds.cf["time"].name
        if name in ds.dims:
            return name
    except KeyError:
        pass
    if "time" in ds.dims:
        return "time"
    raise UsageError(
        f"could not identify a time dim via CF metadata or name heuristics in "
        f"{list(ds.dims)}. Pass --time-dim to override."
    )

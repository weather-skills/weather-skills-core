"""Helpers skills call: dates, bbox, lon wrap, GeoJSON, env, transient errors."""

from __future__ import annotations

import os
import re
from calendar import day_name, monthrange
from datetime import date, datetime, timedelta

from weather_skills_core.errors import DataError, UsageError
from weather_skills_core.standard_dataset import detect_spatial_dims, names_for

_ABS_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Retryable HTTP status codes as whole tokens (avoid matching "14290", "50000").
_STATUS_RE = re.compile(r"\b(?:429|500|502|503|504)\b")
# Specific connection phrases only — bare "connection" appears on permanent urllib3 errors too.
_TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
    "connection refused",
    "connection aborted",
)


def is_transient(exc: Exception) -> bool:
    """Heuristic: does this error look like a retryable transient/rate-limit?

    Matches error text for HTTP 429/5xx status tokens, timeout markers, or
    specific connection-failure phrases. Case-insensitive. Retry policy stays
    with the caller.
    """
    text = str(exc).lower()
    return bool(_STATUS_RE.search(text)) or any(m in text for m in _TRANSIENT_MARKERS)


def normalize_step_coord(ds, dim: str = "step"):
    """Cast forecast ``step`` to ``timedelta64[ns]`` when it is a timedelta axis.

    Mixed microsecond/day encodings (common from remote Zarr) break
    ``infer_timestep`` when values are misread through a fixed cast.
    """
    import numpy as np

    if dim not in ds.dims and dim not in ds.coords:
        return ds
    values = np.asarray(ds[dim].values)
    if not np.issubdtype(values.dtype, np.timedelta64):
        return ds
    if values.dtype == np.dtype("timedelta64[ns]"):
        return ds
    # Keep the coord's attrs (standard_name=forecast_period, long_name).
    return ds.assign_coords(
        {dim: (ds[dim].dims, values.astype("timedelta64[ns]"), dict(ds[dim].attrs))}
    )


_LATLON_DECIMALS = 5


def normalize_latlon_coords(ds):
    """Round lat/lon coords to 5 decimals and store them as float32.

    Snaps float64 noise (``5.9749991`` vs ``5.975``) so grids
    that are the same at ~1 m join. A real half-cell offset (0.025°) is kept.
    """
    import numpy as np

    names = [n for key in ("lat", "lon") for n in names_for(key) if n in ds.coords]
    names += [
        n
        for n in ds.coords
        if n not in names and ds[n].attrs.get("standard_name") in ("latitude", "longitude")
    ]

    updates = {}
    for name in dict.fromkeys(names):
        values = np.asarray(ds[name].values)
        if not np.issubdtype(values.dtype, np.floating):
            continue
        snapped = np.round(values.astype(np.float64), _LATLON_DECIMALS).astype(np.float32)
        if values.dtype != np.float32 or not np.array_equal(values, snapped):
            updates[name] = (ds[name].dims, snapped, dict(ds[name].attrs))
    if not updates:
        return ds
    out = ds.assign_coords(updates)
    for name in updates:
        out[name].encoding["dtype"] = "float32"
    return out


def fill_missing_data_var_attrs(src, dst):
    """Fill attrs present on ``src`` but missing on matching ``dst`` data vars.

    Heals stripped metadata (regrid, etc.) without overwriting attrs a skill set.
    Does not restore ``aggregation_period`` or rate descriptions (``comment``,
    ``short_name``, ``step_type``, e.g. "Average precipitation rate…", ``prate``,
    ``avg``) onto vars whose ``cell_methods`` already record a sum (period totals).
    """
    from weather_skills_core.units import AGGREGATION_PERIOD_ATTR, cell_methods_has_sum

    out = dst
    dirty = False
    for name in dst.data_vars:
        if name not in src.data_vars:
            continue
        is_total = cell_methods_has_sum(dst[name].attrs.get("cell_methods"))
        for key, value in src[name].attrs.items():
            if key in dst[name].attrs:
                continue
            if is_total and key in (AGGREGATION_PERIOD_ATTR, "comment", "short_name", "step_type"):
                continue
            if not dirty:
                out = dst.copy()
                dirty = True
            out[name].attrs[key] = value
    return out


def require_env(*names: str, message: str | None = None) -> tuple:
    """Return the values of the named environment variables, in order.

    Unset or empty vars are missing; raises ``UsageError`` with ``message`` or a
    default listing the missing names. Never print or log the values.
    """
    values = [os.environ.get(name) for name in names]
    missing = [name for name, value in zip(names, values, strict=True) if not value]
    if missing:
        raise UsageError(message or f"missing required env var(s): {', '.join(missing)}")
    return tuple(values)


def np_to_date(value) -> date:
    """Convert a numpy datetime64 to a calendar date (truncating time-of-day)."""
    import numpy as np

    if np.isnat(value):
        raise DataError(
            "time coordinate value is NaT (not-a-time); the dataset has a missing or "
            "unfilled time entry where a valid date is required."
        )
    return date.fromisoformat(np.datetime_as_string(value, unit="D"))


def parse_date(value: str) -> date:
    """Parse an absolute ``YYYY-MM-DD`` date string."""
    try:
        if not _ABS_DATE_RE.match(value):
            raise ValueError
        return date.fromisoformat(value)
    except ValueError:
        raise UsageError(
            f"invalid date value {value!r}: expected an absolute date YYYY-MM-DD"
        ) from None


def parse_bbox(bbox: str) -> tuple:
    """Parse ``N/W/S/E`` decimal degrees into ``(N, W, S, E)`` floats."""
    try:
        north, west, south, east = (float(x) for x in bbox.split("/"))
    except ValueError:
        raise UsageError("--bbox must be four decimal degrees N/W/S/E.") from None
    if not (-90 <= south <= north <= 90):
        raise UsageError(
            f"--bbox {bbox}: needs -90 <= S <= N <= 90 (order is N/W/S/E); "
            "west > east is fine and crosses the antimeridian."
        )
    return north, west, south, east


def lat_slice(lat_vals, north, south) -> slice:
    """Build a latitude ``sel`` slice for ascending or descending axes."""
    if lat_vals.size and lat_vals[0] > lat_vals[-1]:
        return slice(north, south)
    return slice(south, north)


def polygon_from_geojson(path, *, flag: str = "--mask-geojson"):
    """Load GeoJSON and return one shapely geometry (union of features).

    ``flag`` is only for error messages (which CLI flag supplied the path).
    """
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise UsageError(f"{flag} file not found: {path}")
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"could not read {flag} {path}: {exc}") from None

    if not isinstance(data, dict):
        # Valid JSON, but a top-level array or scalar is no GeoJSON object.
        raise UsageError(f"{flag} {path} has no usable geometry.")
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not isinstance(features, list):
            raise UsageError(f"{flag} {path}: 'features' is not a list.")
        geoms = []
        for feature in features:
            if not isinstance(feature, dict):
                raise UsageError(f"{flag} {path}: a feature is not a JSON object.")
            if feature.get("geometry"):
                geoms.append(feature["geometry"])
    elif data.get("type") == "Feature":
        geoms = [data["geometry"]] if data.get("geometry") else []
    else:
        # A bare geometry object.
        geoms = [data]

    if not geoms:
        raise UsageError(f"{flag} {path} has no usable geometry.")

    from shapely.errors import GeometryTypeError
    from shapely.geometry import shape
    from shapely.ops import unary_union

    # shape()/unary_union() raise on a well-formed JSON value that is not a
    # valid geometry (an unknown type, missing/malformed coordinates, a
    # non-object entry); convert every such failure to a flag-named UsageError
    # so no JSON input produces a raw traceback.
    try:
        return unary_union([shape(g) for g in geoms])
    except (GeometryTypeError, KeyError, AttributeError, ValueError, TypeError) as exc:
        raise UsageError(f"{flag} {path} has no usable geometry ({exc}).") from None


def ensure_normalized_longitude(obj, lon_dim: str | None = None):
    """Wrap 0..360 longitude to [-180, 180) when needed; otherwise leave ``obj``.

    No-op when ``lon_dim`` is missing, empty, or already ≤ 180. Accepts a
    Dataset or DataArray. Wrapping sorts the axis ascending and drops
    duplicate labels produced by wrapping 0 and 360. Station longitudes
    (not a dimension) are wrapped in place and not sorted.
    """
    import numpy as np

    if lon_dim is None:
        try:
            _, lon_dim = detect_spatial_dims(obj)
        except UsageError:
            return obj
    if lon_dim not in obj.coords and lon_dim not in getattr(obj, "variables", ()):
        return obj
    lon_vals = np.asarray(obj[lon_dim].values)
    if not lon_vals.size or float(np.nanmax(lon_vals)) <= 180.0:
        return obj

    lon = ((obj[lon_dim] + 180) % 360) - 180
    lon.attrs = dict(obj[lon_dim].attrs)
    obj = obj.assign_coords({lon_dim: lon})
    if lon_dim in obj.dims:
        # np.unique returns each unique value's first-occurrence index; keeping
        # those (in input order) drops any later duplicate the wrap produced.
        _, first = np.unique(obj[lon_dim].values, return_index=True)
        if len(first) < obj.sizes[lon_dim]:
            obj = obj.isel({lon_dim: np.sort(first)})
        obj = obj.sortby(lon_dim)
    return obj


def _point_spatial(ds):
    """``(point_dim, lat, lon)`` for tabular point_obs, else ``None``."""
    point_dim = next((n for n in names_for("point_id") if n in ds.dims), None)
    if point_dim is None:
        return None
    lat, lon = (
        next(
            (n for n in names_for(key) if n in ds.coords and ds[n].dims == (point_dim,)),
            None,
        )
        for key in ("lat", "lon")
    )
    if lat is None or lon is None:
        raise UsageError(
            f"station clip requires 1-D latitude/longitude coords on {point_dim} "
            f"(coords: {list(ds.coords)})"
        )
    return point_dim, lat, lon


def _empty_bbox(north, west, south, east, kind: str):
    bbox_str = f"{north}/{west}/{south}/{east}"
    if west > east:
        raise DataError(
            f"--bbox {bbox_str} crosses the antimeridian (west {west} > east {east}) "
            f"but selects no {kind}; check the N/S extent and that west/east "
            "bracket the intended dateline-crossing span."
        )
    raise DataError(f"--bbox {bbox_str} selects no {kind}; check the extent and N/W/S/E order.")


def bbox_subset(ds, bbox, *, lat_dim: str | None = None, lon_dim: str | None = None):
    """Subset a gridded or station dataset to an ``N/W/S/E`` bbox (string or tuple).

    Point obs (``station_id`` / ``point_id`` with 1-D lat/lon coords) are filtered
    per station; grids are sliced on lat/lon dims. Wraps 0..360 lon, supports
    antimeridian (west > east). Empty selection raises DataError.
    """
    import numpy as np

    if isinstance(bbox, str):
        north, west, south, east = parse_bbox(bbox)
    else:
        north, west, south, east = bbox

    point = _point_spatial(ds)
    if point is not None:
        point_dim, lat_name, lon_name = point
        ds = ensure_normalized_longitude(ds, lon_dim=lon_name)
        lat = np.asarray(ds[lat_name].values)
        lon = np.asarray(ds[lon_name].values)
        in_lon = (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
        mask = (lat >= south) & (lat <= north) & in_lon
        if not np.any(mask):
            _empty_bbox(north, west, south, east, "stations")
        return ds.isel({point_dim: np.nonzero(mask)[0]})

    if lat_dim is None or lon_dim is None:
        lat_dim, lon_dim = detect_spatial_dims(ds)

    # Wrap lon to [-180, 180] before the slice so a 0..360 input grid still
    # intersects bboxes that use negative west/east values.
    ds = ensure_normalized_longitude(ds, lon_dim=lon_dim)
    for axis, dim in (("lat", lat_dim), ("lon", lon_dim)):
        vals = np.asarray(ds[dim].values)
        if vals.size == 0:
            raise UsageError(f"{axis} axis has length 0; cannot subset.")
        diffs = np.diff(vals)
        if not ((diffs > 0).all() or (diffs < 0).all()):
            raise UsageError(
                f"{axis} axis is non-monotonic; cannot infer slice orientation. "
                "Re-sort the input or pre-process before subsetting."
            )
    lon_vals = np.asarray(ds[lon_dim].values)
    lon_ascending = lon_vals.size == 1 or lon_vals[1] > lon_vals[0]
    ds = ds.sel({lat_dim: lat_slice(np.asarray(ds[lat_dim].values), north, south)})

    if west <= east:
        # Contiguous longitude span. Slice in the axis's own monotonic order.
        lon_sel = slice(west, east) if lon_ascending else slice(east, west)
        ds = ds.sel({lon_dim: lon_sel})
    else:
        # Antimeridian crossing (west > east): the span runs west .. +180 and
        # -180 .. east. Select each wing with a label slice and concatenate
        # in the axis's native order; unlike a where(..., drop=True) mask
        # this never materializes a full-grid mask and keeps integer
        # variables integer (masking promotes them to float).
        import xarray as xr

        if lon_ascending:
            wings = [ds.sel({lon_dim: slice(None, east)}), ds.sel({lon_dim: slice(west, None)})]
        else:
            wings = [ds.sel({lon_dim: slice(None, west)}), ds.sel({lon_dim: slice(east, None)})]
        ds = xr.concat(
            wings,
            dim=lon_dim,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="exact",
        )

    if ds.sizes.get(lat_dim, 0) == 0 or ds.sizes.get(lon_dim, 0) == 0:
        _empty_bbox(north, west, south, east, "grid cells")
    return ds


def clip_by_geometry(
    ds, geometry, *, lat_dim: str | None = None, lon_dim: str | None = None, drop: bool = True
):
    """Clip gridded or station dataset to a shapely geometry (NaN outside; optional drop)."""
    import numpy as np
    import shapely
    import xarray as xr

    if geometry is None:
        return ds
    point = _point_spatial(ds)
    if point is not None:
        point_dim, lat_name, lon_name = point
        ds = ensure_normalized_longitude(ds, lon_dim=lon_name)
        mask = shapely.contains_xy(
            geometry,
            np.asarray(ds[lon_name].values),
            np.asarray(ds[lat_name].values),
        )
        if not mask.any():
            raise DataError("geometry selects no stations")
        if drop:
            return ds.isel({point_dim: np.nonzero(mask)[0]})
        # NaN only the variables that run along the station axis; ds.where
        # would broadcast the mask onto every other variable.
        keep = xr.DataArray(mask, dims=point_dim)
        return ds.assign(
            {name: var.where(keep) for name, var in ds.data_vars.items() if point_dim in var.dims}
        )

    if lat_dim is None or lon_dim is None:
        lat_dim, lon_dim = detect_spatial_dims(ds)
    ds = ensure_normalized_longitude(ds, lon_dim=lon_dim)

    def _edges(vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            raise UsageError("spatial axis has length 0; cannot clip by geometry.")
        if vals.size == 1:
            # No spacing to infer cell area: treat as a point cell.
            return np.array([vals[0], vals[0]], dtype=float)
        diffs = np.diff(vals)
        if not ((diffs > 0).all() or (diffs < 0).all()):
            raise UsageError(
                "spatial axis is non-monotonic; cannot infer cell boundaries for clipping."
            )
        mids = (vals[:-1] + vals[1:]) / 2.0
        first = vals[0] - (vals[1] - vals[0]) / 2.0
        last = vals[-1] + (vals[-1] - vals[-2]) / 2.0
        return np.concatenate(([first], mids, [last]))

    lat_edges = _edges(np.asarray(ds[lat_dim].values))
    lon_edges = _edges(np.asarray(ds[lon_dim].values))

    if len(lat_edges) == 2 or len(lon_edges) == 2:
        # Degenerate 1-cell axis: fall back to center-point containment.
        lon2d, lat2d = np.meshgrid(ds[lon_dim].values, ds[lat_dim].values)
        mask = shapely.contains_xy(geometry, lon2d, lat2d)
    else:
        west = np.minimum(lon_edges[:-1], lon_edges[1:])
        east = np.maximum(lon_edges[:-1], lon_edges[1:])
        south = np.minimum(lat_edges[:-1], lat_edges[1:])
        north = np.maximum(lat_edges[:-1], lat_edges[1:])
        # Only cells inside the geometry's bounding box can overlap it; build
        # cell boxes for that window alone, not for the whole (maybe global) grid.
        g_west, g_south, g_east, g_north = geometry.bounds
        rows = np.nonzero((north > g_south) & (south < g_north))[0]
        cols = np.nonzero((east > g_west) & (west < g_east))[0]
        mask = np.zeros((south.size, west.size), dtype=bool)
        if rows.size and cols.size:
            west2d, south2d = np.meshgrid(west[cols], south[rows])
            east2d, north2d = np.meshgrid(east[cols], north[rows])
            cells = shapely.box(west2d, south2d, east2d, north2d)
            # Keep cells with a positive-area overlap (not just edge/corner touch).
            mask[np.ix_(rows, cols)] = shapely.area(shapely.intersection(cells, geometry)) > 0.0

    if not mask.any():
        raise DataError("geometry selects no grid cells")
    if drop:
        # Trim to the span of rows and columns the geometry touches, decided by
        # the mask rather than the data. Rows between the parts of a
        # MultiPolygon stay (as NaN), so the grid keeps its regular spacing,
        # and nothing is computed.
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        rows = slice(rows[0], rows[-1] + 1)
        cols = slice(cols[0], cols[-1] + 1)
        ds = ds.isel({lat_dim: rows, lon_dim: cols})
        mask = mask[rows, cols]
    # NaN only the variables on the grid; ds.where would broadcast the mask
    # onto every other variable (time bounds, station metadata, ...).
    mask_da = xr.DataArray(mask, dims=(lat_dim, lon_dim))
    return ds.assign(
        {
            name: var.where(mask_da)
            for name, var in ds.data_vars.items()
            if lat_dim in var.dims and lon_dim in var.dims
        }
    )


def grid_spacing(coord_vals) -> float:
    """Median absolute spacing of a 1-D coordinate (degrees or similar)."""
    import numpy as np

    coord = np.asarray(coord_vals)
    if coord.size < 2:
        raise ValueError(f"Cannot infer spacing for coord with size {coord.size}")
    return float(abs(np.median(np.diff(coord))))


def apply_write_encoding(ds, *, time_units=None, time_calendar=None, fills=None):
    """Set time encoding and optional per-variable ``_FillValue`` encodings in place."""
    if time_units is not None and "time" in ds.coords:
        ds["time"].encoding["units"] = time_units
    if time_calendar is not None and "time" in ds.coords:
        ds["time"].encoding["calendar"] = time_calendar
    if fills:
        for var, fill in fills.items():
            if fill is not None and var in ds.variables:
                ds[var].encoding["_FillValue"] = fill
    return ds


def verify_cf_decode(ds, axes: tuple = ("X", "Y", "T")):
    """Raise DataError if cf-xarray cannot resolve the given axes."""
    from weather_skills_core.cf import cf_axes_missing

    missing = cf_axes_missing(ds, axes=axes)
    if missing:
        raise DataError(
            f"cf-xarray did not resolve axes {missing} "
            f"(expected {list(axes)}); the output is not CF-compliant."
        )


def latitude_weights(lats):
    """Cosine latitude weights normalized to mean 1."""
    import numpy as np
    import xarray as xr

    if not isinstance(lats, xr.DataArray):
        lats = xr.DataArray(lats)
    weights = np.cos(np.deg2rad(lats))
    return weights / weights.mean()


_WEEKDAYS = {name.lower(): i for i, name in enumerate(day_name)}


def stride_dates(start, end, stride: str = "day"):
    """Inclusive date list from start to end.

    ``stride`` is ``day``/``week``/``month``/``year``, or weekday names
    (``Monday``, ``Monday/Thursday``).
    """
    import numpy as np

    def as_datetime(value) -> datetime:
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, date) and not isinstance(value, datetime):
            return datetime(value.year, value.month, value.day)  # noqa: DTZ001
        if isinstance(value, np.datetime64):
            return datetime.fromisoformat(np.datetime_as_string(value, unit="D"))
        text = str(value)
        if "T" in text:
            text = text.split("T", 1)[0]
        return datetime.fromisoformat(text[:10])

    start_dt, end_dt = as_datetime(start), as_datetime(end)
    if end_dt < start_dt:
        raise UsageError(f"stride start {start_dt.date()} is after end {end_dt.date()}")

    parts = [p.strip().lower() for p in stride.split("/") if p.strip()]
    if parts and all(p in _WEEKDAYS for p in parts):
        want = {_WEEKDAYS[p] for p in parts}
        out = []
        cur = start_dt
        while cur <= end_dt:
            if cur.weekday() in want:
                out.append(cur)
            cur += timedelta(days=1)
        return np.array(out, dtype="datetime64[ns]")

    key = stride.strip().lower()
    if key == "day":
        delta, months, years = timedelta(days=1), 0, 0
    elif key == "week":
        delta, months, years = timedelta(days=7), 0, 0
    elif key == "month":
        delta, months, years = None, 1, 0
    elif key == "year":
        delta, months, years = None, 0, 1
    else:
        raise UsageError(
            f"invalid stride {stride!r}; use day/week/month/year or weekday "
            "names (e.g. Monday, Monday/Thursday)"
        )

    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur)
        if delta is not None:
            cur = cur + delta
        else:
            y = cur.year + years + (cur.month - 1 + months) // 12
            m = (cur.month - 1 + months) % 12 + 1
            d = min(cur.day, monthrange(y, m)[1])
            cur = datetime(y, m, d)  # noqa: DTZ001
    return np.array(out, dtype="datetime64[ns]")


def roll_and_agg(
    ds,
    window: int,
    dim: str,
    method: str = "mean",
    *,
    align: str = "left",
    stride=None,
    min_periods: int | None = None,
):
    """N-step rolling aggregation, then optional stride subsample.

    ``window`` is in axis steps (days for daily ``time``, or steps for ``step``).
    ``align`` is left/right/center label placement. ``stride`` is an int step,
    or a ``stride_dates`` string when ``dim`` is datetime64.
    Methods: ``mean``, ``min``, ``max`` only (rates-first).
    """
    import numpy as np

    from weather_skills_core import units  # noqa: F401  (registers the .pint accessor)

    if window < 1:
        raise UsageError(f"rolling window must be >= 1; got {window}")
    if window == 1 and stride is None:
        return ds
    if min_periods is None:
        min_periods = window
    dtype = ds[dim].dtype
    if not (np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)):
        raise UsageError(
            f"rolling aggregation requires datetime64 or timedelta64 on {dim!r}; got dtype {dtype}"
        )
    # Roll on plain numbers: pint refuses the window sum behind a mean for an
    # offset unit (degree_Celsius), though mean/min/max of one are well defined.
    quantified = any(getattr(ds[v].pint, "units", None) is not None for v in ds.data_vars)
    plain = ds.pint.dequantify() if quantified else ds
    rolled = plain.rolling({dim: window}, min_periods=min_periods, center=False)
    fn = {"mean": rolled.mean, "max": rolled.max, "min": rolled.min}.get(method)
    if fn is None:
        raise UsageError(f"unsupported rolling method {method!r}; use mean|min|max")
    out = fn(skipna=True, keep_attrs=True)
    out = out.pint.quantify() if quantified else out
    out = out.isel({dim: slice(window - 1, None)})
    step_by_align = {"center": (window - 1) // 2, "right": 0, "left": window - 1}
    if align not in step_by_align:
        raise UsageError(f"align must be left/right/center; got {align!r}")
    step_shift = step_by_align[align]
    # Label each window with the original coordinate of its left/center/right
    # sample, so uneven or sub-daily spacing is labelled exactly.
    start = window - 1 - step_shift
    labels = ds[dim].values[start : start + out.sizes[dim]]
    out = out.assign_coords({dim: (dim, labels, ds[dim].attrs)})
    if stride is None:
        return out
    if isinstance(stride, int):
        if stride < 1:
            raise UsageError(f"stride must be >= 1; got {stride}")
        return out.isel({dim: slice(None, None, stride)})
    if not np.issubdtype(out[dim].dtype, np.datetime64):
        raise UsageError("string --stride requires a datetime64 time axis")
    times = stride_dates(out[dim].values[0], out[dim].values[-1], stride=str(stride))
    return out.sel({dim: times})

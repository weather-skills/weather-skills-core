"""Collapse dimensions of a Dataset, whole or per group of a label coord, with one statistic."""

import datetime
import itertools
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

from weather_skills_core.errors import DataError, UsageError
from weather_skills_core.units import (
    AGGREGATION_COVERAGE_COORD,
    AGGREGATION_PERIOD_ATTR,
    data_interval_of,
    infer_timestep,
    parse_aggregation_period,
)

METHODS = ("mean", "std", "min", "max", "sum", "median")
# Time labels whose full calendar period is known, as label-group writes them.
TIME_LABELS = ("month", "year", "dayofyear", "month_group")
_INITIALS = "JFMAMJJASOND"


def summarize(ds, dims=(), by=(), method="mean", variables=None, weights=None, min_coverage=1.0):
    """Collapse ``dims`` of ``ds``, or each group of the ``by`` label coords, with ``method``.

    ``ds`` is quantified, as ``@weather_skill`` passes inputs; the result is too.

    ``dims`` are dimensions to collapse; ``by`` are non-dimension label coords (a county or
    a season label) whose dims collapse to one entry per label. Labels on the same dims
    match together; labels on different dims combine as a product. The ``""`` label is
    left out, and other label coords on the grouped dims are dropped.

    ``method`` is one of ``METHODS``. NaNs are skipped; ``sum`` of an all-NaN slice is NaN
    and ``std`` is the sample standard deviation (``ddof=1``). ``sum`` over time or step of
    a rate is refused.

    ``weights`` (``method="mean"`` only) is a DataArray on some of the data's dims, such
    as ``latitude_weights``; each value counts in proportion to its weight, NaN values and
    their weights left out. It works with or without ``by``.

    With a ``by`` label along time (one of ``TIME_LABELS``), each input cell (station,
    grid cell, member) whose data cover less than ``min_coverage`` (0-1) of a group's full
    calendar period is left out before any other reduction; missing steps are never
    filled. A ``sum`` kept below full coverage writes ``<var>_coverage``.

    Only ``variables`` (default: all) carrying every grouped dim are summarized; others
    pass through. A collapsed dim is removed once no variable carries it.
    """
    if method not in METHODS:
        raise UsageError(f"method {method!r} is not one of {', '.join(METHODS)}")
    dims = list(dict.fromkeys(dims or []))
    by = list(dict.fromkeys(by or []))
    if not dims and not by:
        raise UsageError("nothing to summarize: give dims, by labels, or both")
    for d in dims:
        if d not in ds.dims:
            raise UsageError(f"dim {d!r} not in dataset (have {list(ds.dims)})")
    for b in by:
        if b not in ds.coords or b in ds.dims:
            labels = [c for c in ds.coords if c not in ds.dims]
            raise UsageError(f"by {b!r} is not a label coord (have {labels})")
    if weights is not None:
        if method != "mean":
            raise UsageError(f"weights need method mean, not {method}")
        if not set(weights.dims) <= set(ds.dims):
            raise UsageError(f"weights dims {weights.dims} are not all dataset dims")
    if not 0.0 <= min_coverage <= 1.0:
        raise UsageError(f"min_coverage must be in [0, 1]; got {min_coverage}")
    time_by = [b for b in by if "time" in ds[b].dims]
    for b in time_by:
        if b not in TIME_LABELS or ds[b].dims != ("time",):
            raise UsageError(
                f"by {b!r} runs along time but is not a label-group time label "
                f"({', '.join(TIME_LABELS)}), so its full calendar period is unknown"
            )
    selected = list(dict.fromkeys(variables)) if variables else list(ds.data_vars)
    for var in selected:
        if var not in ds.data_vars:
            raise UsageError(f"variable {var!r} not in dataset (have {list(ds.data_vars)})")
    # Read before the label coords on time are dropped below.
    seasons = ds["month_group"].values if "month_group" in ds.coords and time_by else None
    step_share = None
    if AGGREGATION_COVERAGE_COORD in ds.coords and ds[AGGREGATION_COVERAGE_COORD].dims == ("time",):
        # An input interval that was itself 3/7 complete counts as 3/7 of a step.
        step_share = np.nan_to_num(np.asarray(ds[AGGREGATION_COVERAGE_COORD].values, dtype=float))
    # The dims a label spans collapse into one entry per label.
    group_dims = list(dict.fromkeys(d for b in by for d in ds[b].dims))
    dims = list(dict.fromkeys([*group_dims, *dims]))
    # Other label coords on the grouped dims (admin_1_name, ...) cannot survive the collapse.
    ds = ds.drop_vars(
        [c for c in ds.coords if c not in (*by, *ds.dims) and set(ds[c].dims) & set(group_dims)]
    )
    # Groups, built once for every variable. Labels on the same dims are matched together;
    # labels on different dims (a county and a year) combine as a product. Each group keeps
    # only the rows/columns/times it touches, plus a mask when that box holds other cells,
    # so no step copies more than one group. With no labels there is one group: everything.
    label_values = {}
    groups = []
    for dset in dict.fromkeys(ds[b].dims for b in by):
        names = [b for b in by if ds[b].dims == dset]
        for b in by:
            if b not in names and set(ds[b].dims) & set(dset):
                raise UsageError(f"by {b!r} and {names[0]!r} share some but not all dims")
        for b in names:
            values = np.unique(ds[b].values)
            label_values[b] = values[values != ""] if values.dtype.kind in "UO" else values
            if label_values[b].size == 0:
                raise DataError(f'by {b!r}: every value is the empty label ""; nothing to group')
        entries = []
        for combo in itertools.product(*(range(label_values[b].size) for b in names)):
            mask = np.ones([ds.sizes[d] for d in dset], dtype=bool)
            for b, i in zip(names, combo, strict=True):
                mask &= ds[b].values == label_values[b][i]
            if not mask.any():
                continue
            index = [
                np.nonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))[0]
                for axis in range(mask.ndim)
            ]
            box = mask[np.ix_(*index)]
            entries.append(
                {
                    "labels": dict(zip(names, combo, strict=True)),
                    "isel": dict(zip(dset, index, strict=True)),
                    "box": None if box.all() else xr.DataArray(box, dims=dset),
                    "expected": None,  # time steps of a complete group, set below
                }
            )
        groups.append(entries)

    # Coverage: for each time group, the steps a complete record would hold. The labels are
    # rebuilt on a gap-free axis at the data's step reaching a year past both ends; a group
    # expects every step of each calendar run (one MAM, one year) the data touch, so a
    # season cut by the start or end of the record is incomplete.
    if time_by:
        times = ds.indexes["time"]
        first = times[0]
        period = next(
            (
                ds[v].attrs[AGGREGATION_PERIOD_ATTR]
                for v in ds.data_vars
                if isinstance(ds[v].attrs.get(AGGREGATION_PERIOD_ATTR), str)
            ),
            None,
        ) or data_interval_of(ds)
        calendar = re.fullmatch(r"\s*(\d+)\s*(month|year)s?\s*", period or "")
        if calendar or (period or "").strip() == "1 dekad":
            # Calendar steps: labels must be month starts (dekads on days 1, 11, 21).
            every = (
                int(calendar.group(1)) * (12 if calendar.group(2) == "year" else 1)
                if calendar
                else 1
            )
            n = (times[-1].year - first.year) * 12 + times[-1].month - first.month
            full = [
                first.replace(
                    year=first.year + (first.month - 1 + k) // 12,
                    month=(first.month - 1 + k) % 12 + 1,
                    day=day,
                )
                for k in range(-24, n + 25)
                if k % every == 0
                for day in ((1,) if calendar else (1, 11, 21))
            ]
        else:
            step = parse_aggregation_period(period) if period else infer_timestep(ds, "time")
            step = datetime.timedelta(seconds=float(step.to("second").magnitude))
            margin = int(np.ceil(datetime.timedelta(days=366) / step))
            n = round((times[-1] - first) / step)
            full = [first + i * step for i in range(-margin, n + margin + 1)]
        full = (
            pd.DatetimeIndex(full) if isinstance(times, pd.DatetimeIndex) else xr.CFTimeIndex(full)
        )
        at = full.get_indexer(times)
        if (at < 0).any():
            raise UsageError(
                f"time values do not fall on a regular {period or 'time'} axis, so the steps "
                "each time group should hold cannot be counted"
            )
        month = np.asarray(full.month)
        season = np.full(full.size, "", dtype=object)
        begins_last_year = np.zeros(full.size, dtype=bool)
        for token in np.unique(seasons[seasons != ""]) if seasons is not None else []:
            starts = [
                start
                for start in range(12)
                if len(token) <= 12
                and all(_INITIALS[(start + i) % 12] == c for i, c in enumerate(token))
            ]
            if len(starts) == 1:  # anything else fails the check below
                months = [(starts[0] + i) % 12 + 1 for i in range(len(token))]
                season[np.isin(month, months)] = token
                begins_last_year |= np.isin(month, months) & (month < months[0])
        rebuilt = {
            "month": month,
            "dayofyear": np.asarray(full.dayofyear),
            # As label-group: a group crossing December takes the year it begins.
            "year": np.asarray(full.year) - begins_last_year,
            "month_group": season.astype(str),
        }
        change = np.zeros(full.size, dtype=bool)
        grouped = np.ones(full.size, dtype=bool)
        for b in time_by:
            if not np.array_equal(rebuilt[b][at], ds[b].values):
                raise UsageError(
                    f"by {b!r} does not hold the calendar {b} of each time step, as "
                    "label-group writes it, so its full calendar period is unknown"
                )
            change[1:] |= rebuilt[b][1:] != rebuilt[b][:-1]
            if b == "month_group":
                grouped &= rebuilt[b] != ""
        run = np.cumsum(change)
        touched = np.isin(run, run[at]) & grouped
        if touched[0] or touched[-1]:
            raise UsageError("a time group lasts over a year, so its full period is unknown")
        t = list(dict.fromkeys(ds[b].dims for b in by)).index(("time",))
        for group in groups[t]:
            matches = [rebuilt[b] == label_values[b][i] for b, i in group["labels"].items()]
            group["expected"] = int((touched & np.logical_and.reduce(matches)).sum())

    out = ds.drop_vars(by)
    for var in selected:
        da = ds[var]
        rdims = [d for d in dims if d in da.dims]
        # With by labels, only variables carrying every label dim are summarized.
        if not rdims or not set(group_dims) <= set(da.dims):
            continue
        units = da.pint.units
        over_time = [d for d in rdims if d in ("time", "step")]
        if method == "sum" and over_time and units is not None:
            if dict(units.dimensionality).get("[time]", 0) < 0:
                raise UsageError(
                    f"sum over {', '.join(over_time)} would add up {var!r} rates "
                    f"({da.attrs.get('units', units)}) into a rate, not a total; convert the "
                    "rates to period totals first (convert-to-totals)"
                )
        # Offset units (degC) differ from their deltas: a spread of degC is a delta_degC,
        # and a sum of degC means nothing.
        delta = None if units is None else (1 * units - 1 * units).units
        if method == "sum" and delta != units:
            raise UsageError(
                f"sum of {var!r} in {da.attrs.get('units', units)} is not meaningful; "
                "convert to kelvin first (unit-convert)"
            )
        if method == "std":
            units = delta
        # A variable without the weights' dims takes a plain mean.
        w = weights if weights is not None and set(weights.dims) <= set(da.dims) else None
        da = da.pint.dequantify()  # values are filled into numpy; units go back on below
        kept = [d for d in da.dims if d not in rdims]
        shape = [da.sizes[d] for d in kept] + [label_values[b].size for b in by]
        # Combinations of labels with no cells stay NaN.
        dtype = np.result_type(da.dtype, np.float32, *([] if w is None else [w.dtype]))
        values = np.full(shape, np.nan, dtype=dtype)
        coverage = np.full(shape, np.nan)
        dropped = 0
        for parts in itertools.product(*groups):
            index = {d: i for group in parts for d, i in group["isel"].items()}
            sub = da.isel(index)
            for group in parts:
                if group["box"] is not None:
                    sub = sub.where(group["box"])
            if time_by:
                # Coverage is per input cell (station, grid cell, member): a cell missing
                # too many of the group's calendar steps is left out before any spatial
                # reduction, so cells missing different days never add up to a complete group.
                present = sub.notnull()
                if step_share is not None:
                    present = present * xr.DataArray(step_share[index["time"]], dims="time")
                expected = next(g["expected"] for g in parts if g["expected"] is not None)
                cell_share = present.sum("time") / expected
                complete = cell_share >= min_coverage - 1e-12
                dropped += int(((cell_share > 0) & ~complete).sum())
                sub = sub.where(complete)
                # The least-covered cell that went into each value.
                share = cell_share.where(complete)
                others = [d for d in rdims if d != "time"]
                share = share.min(others) if others else share
            if w is not None:
                # NaN values drop out together with their weights.
                group_w = w.isel({d: i for d, i in index.items() if d in w.dims})
                reduced = sub.weighted(group_w).mean(dim=rdims)
            elif method == "sum":
                reduced = sub.sum(dim=rdims, min_count=1)  # all-NaN → NaN, not 0
            elif method == "std":
                reduced = sub.std(dim=rdims, ddof=1)
            else:
                reduced = getattr(sub, method)(dim=rdims)
            where = {b: i for group in parts for b, i in group["labels"].items()}
            at = (..., *(where[b] for b in by))
            values[at] = reduced.transpose(*kept).values
            if time_by:
                coverage[at] = share.transpose(*kept).values
        coords = {
            **{c: v for c, v in da.coords.items() if set(v.dims) <= set(kept)},
            **{b: label_values[b] for b in by},
        }
        if dropped:
            print(
                f"{var}: left out {dropped} cell-group(s) with data for less than "
                f"min_coverage {min_coverage:g} of the group's calendar time steps",
                file=sys.stderr,
            )
        out[var] = xr.DataArray(
            values if units is None else values * units,
            dims=[*kept, *by],
            coords=coords,
            attrs=da.attrs,
        )
        if time_by and method == "sum" and min_coverage < 1.0:
            # Kept so the output records which sums are estimates.
            partial = int((np.isfinite(values) & (coverage < 1.0 - 1e-12)).sum())
            if partial:
                print(
                    f"WARNING: {partial} {var} sum(s) come from groups missing some time "
                    f"steps (min_coverage {min_coverage:g}); missing steps are not filled, "
                    "so those sums are estimates from partial data and fall short of the "
                    f"full-period total. See {var}_coverage in the output.",
                    file=sys.stderr,
                )
            out[f"{var}_coverage"] = xr.DataArray(
                coverage,
                dims=[*kept, *by],
                coords=coords,
                attrs={
                    "long_name": f"Share of each group's calendar time steps with {var} data",
                    "units": "1",
                },
            )
            out[var].attrs["ancillary_variables"] = f"{var}_coverage"
        # Record the statistic (CF cell_methods). A spread or a sum is not the
        # input quantity, so its long_name must not read "Total precipitation".
        attrs = dict(out[var].attrs)
        cf_method = {"std": "standard_deviation", "min": "minimum", "max": "maximum"}
        step = f"{': '.join(rdims)}: {cf_method.get(method, method)}"
        if by:
            step += f" (grouped by {', '.join(by)})"
        attrs["cell_methods"] = (
            f"{attrs['cell_methods']} {step}" if attrs.get("cell_methods") else step
        )
        if method in ("std", "sum"):
            label = "Standard deviation" if method == "std" else "Sum"
            what = attrs.get("long_name") or attrs.get("standard_name") or var
            attrs["long_name"] = f"{label} of {what} across {', '.join(rdims)}"
            # standard_name stays: CF encodes a spread as the quantity's
            # standard_name plus a cell_methods statistic.
            for key in ("GRIB_name", "short_name", "comment"):
                attrs.pop(key, None)
        out[var].attrs = attrs

    for d in dims:
        if d in out.dims and all(d not in out[v].dims for v in out.data_vars):
            out = out.drop_dims(d)
    return out

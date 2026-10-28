"""Reduce a Dataset with one statistic, over whole dims or per group of label coords."""

import itertools
import sys

import numpy as np
import xarray as xr

from weather_skills_core.errors import DataError, UsageError
from weather_skills_core.standard_dataset import names_for
from weather_skills_core.standard_utils import TIME_LABELS, calendar_expected_steps
from weather_skills_core.units import (
    AGGREGATION_COVERAGE_COORD,
    AGGREGATION_PERIOD_ATTR,
    data_interval_of,
    format_duration,
    infer_timestep,
)

METHODS = ("mean", "std", "min", "max", "sum", "median", "quantile")
# Grid and station dims a sum must not add up (latitude, lon, station_id, ...).
_SPATIAL_DIMS = frozenset((*names_for("lat"), *names_for("lon"), *names_for("point_id")))


def summarize(
    ds, dims=(), by=(), method="mean", q=None, variables=None, weights=None, min_coverage=1.0
):
    """Reduce ``ds`` with one statistic, over whole dims or per group of label coords.

    Modes (pick by which arguments you pass):

    - ``dims`` only: collapse those dims. ``dims=["time"]`` is a time mean per cell.
    - ``by`` only: one value per label. ``by=["region"]`` collapses the dims that
      ``region`` spans (e.g. latitude and longitude) into a ``region`` dim.
    - ``dims`` and ``by``: collapse ``dims`` inside each group as well.
      ``dims=["time"], by=["region"]`` is one value per region over all times.
    - ``weights``: a weighted mean, in any of the modes above.
    - ``method="quantile"`` with ``q``: quantiles, in any of the modes above.
    - ``min_coverage``: with a ``by`` time label, leave out incomplete groups.

    Methods (``METHODS``), all skipping NaN:

    - ``mean``: the mean; with ``weights``, ``sum(w * x) / sum(w)`` over non-NaN values.
    - ``std``: the sample standard deviation (``ddof=1``), NaN from fewer than 2 values.
      Offset units become deltas (``degree_Celsius`` -> ``delta_degC``).
    - ``min``, ``max``, ``median``: as named.
    - ``quantile``: the ``q`` quantiles (linear interpolation, as ``numpy.quantile``),
      along a new ``quantile`` dim holding the ``q`` values. Not weighted.
    - ``sum``: the total, NaN (not 0) if every value is NaN. Refused for rates over
      ``time`` or ``step`` (``mm day-1``), for offset units (``degree_Celsius``), and
      over grid cells or stations (``latitude``, ``longitude``, ``point_id``, directly or
      through a spatial ``by`` label) unless the variable is dimensionless (units ``1``,
      such as 0/1 flags, which sum to a count).

    Args:
        ds: Quantified Dataset, as ``@weather_skill`` passes inputs.
        dims: Dims to collapse.
        by: Label coords to group by: coords that are not dims, such as a county name on
            (latitude, longitude) or a season on time. Labels on the same dims are matched
            together (region and zone give each region-zone pair); labels on different dims
            combine (region and month give each region in each month). The ``""`` label is
            left out, and a combination with no cells is NaN.
        method: One of ``METHODS``.
        q: With ``method="quantile"`` only: quantiles to compute, each in [0, 1].
        variables: Data variables to reduce; default all. The others pass through
            unchanged, as do variables missing a grouped dim.
        weights: DataArray on some of ``ds``'s dims, such as ``latitude_weights`` for
            area. ``method="mean"`` only. A variable without those dims gets a plain mean.
        min_coverage: 0-1, used only with a ``by`` label along time (one of
            ``TIME_LABELS``, as ``calendar_labels`` builds them). For each group, each input cell
            (grid cell, station, member) with data for less than this share of the
            group's full calendar period (all 92 days of a MAM) is left out before cells
            are pooled. Missing steps are never filled. Default 1.0: complete groups only.
            A repeated time step is refused, as is a step longer than a day that runs into
            the next group (a week from 27 May under ``MAM``).

    Returns:
        Quantified Dataset. Collapsed dims are gone and each ``by`` label is a dim (after
        a ``quantile`` dim, for ``method="quantile"``). Other
        label coords on the grouped dims are dropped. ``cell_methods`` records the step
        (``time: mean (grouped by region)``). A ``sum`` kept below full coverage adds
        ``<var>_coverage``. Left-out cells and partial sums are reported on stderr.

    Raises:
        UsageError: an unknown dim, label, variable or method; labels that share some but
            not all dims; weights with a method other than mean; ``q`` missing, out of
            [0, 1], or given without ``method="quantile"``; a refused sum; a time
            label whose calendar period cannot be counted; a repeated time step or a step
            running into the next group (with a ``by`` time label).
        DataError: a label whose every value is ``""``; labels on the same dims that
            never meet (no cell has a non-empty value of each).
    """
    if method not in METHODS:
        raise UsageError(f"method {method!r} is not one of {', '.join(METHODS)}")
    if method == "quantile":
        if not q:
            raise UsageError("method quantile needs q, the quantiles to compute (e.g. 0.1 0.9)")
        if any(not 0.0 <= x <= 1.0 for x in q):
            raise UsageError(f"q must each be in [0, 1]; got {list(q)}")
        if "quantile" in ds.dims:
            raise UsageError("dataset already has a quantile dim")
    elif q:
        raise UsageError(f"q is only used with method quantile, not {method}")
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
            raise UsageError(
                f"weights need method mean, not {method}"
                + ("; a weighted quantile is not supported" if method == "quantile" else "")
            )
        if not set(weights.dims) <= set(ds.dims):
            raise UsageError(f"weights dims {weights.dims} are not all dataset dims")
    if not 0.0 <= min_coverage <= 1.0:
        raise UsageError(f"min_coverage must be in [0, 1]; got {min_coverage}")
    time_by = [b for b in by if "time" in ds[b].dims]
    for b in time_by:
        if b not in TIME_LABELS or ds[b].dims != ("time",):
            raise UsageError(
                f"by {b!r} runs along time but is not a calendar time label "
                f"({', '.join(TIME_LABELS)}), so its full calendar period is unknown"
            )
    selected = list(dict.fromkeys(variables)) if variables else list(ds.data_vars)
    for var in selected:
        if var not in ds.data_vars:
            raise UsageError(f"variable {var!r} not in dataset (have {list(ds.data_vars)})")
    # The dims a label spans collapse into one entry per label.
    group_dims = list(dict.fromkeys(d for b in by for d in ds[b].dims))
    dims = list(dict.fromkeys([*group_dims, *dims]))
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
        if not entries:
            raise DataError(
                f"by {', '.join(names)}: no cell or step has a non-empty value of every one "
                "of these labels at once; nothing to group"
            )
        groups.append(entries)

    # Coverage: the steps a complete group holds, from the calendar of the time labels.
    if time_by:
        period = next(
            (
                ds[v].attrs[AGGREGATION_PERIOD_ATTR]
                for v in ds.data_vars
                if isinstance(ds[v].attrs.get(AGGREGATION_PERIOD_ATTR), str)
            ),
            None,
        ) or data_interval_of(ds)
        seasons = ds["month_group"].values if "month_group" in ds.coords else np.array([])
        expected = calendar_expected_steps(
            ds.indexes["time"],
            period or format_duration(infer_timestep(ds, "time")),
            {b: ds[b].values for b in time_by},
            list(np.unique(seasons[seasons != ""])),
        )
        t = list(dict.fromkeys(ds[b].dims for b in by)).index(("time",))
        for group in groups[t]:
            key = tuple(label_values[b][i] for b, i in group["labels"].items())
            group["expected"] = expected[key]
        # An input interval that was itself 3/7 complete counts as 3/7 of a step.
        step_share = None
        if AGGREGATION_COVERAGE_COORD in ds.coords and ds[AGGREGATION_COVERAGE_COORD].dims == (
            "time",
        ):
            step_share = np.nan_to_num(ds[AGGREGATION_COVERAGE_COORD].values.astype(float))

    # Other label coords on the grouped dims (admin_1_name, ...) cannot survive the collapse.
    out = ds.drop_vars(
        [c for c in ds.coords if c not in ds.dims and set(ds[c].dims) & set(group_dims)]
    )
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
        # Adding up cells or stations of a physical quantity (mm, degC) grows with the
        # area and means nothing; dimensionless counts and flags add up to how many.
        over_space = [d for d in rdims if d in _SPATIAL_DIMS]
        if method == "sum" and over_space and (units is None or units.dimensionality):
            spatial_by = [b for b in by if set(ds[b].dims) & set(over_space)]
            via = f" (via by {', '.join(spatial_by)})" if spatial_by else ""
            raise UsageError(
                f"sum over {', '.join(over_space)}{via} would add up {var!r} "
                f"({'no units' if units is None else format(units, '~')}) across cells or "
                "stations; use method mean for a regional value, after summing over time in "
                "a separate call. Only dimensionless variables (units 1) sum over space"
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
        # quantile adds a leading quantile dim, as xarray's quantile does.
        qdims = ["quantile"] if method == "quantile" else []
        # Combinations of labels with no cells stay NaN.
        dtype = np.result_type(da.dtype, np.float32, *([] if w is None else [w.dtype]))
        values = np.full([len(q), *shape] if qdims else shape, np.nan, dtype=dtype)
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
            elif method == "quantile":
                reduced = sub.quantile(list(q), dim=rdims, skipna=True)
            else:
                reduced = getattr(sub, method)(dim=rdims)
            where = {b: i for group in parts for b, i in group["labels"].items()}
            at = (..., *(where[b] for b in by))
            values[at] = reduced.transpose(*qdims, *kept).values
            if time_by:
                coverage[at] = share.transpose(*kept).values
        coords = {
            **{c: v for c, v in da.coords.items() if set(v.dims) <= set(kept)},
            **{b: label_values[b] for b in by},
            **({"quantile": np.asarray(q, dtype=float)} if qdims else {}),
        }
        if dropped:
            print(
                f"{var}: left out {dropped} cell-group(s) with data for less than "
                f"min_coverage {min_coverage:g} of the group's calendar time steps",
                file=sys.stderr,
            )
        out[var] = xr.DataArray(
            values if units is None else values * units,
            dims=[*qdims, *kept, *by],
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
        notes = [f"q: {' '.join(f'{x:g}' for x in q)}"] if qdims else []
        notes += [f"grouped by {', '.join(by)}"] if by else []
        if notes:
            step += f" ({'; '.join(notes)})"
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

"""Tests for units equivalence, quantify, aggregation_period, and standard conversion."""

import numpy as np
import pytest
import xarray as xr
from conftest import make_forecast, make_gridded

from weather_skills_core.errors import UsageError
from weather_skills_core.units import (
    AGGREGATION_COVERAGE_COORD,
    AGGREGATION_PERIOD_ATTR,
    DATA_INTERVAL_ATTR,
    PRECIP_AMOUNT_LONG_NAME,
    STANDARD,
    assert_nonoverlapping_intervals,
    classify_variable,
    convert_dataarray,
    convert_values,
    deaccumulate_along_step,
    dequantify_dataset,
    expected_samples_in_period,
    filter_min_coverage,
    format_cell_methods,
    format_duration,
    infer_timestep,
    interval_days,
    looks_like_rate_display_name,
    parse_aggregation_period,
    precip_amounts_to_rates,
    quantify_dataset,
    rate_to_total,
    stamp_data_interval,
    stamp_precip_amounts,
    to_standard_units,
    units_convertible,
    units_equal,
    ureg,
)


def test_units_equal_spelling():
    """Pint-equivalent unit strings compare equal; rates and amounts do not."""
    assert units_equal("mm/day", "mm day-1")
    assert units_equal("degC", "degree_Celsius")
    assert not units_equal("mm", "mm day-1")


def test_pentad_dekad_registry():
    """pentad and dekad convert to day on the units registry."""
    assert ureg.Quantity(1, "mm/pentad").to("mm/day").magnitude == pytest.approx(0.2)
    assert parse_aggregation_period("1 dekad").to("day").magnitude == pytest.approx(10.0)
    assert parse_aggregation_period("7 day").to("day").magnitude == pytest.approx(7.0)


def test_convert_values_temp_and_precip_density():
    """K→°C and kg m-2 flux/amount convert with the water-density path."""
    k, _ = convert_values(np.array([273.15]), "K", STANDARD["temp"]["units"])
    np.testing.assert_allclose(k, [0.0], atol=1e-6)
    mm, density_converted = convert_values(
        np.array([1.0]), "kg m-2", STANDARD["precip_amount"]["units"]
    )
    assert density_converted
    np.testing.assert_allclose(mm, [1.0], atol=1e-6)
    rate, density_converted = convert_values(
        np.array([1e-3]), "kg m-2 s-1", STANDARD["precip"]["units"]
    )
    assert density_converted
    np.testing.assert_allclose(rate, [86.4], rtol=1e-5)


def test_classify_variable():
    """Name, units, and standard_name classify temp vs precip vs unknown."""
    assert classify_variable("t2m", units="K", standard_name="air_temperature") == "temp"
    assert classify_variable("2m_temperature", units="K") == "temp"
    assert classify_variable("tas", units="K") == "temp"
    assert classify_variable("tp", units="kg m-2", standard_name="precipitation_amount") == (
        "precip_amount"
    )
    assert classify_variable("precip", units="mm/day") == "precip"
    assert classify_variable("tp", units="kg m-2 s-1") == "precip"  # named hint
    assert classify_variable("tp", units="kg m-2") == "precip_amount"  # amount units win
    assert classify_variable("flux", units="kg m-2 s-1") is None  # units alone are not enough
    assert classify_variable("humidity", units="1") is None
    # Prefix "precip" must not classify a dimensionless companion field.
    assert classify_variable("precipitation_quality_index_surface", units="1") is None
    assert classify_variable("precipitation_surface", units="kg m-2 s-1") == "precip"
    assert classify_variable("precipitation_surface", units="mm/h") == "precip"
    # Not air / 2 m temperature
    assert classify_variable("sst", units="K", standard_name="sea_surface_temperature") is None
    assert classify_variable("d2m", units="K", standard_name="dew_point_temperature") is None
    assert classify_variable("skt", units="K", standard_name="surface_temperature") is None


def test_to_standard_units_converts_amount_named_tp_to_mm():
    """Named tp in kg m-2 becomes mm with the amount standard_name."""
    ds = make_gridded(name="tp", fill=2.0, units="kg m-2")
    out = to_standard_units(ds)
    assert out["tp"].attrs["units"] == STANDARD["precip_amount"]["units"]
    assert out["tp"].attrs["standard_name"] == STANDARD["precip_amount"]["standard_name"]
    np.testing.assert_allclose(out["tp"].values, 2.0, atol=1e-6)


def test_to_standard_units_name_hint_sets_standard_name_keeps_name():
    """A rate named tp keeps its name and gets the rate standard_name."""
    ds = make_gridded(name="tp", fill=1.0, units="mm/day")
    out = to_standard_units(ds)
    assert "tp" in out.data_vars
    assert list(out.data_vars) == ["tp"]
    assert out["tp"].attrs["units"] == STANDARD["precip"]["units"]
    assert out["tp"].attrs["standard_name"] == "lwe_precipitation_rate"


def test_to_standard_units_normalizes_amount_and_temp():
    """Amount precip and air temp convert to display units."""
    ds = make_gridded(name="tp", fill=2.0, units=None)
    ds["tp"].attrs.update(units="kg m-2", standard_name="precipitation_amount")
    out = to_standard_units(ds)
    assert out["tp"].attrs["units"] == STANDARD["precip_amount"]["units"]
    assert out["tp"].attrs["standard_name"] == STANDARD["precip_amount"]["standard_name"]

    tds = make_gridded(name="t2m", fill=300.0, units="K")
    tds["t2m"].attrs["standard_name"] = "air_temperature"
    tout = to_standard_units(tds)
    assert tout["t2m"].attrs["units"] == STANDARD["temp"]["units"]
    np.testing.assert_allclose(tout["t2m"].values, 300.0 - 273.15, rtol=1e-5)


def test_to_standard_units_noop_unknown():
    """An unclassified variable is left unchanged."""
    ds = make_gridded(name="humidity", fill=0.5, units="1")
    out = to_standard_units(ds)
    assert out["humidity"].attrs["units"] == "1"


def test_to_standard_units_skips_dimensionless_precip_named_companion():
    """A dimensionless precip-named companion is not converted."""
    ds = make_gridded(name="precipitation_quality_index_surface", fill=0.8, units="1")
    out = to_standard_units(ds)
    assert out["precipitation_quality_index_surface"].attrs["units"] == "1"
    np.testing.assert_allclose(out["precipitation_quality_index_surface"].values, 0.8)


def test_units_convertible():
    """Flux and mm/h convert to mm day-1; dimensionless does not."""
    assert units_convertible("kg m-2 s-1", "mm day-1")
    assert units_convertible("mm/h", "mm day-1")
    assert not units_convertible("1", "mm day-1")


def test_pressure_is_a_standard_kind_in_hpa():
    """MSLP / surface pressure in Pa convert to hPa; pressure-named anomalies do not."""
    sn = "air_pressure_at_mean_sea_level"
    assert classify_variable("msl", units="Pa", standard_name=sn) == "pressure"
    assert classify_variable("sp", units="Pa", standard_name="surface_air_pressure") == "pressure"
    assert classify_variable("prmsl", units="Pa") == "pressure"
    assert classify_variable("msl", units="1") is None  # a normalized anomaly
    assert classify_variable("speed", units="m s-1") is None
    assert classify_variable("spi", units="1") is None

    ds = make_gridded(name="msl", fill=101325.0, units="Pa")
    ds["msl"].attrs["standard_name"] = sn
    out = to_standard_units(ds)
    assert out["msl"].attrs["units"] == "hPa"
    assert out["msl"].attrs["standard_name"] == sn
    np.testing.assert_allclose(out["msl"].values, 1013.25)
    # A Pa difference (a pressure anomaly from `difference`) scales the same way.
    diff = make_gridded(name="msl", fill=-250.0, units="Pa")
    np.testing.assert_allclose(to_standard_units(diff)["msl"].values, -2.5)
    assert (
        to_standard_units(make_gridded(name="msl", fill=1.5, units="1"))["msl"].attrs["units"]
        == "1"
    )


def test_to_standard_units_missing_variable():
    """A missing --variable name is a UsageError."""
    ds = make_gridded()
    with pytest.raises(UsageError, match="not in dataset"):
        to_standard_units(ds, variables=["nope"])


def test_to_standard_units_normalizes_already_standard_spelling():
    """mm/day spelling is rewritten without changing values."""
    ds = make_gridded(name="precip", fill=1.0, units="mm/day")
    ds["precip"].attrs["standard_name"] = "precipitation_flux"
    values_before = ds["precip"].values.copy()
    out = to_standard_units(ds)
    assert out["precip"].attrs["units"] == STANDARD["precip"]["units"]
    assert out["precip"].attrs["standard_name"] == "lwe_precipitation_rate"
    np.testing.assert_array_equal(out["precip"].values, values_before)


def test_to_standard_units_raises_when_classified_but_not_convertible():
    """Temp classified with wind units is a UsageError."""
    ds = make_gridded(name="t2m", fill=1.0, units="m s-1")
    ds["t2m"].attrs["standard_name"] = "air_temperature"
    with pytest.raises(UsageError, match="not convertible"):
        to_standard_units(ds)


def test_quantify_dataset_requires_units_for_temp_precip():
    """Temp/precip without units refuse to quantify."""
    ds = make_gridded(name="precip", units=None)
    with pytest.raises(UsageError, match="requires a units attribute"):
        quantify_dataset(ds)


def test_quantify_dataset_accepts_amount_totals():
    """mm amounts quantify."""
    ds = make_gridded(name="precip", units="mm")
    q = quantify_dataset(ds)
    assert q["precip"].pint.units is not None


def test_quantify_dataset_preserves_encoding_source():
    """Zarr ``encoding['source']`` survives pint.quantify for legend filenames."""
    ds = make_gridded()
    ds.encoding["source"] = "/tmp/ta00072.zarr"
    q = quantify_dataset(ds)
    assert q.encoding.get("source") == "/tmp/ta00072.zarr"


def test_quantify_dataset_accepts_cell_methods_sum():
    """A summed rate still quantifies."""
    ds = make_gridded(name="precip", units="mm day-1")
    ds["precip"].attrs["cell_methods"] = "time: sum"
    q = quantify_dataset(ds)
    assert q["precip"].pint.units is not None


def test_rate_to_total_refuses_amount_totals():
    """rate_to_total refuses variables that are already amounts."""
    ds = make_gridded(name="precip", units="mm")
    q = quantify_dataset(ds)
    with pytest.raises(UsageError, match="precip total"):
        rate_to_total(q["precip"], "1 day")


def test_rate_to_total_refuses_cell_methods_sum():
    """rate_to_total refuses cell_methods sum (already a total)."""
    ds = make_gridded(name="precip", units="mm day-1")
    ds["precip"].attrs["cell_methods"] = "time: sum"
    q = quantify_dataset(ds)
    with pytest.raises(UsageError, match="precip total"):
        rate_to_total(q["precip"], "1 day")


def test_quantify_dataset_passes_unitless_other_vars():
    """Non-temp/precip vars may stay unitless."""
    ds = make_gridded(name="humidity", units=None)
    ds["humidity"].attrs.pop("units", None)
    q = quantify_dataset(ds)
    assert q["humidity"].pint.units is None
    assert not hasattr(q["humidity"].data, "units")


def test_quantify_dataset_quantifies_and_preserves_coord_attrs():
    """Data vars quantify; coordinate unit attrs are preserved."""
    ds = make_gridded(name="precip", units="mm day-1")
    ds["latitude"].attrs["units"] = "degrees_north"
    q = quantify_dataset(ds)
    assert q["precip"].pint.units is not None
    assert q["latitude"].pint.units is None
    assert q["latitude"].attrs["units"] == "degrees_north"
    plain = dequantify_dataset(q)
    assert plain["precip"].attrs["units"] == "mm day-1"


def test_dequantify_dataset_spells_units_standard_or_cf():
    """Converted units keep STANDARD spellings, else CF; never pint long form."""
    ds = make_gridded(name="precip", units="mm hr-1")
    ds["temp"] = ds["precip"].copy()
    ds["temp"].attrs = {"units": "K"}
    ds["wind"] = ds["precip"].copy()
    ds["wind"].attrs = {"units": "m s-1"}
    q = quantify_dataset(ds)
    q["precip"] = q["precip"].pint.to("mm day-1")
    q["temp"] = q["temp"].pint.to("degree_Celsius")
    plain = dequantify_dataset(q)
    assert plain["precip"].attrs["units"] == "mm day-1"
    assert plain["temp"].attrs["units"] == "degree_Celsius"
    assert plain["wind"].attrs["units"] == "m s-1"


def test_format_cell_methods():
    """cell_methods strings include an optional interval clause."""
    assert format_cell_methods("time", "mean") == "time: mean"
    assert format_cell_methods("time", "mean", interval="1 day") == ("time: mean (interval: 1 day)")


def test_timestep_gate_and_rate_to_total():
    """Weekly bins convert; daily labels refuse a 7-day period."""
    times = np.array(["2026-01-07", "2026-01-14"], dtype="datetime64[D]")
    ds = xr.Dataset(
        {
            "precip": (
                ("time",),
                [1.0, 2.0],
                {AGGREGATION_PERIOD_ATTR: "7 day", "units": "mm day-1"},
            )
        },
        coords={"time": times},
    )
    assert_nonoverlapping_intervals(ds, "time", "7 day")
    q = quantify_dataset(ds)
    total = rate_to_total(q["precip"], "7 day")
    np.testing.assert_allclose(total.pint.dequantify().values, [7.0, 14.0])

    daily = xr.Dataset(
        {"precip": (("time",), np.ones(7), {"units": "mm day-1"})},
        coords={"time": np.arange("2026-01-01", "2026-01-08", dtype="datetime64[D]")},
    )
    with pytest.raises(UsageError, match="non-overlapping subset"):
        assert_nonoverlapping_intervals(daily, "time", "7 day")


def test_timestep_gate_allows_singleton():
    """One aggregated bin has no adjacent labels; conversion is still well-defined."""
    ds = xr.Dataset(
        {
            "precip": (
                ("time",),
                [1.0],
                {AGGREGATION_PERIOD_ATTR: "7 day", "units": "mm day-1"},
            )
        },
        coords={"time": np.array(["2026-01-07"], dtype="datetime64[D]")},
    )
    assert_nonoverlapping_intervals(ds, "time", "7 day")
    q = quantify_dataset(ds)
    total = rate_to_total(q["precip"], "7 day")
    np.testing.assert_allclose(total.pint.dequantify().values, [7.0])

    step_ds = xr.Dataset(
        {"precip": (("step",), [2.0], {"units": "mm day-1"})},
        coords={"step": np.array([np.timedelta64(7, "D")])},
    )
    assert_nonoverlapping_intervals(step_ds, "step", "7 day")


def test_infer_timestep_timedelta64_us_weekly():
    """dynamical-fetch writes step as timedelta64[us]; must not read 7 day as 0.007 d."""
    steps = (np.arange(1, 5) * np.timedelta64(7, "D")).astype("timedelta64[us]")
    ds = xr.Dataset(
        {"precip": (("step",), np.ones(4), {"units": "mm day-1"})},
        coords={"step": steps},
    )
    dt = infer_timestep(ds, "step")
    assert abs(float(dt.to("day").magnitude) - 7.0) < 1e-9
    assert_nonoverlapping_intervals(ds, "step", "7 day")


def test_format_duration_subday():
    """Durations format as pint-style 'N unit' strings."""
    assert format_duration(ureg.Quantity("30 minute")) == "30 minute"
    assert format_duration(ureg.Quantity("7 day")) == "7 day"
    assert format_duration(parse_aggregation_period("1 hour")) == "1 hour"


def test_stamp_data_interval_explicit_and_inferred():
    """An explicit period stamps; otherwise daily spacing is inferred."""
    ds = make_gridded(n_time=2)
    stamp_data_interval(ds, period="30 minute")
    assert ds["precip"].attrs[DATA_INTERVAL_ATTR] == "30 minute"
    assert AGGREGATION_PERIOD_ATTR not in ds["precip"].attrs
    assert AGGREGATION_COVERAGE_COORD not in ds.coords

    daily = make_gridded(n_time=3)
    stamp_data_interval(daily)
    assert daily["precip"].attrs[DATA_INTERVAL_ATTR] == "1 day"


def test_deaccumulate_two_steps_stamps_the_one_remaining_cell():
    """Two cumulative steps leave one cell, [0, 7 day), stamped from the dropped lead."""
    steps = np.array([0, 7], dtype="timedelta64[D]").astype("timedelta64[ns]")
    ds = xr.Dataset(
        {"tp": (("step",), np.array([0.0, 14.0]), {"units": "mm"})},
        coords={"step": steps, "time": np.datetime64("2026-01-01", "ns")},
    )
    out = deaccumulate_along_step(quantify_dataset(ds))
    assert out.sizes["step"] == 1
    assert out["tp"].attrs[DATA_INTERVAL_ATTR] == "7 day"


def test_deaccumulate_irregular_steps_write_left_labeled_bounds():
    """Uneven leads get CF bounds: each cell runs from its step to the next input step."""
    hours = np.array([0, 3, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    ds = xr.Dataset(
        {"tp": (("step",), np.array([0.0, 1.0, 2.0, 4.0]), {"units": "mm"})},
        coords={"step": hours, "time": np.datetime64("2026-01-01", "ns")},
    )
    out = deaccumulate_along_step(quantify_dataset(ds))
    assert DATA_INTERVAL_ATTR not in out["tp"].attrs
    np.testing.assert_array_equal(out["step_bounds"].values[:, 0], hours[:-1])
    np.testing.assert_array_equal(out["step_bounds"].values[:, 1], hours[1:])


def test_stamp_data_interval_singleton_step_without_origin_is_noop():
    """A lone step cannot infer spacing; keep any existing interval and do not error."""
    step = np.array([np.timedelta64(7, "D")], dtype="timedelta64[ns]")
    ds = xr.Dataset(
        {
            "tp": (
                ("step",),
                np.ones(1),
                {"units": "mm day-1", DATA_INTERVAL_ATTR: "6 hour"},
            )
        },
        coords={"step": step, "time": np.datetime64("2026-01-01", "ns")},
    )
    out = stamp_data_interval(ds)
    assert out["tp"].attrs[DATA_INTERVAL_ATTR] == "6 hour"

    bare = xr.Dataset(
        {"tp": (("step",), np.ones(1), {"units": "mm day-1"})},
        coords={"step": step, "time": np.datetime64("2026-01-01", "ns")},
    )
    kept = stamp_data_interval(bare)
    assert DATA_INTERVAL_ATTR not in kept["tp"].attrs


def test_stamp_data_interval_irregular_step_writes_cf_bounds():
    """Irregular step with no existing geometry writes CF bounds."""
    days = np.array([7, 10, 14, 21], dtype="timedelta64[D]").astype("timedelta64[ns]")
    ds = xr.Dataset(
        {"tp": (("step",), np.ones(4), {"units": "mm day-1"})},
        coords={"step": days},
    )
    out = stamp_data_interval(ds)
    assert DATA_INTERVAL_ATTR not in out["tp"].attrs
    assert out["step"].attrs["bounds"] == "step_bounds"
    bounds = np.asarray(out["step_bounds"].values)
    assert bounds.shape == (4, 2)
    np.testing.assert_array_equal(bounds[:, 1], days)
    np.testing.assert_array_equal(
        bounds[:, 0],
        np.array([0, 7, 10, 14], dtype="timedelta64[D]").astype("timedelta64[ns]"),
    )


def test_stamp_data_interval_keeps_existing_when_ticks_would_differ():
    """A stamped 7-day interval survives select of two leads 14 days apart."""
    days = np.array([7, 21], dtype="timedelta64[D]").astype("timedelta64[ns]")
    ds = xr.Dataset(
        {"tp": (("step",), np.ones(2), {"units": "mm day-1", DATA_INTERVAL_ATTR: "7 day"})},
        coords={"step": days, "time": np.datetime64("2026-01-01", "ns")},
    )
    out = stamp_data_interval(ds)
    assert out["tp"].attrs[DATA_INTERVAL_ATTR] == "7 day"
    assert "step_bounds" not in out.variables


def test_stamp_data_interval_explicit_period_overwrites():
    """period= is the product telling us the interval; it replaces a prior stamp."""
    ds = make_gridded(n_time=3)
    ds["precip"].attrs[DATA_INTERVAL_ATTR] = "1 day"
    out = stamp_data_interval(ds, period="30 minute")
    assert out["precip"].attrs[DATA_INTERVAL_ATTR] == "30 minute"


def test_deaccumulate_origin_restamps_existing_interval():
    """Deaccumulate changes cell geometry, so origin overrides a leftover interval."""
    ds = make_forecast(n_number=None, n_step=4)
    ds["tp"].attrs.update(
        units="mm",
        standard_name="lwe_thickness_of_precipitation_amount",
        **{DATA_INTERVAL_ATTR: "6 hour"},
    )
    out = deaccumulate_along_step(ds)
    assert out["tp"].attrs[DATA_INTERVAL_ATTR] == "1 day"


def test_stamp_data_interval_irregular_datetime_defaults_origin():
    """Irregular times bound the first bin from midnight of that day."""
    times = np.array(
        ["2026-08-18T03:00", "2026-08-18T06:00", "2026-08-18T10:00"],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        {"tp": (("time",), np.ones(3), {"units": "mm day-1"})},
        coords={"time": times},
    )
    out = stamp_data_interval(ds)
    bounds = np.asarray(out["time_bounds"].values)
    assert bounds[0, 0] == np.datetime64("2026-08-18T00:00")
    assert bounds[0, 1] == times[0]


def test_stamp_data_interval_datetime_gaps_stay_regular():
    """Missing hours are missing samples, not wider cells stretched over the gap."""
    times = np.array(
        ["2026-08-18T00:00", "2026-08-18T01:00", "2026-08-18T04:00", "2026-08-18T05:00"],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        {"pm25": (("time",), np.ones(4), {"units": "ug m-3"})},
        coords={"time": times},
    )
    out = stamp_data_interval(ds)
    assert "time_bounds" not in out
    assert out["pm25"].attrs["data_interval"] == "1 hour"


def test_assert_nonoverlapping_uses_min_spacing_when_bounds_present():
    """Median gap can hide overlap; bounds path uses the minimum label spacing."""
    steps = np.array([7, 10, 14, 21], dtype="timedelta64[D]")
    ds = stamp_data_interval(
        xr.Dataset(
            {"tp": (("step",), np.ones(4), {"units": "mm day-1"})},
            coords={"step": steps},
        )
    )
    with pytest.raises(UsageError, match="overlapping"):
        assert_nonoverlapping_intervals(ds, "step", "7 day")


def test_deaccumulate_stamps_scalar_on_daily_step():
    """Daily accumulated tp deaccumulates with a scalar data_interval."""
    ds = make_forecast(n_number=None, n_step=4)
    ds["tp"].attrs.update(units="mm", standard_name="lwe_thickness_of_precipitation_amount")
    out = deaccumulate_along_step(ds)
    assert out["tp"].attrs[DATA_INTERVAL_ATTR] == "1 day"
    assert "step_bounds" not in out.variables
    np.testing.assert_array_equal(
        np.asarray(out["step"].values).astype("timedelta64[D]"),
        np.array([0, 1, 2], dtype="timedelta64[D]"),
    )


def test_deaccumulate_two_steps_stamps_remaining_interval():
    """Deaccumulating two leads leaves one step whose interval is still known."""
    ds = make_forecast(n_number=None, n_step=2)
    ds["tp"].attrs.update(units="mm", standard_name="lwe_thickness_of_precipitation_amount")
    ds["tp"].values[:] = np.array([[[1.0, 1.0], [1.0, 1.0]], [[3.0, 3.0], [3.0, 3.0]]])
    out = deaccumulate_along_step(ds)
    assert out.sizes["step"] == 1
    assert out["tp"].attrs[DATA_INTERVAL_ATTR] == "1 day"
    np.testing.assert_allclose(out["tp"].values, 2.0)
    assert out["step"].values[0].astype("timedelta64[D]") == np.timedelta64(0, "D")


def test_deaccumulate_stamps_bounds_on_irregular_step():
    """Irregular step deaccumulation writes CF bounds on left-labeled cells."""
    days = [0, 7, 10, 14, 20, 21, 28]
    steps = np.array(days, dtype="timedelta64[D]")
    accum = np.cumsum(np.arange(len(days), dtype=float))
    ds = xr.Dataset(
        {
            "tp": (
                ("step",),
                accum,
                {"units": "mm", "standard_name": "lwe_thickness_of_precipitation_amount"},
            )
        },
        coords={"step": steps},
    )
    out = deaccumulate_along_step(ds)
    assert DATA_INTERVAL_ATTR not in out["tp"].attrs
    assert out["step"].attrs["bounds"] == "step_bounds"
    np.testing.assert_array_equal(
        np.asarray(out["step"].values).astype("timedelta64[D]"),
        np.array([0, 7, 10, 14, 20, 21], dtype="timedelta64[D]"),
    )
    bounds = np.asarray(out["step_bounds"].values)
    # Coord is the left edge; origin is the dropped last sample (28d).
    assert bounds[0, 0] == np.timedelta64(0, "D")
    assert bounds[0, 1] == np.timedelta64(7, "D")
    assert bounds[1, 0] == np.timedelta64(7, "D")
    assert bounds[1, 1] == np.timedelta64(10, "D")
    assert bounds[-1, 0] == np.timedelta64(21, "D")
    assert bounds[-1, 1] == np.timedelta64(28, "D")


def test_expected_samples_and_filter_min_coverage():
    """Coverage filtering drops bins below the threshold and refuses an empty axis."""
    assert expected_samples_in_period("7 day", "30 minute") == 336
    assert expected_samples_in_period("21 day", "1 day") == 21

    times = np.array(["2026-01-21", "2026-02-11"], dtype="datetime64[D]")
    ds = xr.Dataset(
        {
            "precip": (
                ("time",),
                [1.0, 2.0],
                {
                    AGGREGATION_PERIOD_ATTR: "21 day",
                    DATA_INTERVAL_ATTR: "1 day",
                    "units": "mm day-1",
                },
            )
        },
        coords={
            "time": times,
            AGGREGATION_COVERAGE_COORD: ("time", [0.9, 1.0]),
        },
    )
    with pytest.raises(UsageError, match="min-coverage"):
        filter_min_coverage(ds.isel(time=slice(0, 1)), "time", 1.0)
    kept = filter_min_coverage(ds, "time", 0.6)
    assert kept.sizes["time"] == 2
    one = filter_min_coverage(ds, "time", 1.0)
    assert one.sizes["time"] == 1

    empty = ds.isel(time=slice(0, 0))
    with pytest.raises(UsageError, match="empty time axis"):
        filter_min_coverage(empty, "time", 1.0)
    assert float(one["precip"].values[0]) == pytest.approx(2.0)


def test_looks_like_rate_display_name():
    """Rate-like long_names match; totals and empty strings do not."""
    assert looks_like_rate_display_name("precipitation rate")
    assert looks_like_rate_display_name("Precipitation rate")
    assert looks_like_rate_display_name("daily precipitation rate")
    assert looks_like_rate_display_name("precipitation_flux")
    assert not looks_like_rate_display_name("Total precipitation")
    assert not looks_like_rate_display_name("CHIRPS daily precipitation")
    assert not looks_like_rate_display_name("")
    assert not looks_like_rate_display_name(None)


def test_stamp_precip_amounts_overwrites_rate_display_names():
    """Rate display names become amount names; a real product long_name is kept."""
    ds = make_gridded(n_time=1, fill=1.0, units="mm")
    ds["precip"].attrs.update(
        standard_name="lwe_precipitation_rate",
        long_name="precipitation rate",
        GRIB_name="Precipitation rate",
    )
    out = stamp_precip_amounts(ds)
    assert out["precip"].attrs["standard_name"] == STANDARD["precip_amount"]["standard_name"]
    assert out["precip"].attrs["long_name"] == PRECIP_AMOUNT_LONG_NAME
    assert out["precip"].attrs["GRIB_name"] == PRECIP_AMOUNT_LONG_NAME

    kept = make_gridded(n_time=1, fill=1.0, units="mm")
    kept["precip"].attrs["long_name"] = "CHIRPS daily precipitation"
    stamped = stamp_precip_amounts(kept)
    assert stamped["precip"].attrs["long_name"] == "CHIRPS daily precipitation"


def test_stamp_precip_amounts_mass_units_are_precipitation_amount():
    """kg m-2 is a mass per area: CF precipitation_amount, not lwe_thickness (a depth)."""
    ds = make_gridded(n_time=1, fill=1.0, units="kg m-2")
    ds["precip"].attrs["standard_name"] = "precipitation_amount"
    assert stamp_precip_amounts(ds)["precip"].attrs["standard_name"] == "precipitation_amount"
    depth = make_gridded(n_time=1, fill=1.0, units="m")
    assert stamp_precip_amounts(depth)["precip"].attrs["standard_name"] == (
        "lwe_thickness_of_precipitation_amount"
    )


def test_precip_amounts_to_rates_daily_and_step():
    """Daily amounts stay numerically; step amounts become per-step rates."""
    daily = make_gridded(n_time=2, fill=4.0, units="mm")
    daily["precip"].attrs["standard_name"] = "precipitation_amount"
    out = precip_amounts_to_rates(daily, interval="1 day")
    assert out["precip"].attrs["units"] == STANDARD["precip"]["units"]
    np.testing.assert_allclose(out["precip"].values, daily["precip"].values)

    steps = np.array([np.timedelta64(d, "D") for d in (1, 2, 3)])
    tp = xr.Dataset(
        {
            "tp": (
                ("step",),
                np.array([1.0, 3.0, 6.0]),
                {"units": "mm", "standard_name": "precipitation_amount"},
            )
        },
        coords={"step": steps},
    )
    rates = precip_amounts_to_rates(tp)
    assert rates.sizes["step"] == 2
    np.testing.assert_array_equal(
        np.asarray(rates["step"].values).astype("timedelta64[D]"),
        np.array([1, 2], dtype="timedelta64[D]"),
    )
    np.testing.assert_allclose(rates["tp"].values, [2.0, 3.0])
    assert rates["tp"].attrs["units"] == STANDARD["precip"]["units"]

    mixed = tp.assign(t2m=("step", [280.0, 281.0, 282.0]))
    mixed["t2m"].attrs.update(units="K", standard_name="air_temperature")
    mixed_out = precip_amounts_to_rates(mixed)
    assert mixed_out.sizes["step"] == 2
    np.testing.assert_allclose(mixed_out["tp"].values, [2.0, 3.0])
    np.testing.assert_allclose(mixed_out["t2m"].values, [280.0, 281.0])

    daily_step = precip_amounts_to_rates(tp, interval="1 day", deaccumulate=False)
    assert daily_step.sizes["step"] == 3
    np.testing.assert_allclose(daily_step["tp"].values, [1.0, 3.0, 6.0])


def test_precip_convertible_names_skips_temp_with_leftover_precip_standard_name():
    """A temperature var with a leftover precip standard_name is not treated as precip."""
    ds = make_gridded(n_time=2, name="2m_temperature", fill=280.0, units="K")
    ds["2m_temperature"].attrs["standard_name"] = "lwe_precipitation_rate"
    from weather_skills_core.units import precip_convertible_names

    assert precip_convertible_names(ds) == []

    rain = make_gridded(n_time=2, fill=1.0, units="mm day-1")
    assert precip_convertible_names(rain) == ["precip"]


def _monthly(times, calendar=None):
    values = np.full((len(times), 1, 1), 2.0)
    if calendar is None:
        time = np.array(times, dtype="datetime64[ns]")
    else:
        time = xr.date_range(times[0], periods=len(times), freq="MS", calendar=calendar)
    return xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), values, {"units": "mm day-1"})},
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )


def test_interval_days_uses_calendar_month_lengths():
    ds = _monthly(["2024-01-01", "2024-02-01", "2025-02-01"])
    assert interval_days(ds["time"], "1 month").tolist() == [31.0, 29.0, 28.0]
    assert interval_days(ds["time"], "3 month")[0] == 31 + 29 + 31
    assert interval_days(ds["time"], "7 day").tolist() == [7.0, 7.0, 7.0]
    noleap = _monthly(["2024-01-01", "2024-02-01"], calendar="noleap")
    assert interval_days(noleap["time"], "1 month").tolist() == [31.0, 28.0]


def test_interval_days_refuses_months_without_month_start_labels():
    ds = _monthly(["2024-01-15", "2024-02-15"])
    with pytest.raises(UsageError):
        interval_days(ds["time"], "1 month")
    with pytest.raises(UsageError):
        interval_days(xr.DataArray(0.0), "1 month")


def test_rate_to_total_multiplies_each_month_by_its_length():
    ds = quantify_dataset(_monthly(["2024-01-01", "2024-02-01"]))
    total = rate_to_total(ds["precip"], "1 month", dim="time").pint.dequantify()
    assert total.isel(latitude=0, longitude=0).values.tolist() == pytest.approx([62.0, 58.0])
    assert total.attrs["units"] in ("mm", "millimeter")


def test_assert_nonoverlapping_accepts_months_and_catches_one_overlapping_pair():
    assert_nonoverlapping_intervals(
        _monthly(["2025-01-01", "2025-02-01", "2025-03-01"]), "time", "1 month"
    )
    times = ["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22", "2026-01-23"]
    weekly = _monthly(times)
    with pytest.raises(UsageError):
        assert_nonoverlapping_intervals(weekly, "time", "7 day")


def test_classify_era5_total_precipitation():
    assert classify_variable("total_precipitation", units="m") == "precip_amount"
    assert classify_variable("total_precipitation", units="mm day-1") == "precip"


def test_to_standard_units_keeps_temperature_differences_as_differences():
    """A 2 K warm anomaly is 2 delta_degC, never 2 - 273.15."""
    ds = make_gridded(name="t2m", units="delta_degF")
    ds["t2m"].attrs["standard_name"] = "air_temperature"
    ds["t2m"].values[:] = 3.6
    out = to_standard_units(quantify_dataset(ds), variables=["t2m"]).pint.dequantify()
    assert float(out["t2m"].mean()) == pytest.approx(2.0)
    assert units_equal(out["t2m"].attrs["units"], "delta_degC")


def test_rate_to_total_converts_mass_flux_through_water_density():
    """kg m-2 s-1 x 1 day is 86400 kg m-2 = 86400 mm, not 1 labelled mm."""
    ds = xr.Dataset(
        {"precip": (("time",), [1.0 / 86400], {"units": "kg m-2 s-1"})},
        coords={"time": np.array(["2026-01-01"], dtype="datetime64[D]")},
    )
    total = rate_to_total(quantify_dataset(ds)["precip"], "1 day")
    assert str(total.pint.units) in ("millimeter", "mm")
    np.testing.assert_allclose(total.pint.dequantify().values, [1.0])


def test_convert_dataarray_keeps_float32():
    """kg m-2 s-1 -> mm day-1 goes through the water density; float32 stays float32."""
    da = make_gridded(n_time=1, fill=1e-4, units="kg m-2 s-1")["precip"].astype("float32")
    da.attrs["units"] = "kg m-2 s-1"
    for src in (da, da.pint.quantify()):
        out, density = convert_dataarray(src, "mm day-1")
        assert density
        assert out.dtype == np.float32
        np.testing.assert_allclose(
            out.pint.dequantify().values if out.pint.units else out.values, 8.64, rtol=1e-5
        )


def test_stamp_precip_amounts_leaves_non_precip_kg_m2_alone():
    """Total column water / water vapour in kg m-2 are not precipitation."""
    ds = make_gridded(n_time=1, fill=20.0, units="kg m-2", name="tcw")
    ds["tcw"].attrs.update(
        standard_name="atmosphere_mass_content_of_water_vapor", long_name="Total column water"
    )
    out = stamp_precip_amounts(ds)
    assert out["tcw"].attrs["standard_name"] == "atmosphere_mass_content_of_water_vapor"
    assert out["tcw"].attrs["long_name"] == "Total column water"
    # A precip rate turned into a total still gets the amount name.
    rate = make_gridded(n_time=1, fill=5.0, units="mm", name="precip")
    rate["precip"].attrs["standard_name"] = "lwe_precipitation_rate"
    stamped = stamp_precip_amounts(rate)["precip"].attrs["standard_name"]
    assert stamped == "lwe_thickness_of_precipitation_amount"

import numpy as np
import pytest
import xarray as xr
from conftest import make_gridded, make_point_obs

from weather_skills_core.errors import DataError, UsageError
from weather_skills_core.reduce import summarize
from weather_skills_core.standard_utils import latitude_weights
from weather_skills_core.units import dequantify_dataset, quantify_dataset, units_equal


def run(ds, **kwargs):
    """Summarize as a skill does: quantified in, dequantified out."""
    return dequantify_dataset(summarize(quantify_dataset(ds), **kwargs))


def test_mean_collapses_time():
    out = run(make_gridded(n_time=3, fill=2.0), dims=["time"], method="mean")
    assert "time" not in out.dims
    assert out["precip"].values == pytest.approx(2.0)


def test_sum_collapses_longitude():
    out = run(make_gridded(), dims=["longitude"], method="sum")
    assert "longitude" not in out.dims
    assert out["precip"].sizes["latitude"] == 3


def _celsius(lats=(0.0, 60.0), lons=(10.0, 11.0)):
    ds = make_gridded(name="t2m", lats=lats, lons=lons)
    # 10 °C at the equator row, 30 °C at 60°N (cos weights 1 and 0.5).
    ds["t2m"].values[:] = np.array([10.0, 30.0])[None, :, None]
    ds["t2m"].attrs.update(units="degree_Celsius", standard_name="air_temperature")
    return ds


def test_lat_weighted_mean_celsius():
    ds = _celsius()
    out = run(ds, dims=["latitude"], weights=latitude_weights(ds["latitude"]))
    assert "latitude" not in out.dims
    # (1 * 10 + 0.5 * 30) / 1.5 = 16.67 °C — not 290 K, and not 16.67 K.
    assert out["t2m"].values == pytest.approx(50.0 / 3.0)
    assert out["t2m"].attrs["units"] == "degree_Celsius"
    assert out["t2m"].attrs["standard_name"] == "air_temperature"


def test_lat_weighted_mean_over_lat_and_lon():
    ds = _celsius()
    out = run(ds, dims=["latitude", "longitude"], weights=latitude_weights(ds["latitude"]))
    assert out["t2m"].dims == ("time",)
    assert out["t2m"].values == pytest.approx(50.0 / 3.0)


def test_lat_weighted_mean_precip_unchanged():
    ds = make_gridded(fill=2.0)
    out = run(ds, dims=["latitude"], weights=latitude_weights(ds["latitude"]))
    assert out["precip"].values == pytest.approx(2.0)
    assert units_equal(out["precip"].attrs["units"], "mm day-1")


def test_keeps_existing_cell_methods():
    ds = make_gridded()
    ds["precip"].attrs.update(
        units="mm", standard_name="lwe_thickness_of_precipitation_amount", cell_methods="time: sum"
    )
    out = run(ds, dims=["time"], method="mean")
    assert "time" not in out.dims
    assert out["precip"].attrs["cell_methods"] == "time: sum time: mean"


def _labeled(n_time=4):
    ds = make_gridded(n_time=n_time, lats=(1.0, 2.0), lons=(10.0, 11.0, 12.0))
    ds["precip"].values[:] = np.arange(n_time * 6, dtype=float).reshape(n_time, 2, 3)
    regions = np.array([["a", "a", ""], ["b", "b", "b"]])
    return ds.assign_coords(
        region=(("latitude", "longitude"), regions),
        region_name=(("latitude", "longitude"), np.char.upper(regions)),
        half=("time", ["x", "x", "y", "y"][:n_time]),
    )


def test_by_label_collapses_its_dims():
    out = run(_labeled(), by=["region"], method="mean")
    assert out["precip"].dims == ("time", "region")
    assert out["region"].values.tolist() == ["a", "b"]  # "" dropped
    # time 0: a = mean(0, 1), b = mean(3, 4, 5)
    assert out["precip"].isel(time=0).values.tolist() == [0.5, 4.0]
    assert "latitude" not in out.dims and "region_name" not in out.coords
    assert "(grouped by region)" in out["precip"].attrs["cell_methods"]


def test_by_with_weights_is_a_weighted_mean_per_group():
    ds = _labeled()
    ds = ds.assign_coords(latitude=[0.0, 60.0])
    out = run(ds, by=["region"], weights=latitude_weights(ds["latitude"]))
    # time 0: a is one row (0, 1) -> 0.5; b is the 60°N row only (3, 4, 5) -> 4.
    assert out["precip"].isel(time=0).values.tolist() == pytest.approx([0.5, 4.0])
    ds = ds.assign_coords(region=(("latitude", "longitude"), np.full((2, 3), "r")))
    out = run(ds, by=["region"], weights=latitude_weights(ds["latitude"]))
    # Rows 0..2 (mean 1) at weight 1 and rows 3..5 (mean 4) at weight 0.5.
    assert out["precip"].isel(time=0).item() == pytest.approx((1 * 1 + 0.5 * 4) / 1.5)
    assert out["precip"].attrs["units"] == "mm day-1"


def test_weighted_mean_skips_nan_and_its_weight():
    ds = make_gridded(n_time=1, lats=(0.0, 60.0), lons=(10.0,))
    ds["precip"].values[0, :, 0] = [np.nan, 4.0]
    ds = ds.assign_coords(region=(("latitude", "longitude"), [["r"], ["r"]]))
    weights = latitude_weights(ds["latitude"])
    plain = run(ds, dims=["latitude", "longitude"], weights=weights)
    grouped = run(ds, by=["region"], weights=weights)
    assert plain["precip"].item() == pytest.approx(4.0)
    assert grouped["precip"].item() == pytest.approx(4.0)


def test_weights_need_mean():
    ds = make_gridded()
    with pytest.raises(UsageError, match="weights need method mean"):
        summarize(quantify_dataset(ds), dims=["time"], method="sum", weights=ds["latitude"])


def test_by_two_labels_with_extra_dim():
    ds = _labeled().drop_vars("half")
    ds = ds.assign_coords(dayofyear=("time", ds["time"].dt.dayofyear.values))
    ds["precip"].attrs.update(units="mm", cell_methods="time: sum")
    out = run(ds, by=["region", "dayofyear"], method="sum")
    assert set(out["precip"].dims) == {"region", "dayofyear"}
    # region a, day 2: cells 0, 1 at time 1 -> 6 + 7
    assert out["precip"].sel(region="a", dayofyear=2).item() == 13.0


def test_by_rejects_dim_coord():
    with pytest.raises(UsageError, match="not a label coord"):
        run(_labeled(), by=["time"])


def test_by_all_empty_labels_is_an_error():
    ds = _labeled().assign_coords(region=(("latitude", "longitude"), np.full((2, 3), "")))
    with pytest.raises(DataError, match="empty label"):
        run(ds, by=["region"])


def test_by_labels_sharing_some_dims_is_an_error():
    ds = _labeled().assign_coords(band=("latitude", ["s", "n"]))
    with pytest.raises(UsageError, match="share some but not all dims"):
        run(ds, by=["region", "band"])


def test_by_same_dims_combination_without_cells_is_nan():
    """Two labels on the same dims: a pair that never co-occurs is NaN, not dropped."""
    zone = np.array([["w", "e", "e"], ["w", "w", "e"]])
    ds = _labeled().assign_coords(zone=(("latitude", "longitude"), zone))
    r = run(ds, by=["region", "zone"], method="max")["precip"].isel(time=0)
    # region a = cells (0,0) w, (0,1) e; region b = (1,0) w, (1,1) w, (1,2) e
    assert r.sel(region="a", zone="w").item() == 0.0
    assert r.sel(region="a", zone="e").item() == 1.0
    assert r.sel(region="b", zone="e").item() == 5.0
    assert r.sel(region="b", zone="w").item() == 4.0


def _seasons(n_time=275, start="2020-04-01"):
    """Two stations of daily totals with year and MAM / OND labels; S1 misses 10 OND days."""
    ds = make_point_obs(n_time=n_time, start=start, fill=1.0)
    ds["precip"].attrs.update(
        units="mm", standard_name="lwe_thickness_of_precipitation_amount", cell_methods="time: sum"
    )
    month = ds["time"].dt.month.values
    ds["precip"].values[1, np.nonzero(month == 11)[0][:10]] = np.nan
    group = np.where(np.isin(month, [3, 4, 5]), "MAM", np.where(month >= 10, "OND", ""))
    ds["elevation"] = ("point_id", [10.0, 20.0])
    return ds.assign_coords(year=("time", ds["time"].dt.year.values), month_group=("time", group))


def test_by_season_blanks_incomplete_groups(capsys):
    """April-May is a partial MAM (61 of 92 days) and S1 misses 10 OND days: both blanked."""
    out = run(_seasons(), by=["year", "month_group"], method="sum")
    precip = out["precip"].sel(year=2020)
    assert precip.dims == ("point_id", "month_group")
    assert precip.sel(point_id="S0", month_group="OND").item() == 92.0
    assert np.isnan(precip.sel(month_group="MAM")).all()
    assert np.isnan(precip.sel(point_id="S1", month_group="OND").item())
    assert "left out 3 cell-group(s)" in capsys.readouterr().err
    assert "precip_coverage" not in out
    # A variable without the label dims passes through.
    assert out["elevation"].values.tolist() == [10.0, 20.0]


def test_coverage_is_per_station_before_a_regional_sum(capsys):
    """Two stations missing different days are never pooled into a complete OND."""
    ds = _seasons()
    month = ds["time"].dt.month.values
    ds["precip"].values[0, np.nonzero(month == 12)[0][:5]] = np.nan  # S0 misses Dec days
    out = run(ds, dims=["point_id"], by=["year", "month_group"], method="sum")
    assert np.isnan(out["precip"].sel(year=2020, month_group="OND").item())
    assert "left out" in capsys.readouterr().err


def test_min_coverage_keeps_partial_sums_with_warning(capsys):
    out = run(_seasons(), by=["year", "month_group"], method="sum", min_coverage=0.5)
    precip = out["precip"].sel(year=2020)
    assert precip.sel(point_id="S0", month_group="MAM").item() == 61.0  # not scaled up
    assert precip.sel(point_id="S1", month_group="OND").item() == 82.0
    assert "estimates from partial data" in capsys.readouterr().err
    coverage = out["precip_coverage"].sel(year=2020)
    assert coverage.sel(point_id="S1", month_group="MAM").item() == pytest.approx(61 / 92)
    assert coverage.sel(point_id="S1", month_group="OND").item() == pytest.approx(82 / 92)
    assert coverage.sel(point_id="S0", month_group="OND").item() == 1.0
    assert out["precip"].attrs["ancillary_variables"] == "precip_coverage"


def test_season_crossing_december_takes_its_start_year():
    """Dec 2020 - Feb 2021 is one complete DJF (90 days) labelled 2020."""
    ds = make_gridded(n_time=90, start="2020-12-01", lats=(1.0,), lons=(10.0,))
    ds["precip"].attrs.update(units="mm", cell_methods="time: sum")
    month = ds["time"].dt.month.values
    ds = ds.assign_coords(
        year=("time", ds["time"].dt.year.values - (month < 12)),
        month_group=("time", np.full(90, "DJF")),
    )
    out = run(ds, by=["year", "month_group"], method="sum")
    assert out["precip"].sel(year=2020, month_group="DJF").values.ravel().tolist() == [90.0]


def test_sum_of_rates_over_time_is_refused():
    with pytest.raises(UsageError, match="convert the rates to period totals first"):
        run(make_gridded(), dims=["time"], method="sum")


def test_unknown_dim_and_variable_are_refused():
    with pytest.raises(UsageError, match="dim 'level' not in dataset"):
        run(make_gridded(), dims=["level"])
    with pytest.raises(UsageError, match="variable 't2m' not in dataset"):
        run(make_gridded(), dims=["time"], variables=["t2m"])


def test_result_is_quantified():
    out = summarize(quantify_dataset(make_gridded()), dims=["time"])
    assert isinstance(out, xr.Dataset)
    assert out["precip"].pint.units is not None

"""Opt-in dask mode: reserved flags, lazy inputs, uniform sharded writes, atomic output."""

import json
import os
import sys

import numpy as np
import pytest
import xarray as xr
import zarr
from conftest import make_forecast, make_gridded

from weather_skills_core import (
    Dataset,
    UsageError,
    weather_skill,
    zarr_layout,
)
from weather_skills_core.provenance import HISTORY_ATTR, load_history, stamp_zarr
from weather_skills_core.units import convert_dataarray, deaccumulate_along_step

# A chunk size below the stored chunks makes xarray warn; that is what TINY is for.
pytestmark = pytest.mark.filterwarnings("ignore:The specified chunks separate")

# Small enough that a 40-step test grid splits into several chunks.
TINY = ["--dask", "--dask-chunk-size", "1KiB"]


def _write(path, ds=None, **kwargs):
    ds = make_gridded(**kwargs) if ds is None else ds
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def _is_dask(ds):
    return any(v.chunks is not None for v in ds.pint.dequantify().variables.values())


def _copy_skill(seen=None, *, transform=lambda ds: ds):
    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def copy(ds, output, dask, dask_workers, dask_chunk_size, exit_stack, **kwargs):
        if seen is not None:
            seen.update(
                dask=dask, workers=dask_workers, chunk_size=dask_chunk_size, lazy=_is_dask(ds)
            )
        return transform(ds)

    return copy


def test_dask_flags_arrive_as_plain_kwargs(tmp_path):
    src = _write(tmp_path / "in.zarr")
    seen = {}
    copy = _copy_skill(seen)
    copy(["-i", str(src), "-o", str(tmp_path / "a.zarr")])
    assert (seen["dask"], seen["workers"], seen["chunk_size"]) == (
        False,
        min(4, os.cpu_count() or 1),
        "64MiB",
    )
    copy(
        ["-i", str(src), "-o", str(tmp_path / "b.zarr")]
        + ["--dask", "--dask-workers", "2", "--dask-chunk-size", "10MiB"]
    )
    assert (seen["dask"], seen["workers"], seen["chunk_size"]) == (True, 2, "10MiB")


def test_dask_value_flags_are_unused_without_dask(tmp_path):
    src = _write(tmp_path / "in.zarr")
    seen = {}
    _copy_skill(seen)(
        ["-i", str(src), "-o", str(tmp_path / "out.zarr"), "--dask-chunk-size", "lots"]
    )
    assert seen["dask"] is False and seen["chunk_size"] == "lots"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--dask", "--dask-workers", "0"], "at least 1"),
        (["--dask", "--dask-chunk-size", "lots"], "is not a size"),
        (["--dask", "--dask-chunk-size", "0"], "is not a size"),
    ],
)
def test_dask_flag_usage_errors(tmp_path, capsys, extra, message):
    ran = []
    src = _write(tmp_path / "in.zarr")

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def skill(ds, output, **kwargs):
        ran.append(True)
        return ds

    with pytest.raises(SystemExit) as exc:
        skill(["-i", str(src), "-o", str(tmp_path / "out.zarr")] + extra)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not ran


def test_dask_flag_without_dask_installed_is_a_usage_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "dask", None)
    src = _write(tmp_path / "in.zarr")
    with pytest.raises(SystemExit) as exc:
        _copy_skill()(["-i", str(src), "-o", str(tmp_path / "out.zarr"), "--dask"])
    assert exc.value.code == 2
    assert "weather-skills-core[dask]" in capsys.readouterr().err
    assert not (tmp_path / "out.zarr").exists()


def test_dask_flags_are_reserved_for_the_decorator():
    for flag in ("--dask", "--dask-workers", "--dask-chunk-size"):
        with pytest.raises(ValueError, match="decorator owns them"):

            @weather_skill(name="s", version="0.1.0", output=False)
            @weather_skill.argument(flag)
            def skill(**kwargs):
                pass


def test_dask_flags_are_on_every_skill_and_in_help(capsys):
    @weather_skill(name="s", version="0.1.0", output=False)
    def skill(**kwargs):
        pass

    with pytest.raises(SystemExit):
        skill(["--help"])
    out = capsys.readouterr().out
    assert "--dask-workers N" in out and "--dask-chunk-size SIZE" in out


def test_inputs_are_numpy_backed_without_the_flag_even_with_dask_installed(tmp_path):
    src = _write(tmp_path / "in.zarr")
    seen = {}
    _copy_skill(seen)(["-i", str(src), "-o", str(tmp_path / "out.zarr")])
    assert seen["lazy"] is False


def test_zipped_input_is_numpy_backed_off_and_dask_backed_on(tmp_path):
    from test_decorator import _zip_dir

    src = _write(tmp_path / "sample.zarr")
    zipped = _zip_dir(src, tmp_path / "sample.zarr.zip", None)
    seen = {}
    copy = _copy_skill(seen)
    copy(["-i", str(zipped), "-o", str(tmp_path / "a.zarr")])
    assert seen["lazy"] is False
    copy(["-i", str(zipped), "-o", str(tmp_path / "b.zarr"), "--dask"])
    assert seen["lazy"] is True


def test_load_history_opens_without_a_dask_graph(tmp_path, monkeypatch):
    src = tmp_path / "in.zarr"
    ds = make_gridded()
    stamp_zarr(ds, [{"skill": "x", "version": "1", "args": {}, "input": None}])
    _write(src, ds)
    calls = []
    open_zarr = xr.open_zarr
    monkeypatch.setattr(xr, "open_zarr", lambda *a, **kw: calls.append(kw) or open_zarr(*a, **kw))
    assert load_history(src)[0]["skill"] == "x"
    assert calls[0]["chunks"] is None


def test_dask_run_reads_chunked_and_writes_a_sharded_uniform_zarr(tmp_path):
    src = _write(tmp_path / "in.zarr", n_time=40, lats=(1.0, 2.0, 3.0), lons=(10.0, 11.0, 12.0))
    out = tmp_path / "out.zarr"
    seen = {}
    _copy_skill(seen)(["-i", str(src), "-o", str(out)] + TINY)
    assert seen["lazy"] is True

    arr = zarr.open_group(out, mode="r")["precip"]
    assert arr.shards is not None and arr.metadata.zarr_format == 3
    assert arr.shards[0] < 40  # several shards, not one
    inner = arr.chunks
    assert all(s % c == 0 for s, c in zip(arr.shards, inner, strict=True))
    back = xr.open_zarr(out, chunks=None, consolidated=True)
    xr.testing.assert_identical(
        back.drop_vars(HISTORY_ATTR, errors="ignore").load(),
        xr.open_zarr(src, chunks=None).load().assign_attrs(back.attrs),
    )


def test_dask_data_is_still_lazy_when_it_reaches_to_zarr(tmp_path, monkeypatch):
    """No unit, attr or coord normalization on the way out may read the data."""
    src = _write(tmp_path / "in.zarr", n_time=40)
    reached = []
    to_zarr = xr.Dataset.to_zarr

    def spy(self, *args, **kwargs):
        reached.append(all(v.chunks is not None for v in self.data_vars.values()))
        return to_zarr(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_zarr", spy)
    _copy_skill()(["-i", str(src), "-o", str(tmp_path / "out.zarr"), "--dask"])
    assert reached == [True]


def test_run_without_the_flag_writes_as_before(tmp_path, monkeypatch):
    """No sharding, no rechunking, and no dask needed."""
    monkeypatch.setitem(sys.modules, "dask", None)
    src = _write(tmp_path / "in.zarr", n_time=40)
    out = tmp_path / "out.zarr"
    seen = {}
    _copy_skill(seen)(["-i", str(src), "-o", str(out)])
    assert seen["lazy"] is False
    arr = zarr.open_group(out, mode="r")["precip"]
    assert arr.shards is None and arr.chunks == arr.shape
    xr.testing.assert_equal(xr.open_zarr(out, chunks=None).load(), make_gridded(n_time=40))


def _ragged(ds):
    lazy = ds.pint.dequantify().chunk(time=3)
    return xr.concat(
        [lazy.isel(time=slice(0, 3)), lazy.isel(time=slice(5, 7)), lazy.isel(time=slice(10, 40))],
        "time",
    )


def test_ragged_dask_chunks_from_a_skill_write_fine(tmp_path):
    src = _write(tmp_path / "in.zarr", n_time=40)
    out = tmp_path / "out.zarr"
    # Plain to_zarr refuses these chunks; the decorator must not.
    with pytest.raises(ValueError, match="uniform"):
        _ragged(make_gridded(n_time=40)).to_zarr(tmp_path / "plain.zarr")

    _copy_skill(transform=_ragged)(["-i", str(src), "-o", str(out)] + TINY)
    expected = xr.open_zarr(src, chunks=None).isel(time=[0, 1, 2, 5, 6, *range(10, 40)])
    got = xr.open_zarr(out, chunks=None)
    np.testing.assert_array_equal(got.precip.values, expected.precip.values)
    np.testing.assert_array_equal(got.time.values, expected.time.values)
    assert zarr.open_group(out, mode="r")["precip"].shards is not None


class _Boom(Exception):
    pass


def _explode(block):
    raise _Boom


def _failing(ds):
    """A lazy result whose computation raises, so the write fails part-way."""
    ds = ds.pint.dequantify().chunk(time=1)
    return ds.assign(precip=(ds.precip.dims, ds.precip.data.map_blocks(_explode, dtype=float)))


@pytest.mark.parametrize("flags", [[], ["--dask"]])
def test_failed_write_leaves_no_partial_output_and_keeps_the_old_one(tmp_path, flags):
    src = _write(tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    _copy_skill()(["-i", str(src), "-o", str(out)])
    before = load_history(out)

    with pytest.raises(_Boom):
        _copy_skill(transform=_failing)(["-i", str(src), "-o", str(out)] + flags)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.zarr", "out.zarr"]
    assert load_history(out) == before
    assert xr.open_zarr(out, chunks=None).sizes["time"] == 2


def test_failed_first_write_leaves_nothing(tmp_path):
    src = _write(tmp_path / "in.zarr")
    with pytest.raises(_Boom):
        _copy_skill(transform=_failing)(
            ["-i", str(src), "-o", str(tmp_path / "out.zarr"), "--dask"]
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.zarr"]


@pytest.mark.parametrize("flags", [[], TINY])
def test_input_and_output_may_be_the_same_path(tmp_path, flags):
    path = tmp_path / "data.zarr"
    ds = make_gridded(n_time=40)
    stamp_zarr(ds, [{"skill": "seed", "version": "1", "args": {}, "input": None}])
    ds.to_zarr(path, mode="w", consolidated=True)
    expected = xr.open_zarr(path, chunks=None).precip.values * 2

    def double(ds):
        # With the flag this is lazy: the read happens while writing over the input.
        return ds.pint.dequantify().assign(precip=lambda d: d.precip * 2)

    _copy_skill(transform=double)(["-i", str(path), "-o", str(path)] + flags)
    np.testing.assert_array_equal(xr.open_zarr(path, chunks=None).precip.values, expected)
    assert [e["skill"] for e in load_history(path)] == ["seed", "copy"]


@pytest.mark.parametrize("flags", [[], ["--dask"]])
def test_exit_stack_closes_after_the_outputs_are_written(tmp_path, flags):
    src = _write(tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    events = []

    @weather_skill(name="s", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
    def skill(ds, output, exit_stack, **kwargs):
        exit_stack.callback(lambda: events.append(("closed", output.exists())))
        events.append(("ran", output.exists()))
        return ds

    skill(["-i", str(src), "-o", str(out)] + flags)
    assert events == [("ran", False), ("closed", True)]


def test_exit_stack_closes_when_the_skill_or_write_fails(tmp_path):
    src = _write(tmp_path / "in.zarr")
    events = []

    def run(body):
        @weather_skill(name="s", version="0.1.0")
        @weather_skill.argument("-i", "--input", type=Dataset("observations"), required=True)
        def skill(ds, output, exit_stack, **kwargs):
            exit_stack.callback(events.append, "closed")
            return body(ds)

        return skill

    def usage(ds):
        raise UsageError("nope")

    with pytest.raises(SystemExit):
        run(usage)(["-i", str(src), "-o", str(tmp_path / "a.zarr")])
    with pytest.raises(_Boom):
        run(_failing)(["-i", str(src), "-o", str(tmp_path / "b.zarr")])
    assert events == ["closed", "closed"]


def test_exit_stack_can_hold_a_temp_dir_lazy_data_reads_from(tmp_path):
    out = tmp_path / "out.zarr"

    @weather_skill(name="s", version="0.1.0")
    def skill(output, exit_stack, dask, **kwargs):
        import tempfile

        scratch = exit_stack.enter_context(tempfile.TemporaryDirectory())
        _write(f"{scratch}/t.zarr")
        # Lazy data that outlives the function: only the write may read it.
        skill.scratch = scratch
        return xr.open_zarr(f"{scratch}/t.zarr", chunks="auto" if dask else None)

    skill(["-o", str(out), "--dask"])
    assert xr.open_zarr(out, chunks=None).sizes["time"] == 2
    assert not os.path.exists(skill.scratch)


def test_dask_flags_are_not_in_provenance(tmp_path):
    src = _write(tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    _copy_skill()(["-i", str(src), "-o", str(out), "--dask", "--dask-workers", "2"])
    entry = load_history(out)[-1]
    assert entry["args"] == {}
    assert "dask" not in json.dumps(entry["args"])


def test_station_dataset_round_trips_with_the_flag(tmp_path):
    from conftest import make_station

    station = make_station().assign_attrs()
    station["precip"].attrs["units"] = "mm day-1"
    src = _write(tmp_path / "in.zarr", station)
    out = tmp_path / "out.zarr"

    @weather_skill(name="copy", version="0.1.0")
    @weather_skill.argument("-i", "--input", type=Dataset("any"), required=True)
    def copy(ds, output, **kwargs):
        return ds

    copy(["-i", str(src), "-o", str(out), "--dask"])
    xr.testing.assert_equal(xr.open_zarr(out, chunks=None).load(), station)


@pytest.mark.parametrize(
    ("shard", "itemsize"),
    [((1, 721, 1440), 4), ((100, 60, 7), 8), ((1441,), 4), ((3, 4), 8), ((997, 997), 8)],
)
def test_inner_chunks_divide_the_shard_and_fit_the_budget(shard, itemsize):
    inner = zarr_layout.inner_chunks(shard, itemsize)
    assert all(s % c == 0 for s, c in zip(shard, inner, strict=True))
    elements = int(np.prod(inner))
    assert elements * itemsize <= zarr_layout.INNER_CHUNK_BYTES or elements == 1


def test_inner_chunks_are_balanced():
    assert zarr_layout.inner_chunks((1, 721, 1440), 4) == (1, 721, 360)
    assert zarr_layout.inner_chunks((1024, 1024), 4) == (512, 512)


def _tp_forecast(n_step=6, dtype="f8", units="kg m-2"):
    ds = make_forecast(n_number=2, n_step=n_step)
    rng = np.random.default_rng(0)
    steps = rng.random((2, n_step, 2, 2)).cumsum(axis=1) * 5
    steps[:, 3] = steps[:, 2] - 1  # a reset, clipped to zero
    ds["tp"] = (ds["tp"].dims, steps.astype(dtype), {"units": units})
    return ds


@pytest.mark.parametrize("dtype", ["f8", "f4"])
@pytest.mark.parametrize("units", ["kg m-2", "mm", "m"])
def test_deaccumulate_matches_on_numpy_and_dask_and_stays_lazy(dtype, units):
    ds = _tp_forecast(dtype=dtype, units=units)
    eager = deaccumulate_along_step(ds)
    lazy = deaccumulate_along_step(ds.chunk(number=1, step=2))
    assert lazy.tp.chunks is not None
    xr.testing.assert_identical(lazy.compute(), eager)
    assert eager.tp.attrs["units"] == "mm day-1"
    assert (eager.tp.values >= 0).all()


def test_deaccumulate_keeps_nan_and_non_precip_vars_lazy():
    ds = _tp_forecast()
    ds["tp"].values[0, 1, 0, 0] = np.nan
    ds["t"] = ds["tp"].copy(data=ds["tp"].values * 0.5).assign_attrs(units="K")
    names = ["t"]
    lazy = deaccumulate_along_step(ds.chunk(number=1), names=names)
    assert lazy.t.chunks is not None
    xr.testing.assert_identical(lazy.compute(), deaccumulate_along_step(ds, names=names))


@pytest.mark.parametrize("dtype", ["f8", "f4"])
@pytest.mark.parametrize(
    ("units", "target"), [("kg m-2 s-1", "mm day-1"), ("kg m-2", "mm"), ("mm", "m")]
)
@pytest.mark.parametrize("quantified", [False, True])
def test_convert_dataarray_matches_on_numpy_and_dask_and_stays_lazy(
    dtype, units, target, quantified
):
    da = xr.DataArray(
        np.arange(12, dtype=dtype).reshape(3, 4), dims=("y", "x"), attrs={"units": units}
    )
    lazy_in = da.chunk(y=1)
    if quantified:
        da, lazy_in = da.pint.quantify(), lazy_in.pint.quantify()
    eager, flag = convert_dataarray(da, target)
    lazy, lazy_flag = convert_dataarray(lazy_in, target)
    assert lazy_flag == flag
    assert lazy.pint.dequantify().chunks is not None if quantified else lazy.chunks is not None
    xr.testing.assert_identical(
        lazy.pint.dequantify().compute() if quantified else lazy.compute(),
        eager.pint.dequantify() if quantified else eager,
    )


def _stand_in(module, name, *bases):
    """An exception class that looks, by qualified name, like one from a network library."""
    return type(name, bases or (Exception,), {"__module__": module})


ClientError = _stand_in("aiohttp.client_exceptions", "ClientError")
ClientResponseError = _stand_in("aiohttp.client_exceptions", "ClientResponseError", ClientError)
HttpError = _stand_in("gcsfs.retry", "HttpError")
RequestException = _stand_in("requests.exceptions", "RequestException", OSError)
HTTPError = _stand_in("requests.exceptions", "HTTPError", RequestException)


def _failing_with(exc):
    def run(ds):
        ds = ds.pint.dequantify().chunk(time=1)

        def raise_it(block):
            raise exc

        return ds.assign(precip=(ds.precip.dims, ds.precip.data.map_blocks(raise_it, dtype=float)))

    return run


@pytest.mark.parametrize("flags", [[], ["--dask"]])
@pytest.mark.parametrize(
    ("exc", "hint"),
    [
        (ClientResponseError("500, message='Internal Server Error'"), False),
        (ClientResponseError("403, message='Forbidden'"), True),
        (HttpError("Forbidden: 403"), True),
        (HTTPError("401 Client Error: Unauthorized"), True),
        (RequestException("connection reset"), False),
    ],
)
def test_network_read_failures_while_writing_become_data_errors(tmp_path, capsys, flags, exc, hint):
    src = _write(tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    with pytest.raises(SystemExit) as raised:
        _copy_skill(transform=_failing_with(exc))(["-i", str(src), "-o", str(out)] + flags)
    err = capsys.readouterr().err
    assert raised.value.code == 1
    assert f"reading the input data failed while writing {out}: {type(exc).__name__}: " in err
    assert ("check your credentials" in err) == hint
    assert raised.value.__context__.__cause__ is exc
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.zarr"]


def test_other_write_failures_propagate_unchanged(tmp_path):
    src = _write(tmp_path / "in.zarr")
    with pytest.raises(KeyError):
        _copy_skill(transform=_failing_with(KeyError("x")))(
            ["-i", str(src), "-o", str(tmp_path / "out.zarr"), "--dask"]
        )

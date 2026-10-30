"""``@weather_skill`` decorator: CLI, open Dataset inputs, stamp/write outputs."""

import argparse
import contextlib
import functools
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path

from weather_skills_core import provenance as provenance_mod
from weather_skills_core import standard_args, zarr_layout
from weather_skills_core import standard_dataset as std
from weather_skills_core.dataset_type import Dataset
from weather_skills_core.errors import DataError, SkillError, UsageError

# xarray, numpy, and pint load inside the functions that use them, so importing
# a skill (and ``--help`` / usage errors) stays fast.

# Accumulated by ``@weather_skill.argument`` (bottom-up); weather_skill reads it.
ARGS_ATTR = "__weather_skill_arguments__"

# The decorator opens each Dataset argument and hands the skill an xr.Dataset, so
# the skill no longer sees the path. This attr records it (ds.attrs[INPUT_PATH_ATTR]
# = the path as given on the command line) so a skill can match a dataset back to a
# file its own arguments name. In-memory only: stripped before any output is written.
INPUT_PATH_ATTR = "weather_skills_input_path"


def argv_has_option(argv: list[str], option_strings) -> bool:
    """True when one of ``option_strings`` is present (bare, space, or ``--flag=``)."""
    for flag in option_strings:
        for arg in argv:
            if arg == flag or (flag.startswith("--") and arg.startswith(f"{flag}=")):
                return True
    return False


class Argument:
    """One stacked CLI flag (same kwargs as ``argparse.add_argument``).

    ``probe=True`` is reserved: when that flag is on argv, the decorator
    drops every ``required=True`` flag (including ``-o``) and skips writing.
    """

    def __init__(self, *option_strings, **kwargs):
        if not option_strings:
            raise ValueError("Argument requires at least one option string or positional name")
        kwargs = dict(kwargs)
        self.probe = bool(kwargs.pop("probe", False))
        self.option_strings = option_strings
        self.kwargs = kwargs
        self.dataset_type: Dataset | None = None
        type_kw = self.kwargs.get("type")
        if isinstance(type_kw, Dataset):
            self.dataset_type = type_kw

    @property
    def dest(self) -> str:
        if "dest" in self.kwargs:
            return self.kwargs["dest"]
        flag = next(
            (f for f in self.option_strings if f.startswith("--")),
            self.option_strings[0],
        )
        return flag.lstrip("-").replace("-", "_")


def _history_json_default(obj):
    """Record objects with ``to_dict()`` (e.g. a parsed spec) as dicts, not their ``str``."""
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return str(obj)


def build_history(name, version, args, params, input_paths, upstream, strip_dests, revision=None):
    """Record this skill as a node on the provenance DAG.

    A fetcher (no inputs) or a single-input transform stays a path: the
    parent's history plus this entry. A multi-input skill is a join: the
    top-level array is only this entry, and every parent's full subgraph is
    nested on that entry's ``input`` list. ``revision`` is the git identity
    of the skill that ran (commit / repo / dirty).
    """
    entry_args = {k: v for k, v in vars(args).items() if k not in strip_dests}
    for dest in ("date", "start_time", "end_time"):
        value = params.get(dest)
        if isinstance(value, list):
            entry_args[dest] = [d.isoformat() for d in value]
        elif value is not None:
            entry_args[dest] = value.isoformat()
    entry_args = json.loads(json.dumps(entry_args, default=_history_json_default))

    if not input_paths:
        input_field, base_history = None, []
    elif len(input_paths) == 1:
        input_field = provenance_mod.input_ref(input_paths[0])
        base_history = list(upstream[0])
    else:
        input_field = [
            provenance_mod.input_ref(path, hist)
            for path, hist in zip(input_paths, upstream, strict=True)
        ]
        # Join node: do not flatten the first parent into a linear spine.
        base_history = []

    return base_history + [
        provenance_mod.build_entry(name, version, entry_args, input_field, revision=revision)
    ]


def prepare_dataset_output(ds, *, first_ds=None):
    """Normalize coords/units metadata every skill writes.

    - GRIB-style ``kg m**-2`` → pint/CF strings
    - precip amount units → amount CF ``standard_name``
    - ``step`` timedelta → ``timedelta64[ns]``
    - lat/lon coords → 5 decimal places, ``float32``
    - fill attrs stripped by geometry ops from the first input (same var names)
    - drop CF reference attrs naming variables the output lacks
    """
    from weather_skills_core.cf import drop_dangling_cf_refs
    from weather_skills_core.standard_utils import (
        fill_missing_data_var_attrs,
        normalize_latlon_coords,
        normalize_step_coord,
    )
    from weather_skills_core.units import (
        dequantify_dataset,
        normalize_unit_strings,
        stamp_precip_amounts,
    )

    if first_ds is not None:
        ref = first_ds
        if any(getattr(first_ds[v].pint, "units", None) is not None for v in first_ds.data_vars):
            ref = dequantify_dataset(first_ds)
        ds = fill_missing_data_var_attrs(ref, ds)
    ds = normalize_unit_strings(ds)
    ds = stamp_precip_amounts(ds)
    ds = normalize_step_coord(ds)
    ds = drop_dangling_cf_refs(ds)
    return normalize_latlon_coords(ds)


# Network-library errors a lazy source read raises while the output is written, matched
# by qualified name so none of those libraries has to be imported.
SOURCE_READ_ERRORS = frozenset(
    {
        "aiohttp.client_exceptions.ClientError",  # fsspec HTTP, gcsfs connection errors
        "gcsfs.retry.HttpError",
        "requests.exceptions.RequestException",
    }
)


def source_read_error(exc, out_path):
    """A ``DataError`` for a network read failure ``exc``, or None for any other exception."""
    if not SOURCE_READ_ERRORS & {f"{c.__module__}.{c.__qualname__}" for c in type(exc).__mro__}:
        return None
    message = f"reading the input data failed while writing {out_path}: {type(exc).__name__}: {exc}"
    if re.search(r"\b(401|403)\b|Forbidden", str(exc)):
        message += " (check your credentials and access to the source)"
    return DataError(message)


def _enter_dask(args, exit_stack):
    """Check the ``--dask`` flags and run under dask; returns the chunk size in bytes.

    Raises UsageError if dask is not installed or ``--dask-chunk-size`` is not a size.
    """
    try:
        import dask
        from dask.utils import parse_bytes
    except ImportError:
        raise UsageError(
            "--dask needs dask, which this skill does not have; the skill must list "
            "`weather-skills-core[dask]` in its script header dependencies"
        ) from None
    if args.dask_workers < 1:
        raise UsageError(f"--dask-workers must be at least 1; got {args.dask_workers}")
    try:
        chunk_bytes = parse_bytes(args.dask_chunk_size)
        if chunk_bytes < 1:
            raise ValueError(args.dask_chunk_size)
    except ValueError:
        raise UsageError(
            f"--dask-chunk-size {args.dask_chunk_size!r} is not a size like 64MiB"
        ) from None
    # Activates the threaded scheduler and chunk size for the whole run (open, skill,
    # write); closing the stack restores the global dask config.
    exit_stack.enter_context(
        dask.config.set(
            scheduler="threads",
            num_workers=args.dask_workers,
            **{"array.chunk-size": args.dask_chunk_size},
        )
    )
    return chunk_bytes


def write_output(value, out_path, history, first_ds, chunk_bytes=None):
    """Write one skill result: stamp a returned Path, or ``to_zarr`` a Dataset.

    A Dataset is written to a sibling path and renamed into place, so a failure
    leaves no partial output and the input may be ``out_path`` itself. With
    ``chunk_bytes`` (``--dask``), data is rechunked to uniform chunks of about that
    size and written as a sharded Zarr.
    """
    if isinstance(value, (str, Path)):
        written = Path(value)
        if written.resolve() != out_path.resolve():
            raise SkillError(f"returned path {written} does not match --output {out_path}")
        if written.is_dir():
            provenance_mod.restamp_zarr(written, history)
        else:
            provenance_mod.stamp_figure(written, history)
        print(f"Wrote: {out_path}", file=sys.stderr)
        return

    if not hasattr(value, "dims"):
        raise SkillError(f"skill returned {type(value).__name__}; expected xr.Dataset or Path")

    if hasattr(value, "pint"):
        from weather_skills_core.units import dequantify_dataset

        value = dequantify_dataset(value)
    value = prepare_dataset_output(value, first_ds=first_ds)
    if first_ds is not None:
        value.attrs = {**first_ds.attrs, **value.attrs}
    value.attrs.pop(INPUT_PATH_ATTR, None)
    provenance_mod.stamp_zarr(value, history)
    encoding = {}
    if chunk_bytes:
        value, encoding = zarr_layout.prepare_chunks(value, chunk_bytes)
    partial = out_path.with_name(f".{out_path.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    try:
        try:
            value.to_zarr(partial, mode="w", consolidated=True, encoding=encoding)
        except Exception as exc:
            error = source_read_error(exc, out_path)
            if error is None:
                raise
            raise error from exc
        # The wrapper already checked that an existing --output is ours to replace.
        if out_path.is_dir():
            shutil.rmtree(out_path)
        elif out_path.exists():
            out_path.unlink()
        partial.rename(out_path)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    print(f"Wrote: {out_path}", file=sys.stderr)


def _chunk_sizes(ds):
    """``{dim: chunk sizes}`` of ``ds``'s dask-backed variables (the first to have a dim wins)."""
    sizes = {}
    for var in ds.pint.dequantify().variables.values():
        for dim, chunks in zip(var.dims, var.chunks or (), strict=False):
            sizes.setdefault(dim, chunks)
    return sizes


def _open_zarr(path, io_spec=None, dask=False, like=None):
    """Open, validate, and quantify one Zarr store.

    Numpy-backed unless ``dask`` is set (xarray would otherwise chunk with dask
    whenever it is installed). With ``dask``, a dim that ``like`` (a ``_chunk_sizes``
    result) also has, at the same length, takes its chunks; other dims are ``"auto"``.
    """
    if not path.exists():
        raise UsageError(f"input not found: {path}")
    import xarray as xr

    from weather_skills_core.standard_utils import normalize_latlon_coords, normalize_step_coord
    from weather_skills_core.units import normalize_unit_strings, quantify_dataset

    if path.is_file():
        import zipfile

        if not zipfile.is_zipfile(path):
            with path.open("rb") as f:
                magic = f.read(4)
            if magic in (b"CDF\x01", b"CDF\x02", b"\x89HDF"):
                raise UsageError(
                    f"{path} is a NetCDF file, not a Zarr store; convert it to a Zarr first"
                )
            raise UsageError(f"{path} is a file, not a Zarr store (a folder or a zipped Zarr)")
        store, group = provenance_mod.zipped_zarr(path)
        # Unconsolidated: consolidated reads miss a store one folder down in the zip.
        source, kwargs = store, {"group": group, "consolidated": False}
    else:
        source, kwargs = path, {"consolidated": True}
    chunks = "auto" if dask else None
    if like:
        sizes = xr.open_zarr(source, chunks=None, **kwargs).sizes
        chunks = {d: like[d] if sum(like.get(d, ())) == n else "auto" for d, n in sizes.items()}
    ds = xr.open_zarr(source, chunks=chunks, **kwargs)
    if io_spec is not None:
        std.validate_input(ds, io_spec, str(path))
    ds = normalize_unit_strings(ds)
    ds = normalize_step_coord(ds)
    # Same lat/lon snap as on write, so a Zarr written before float32 coords
    # still joins cell-for-cell with a new one.
    ds = normalize_latlon_coords(ds)
    return quantify_dataset(ds)


def open_dataset_params(params, arguments, dask=False):
    """Replace Dataset-typed Path values with opened/validated/quantified datasets.

    Values (or list items) that expose ``zarr_paths() -> list[Path]`` are also
    opened and hashed. Identical paths are opened once and reused. Each holder
    gets ``.ds`` (one path) or ``.datasets`` (several). With ``dask``, every input
    after the first opens with the first input's chunk sizes on the dims they share
    (same length), so combining inputs elementwise stays blockwise.
    """
    input_paths: list[Path] = []
    upstream: list = []
    cache: dict = {}
    like: dict = {}

    def open_one(path, io_spec=None):
        path = Path(path)
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = _open_zarr(path, io_spec, dask, like)
            if dask and not like:
                like.update(_chunk_sizes(cache[resolved]))
            input_paths.append(path)
            upstream.append(provenance_mod.load_history(path))
        ds = cache[resolved]
        ds.attrs[INPUT_PATH_ATTR] = str(path)
        return ds

    for arg in arguments:
        raw = params.get(arg.dest)
        if arg.dataset_type is None or raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            params[arg.dest] = [open_one(p, arg.dataset_type.io_spec) for p in raw]
        else:
            params[arg.dest] = open_one(raw, arg.dataset_type.io_spec)

    seen: set[int] = set()
    for raw in list(params.values()):
        for holder in raw if isinstance(raw, (list, tuple)) else (raw,):
            if not callable(getattr(holder, "zarr_paths", None)) or id(holder) in seen:
                continue
            seen.add(id(holder))
            opened = [open_one(p) for p in holder.zarr_paths() or []]
            if len(opened) == 1:
                holder.ds = opened[0]
            elif opened:
                holder.datasets = opened

    first_ds = next(iter(cache.values()), None)
    return params, input_paths, upstream, first_ds


def argument(*option_strings, **kwargs):
    """Declare a CLI flag under ``@weather_skill`` (argparse-style)."""
    arg = Argument(*option_strings, **kwargs)

    def decorate(fn):
        existing = list(getattr(fn, ARGS_ATTR, []))
        existing.insert(0, arg)
        setattr(fn, ARGS_ATTR, existing)
        return fn

    return decorate


def weather_skill(
    *,
    name,
    version,
    output: bool = True,
):
    """Turn a function into a weather skill CLI with validated I/O and provenance.

    Stack ``@weather_skill.argument`` for flags. Use ``type=Dataset(...)`` for
    Zarr inputs (opened and dim-checked before the skill runs). Dataset
    ``--input`` is passed to the skill as ``ds`` (a list when ``action="append"``).
    Converted values that expose ``zarr_paths() -> list[Path]`` are opened and
    hashed the same way (identical paths are reused); each holder gets ``.ds``
    or ``.datasets``.

    When ``output=True`` (default), the decorator owns ``-o/--output``
    (repeatable). It injects ``output`` as a ``Path`` (one path) or
    ``list[Path]`` (several). Returning an ``xr.Dataset`` writes Zarr there;
    returning a ``Path`` stamps that file; returning a sequence writes one
    artifact per ``--output``. The number of returned values must match the
    number of ``--output`` paths. Returning ``None`` skips decorator write
    (skill already wrote). Set ``output=False`` for inspect-only skills.
    ``probe=True`` on an argument drops required flags and skips writing.

    The decorator also owns ``--dask``, ``--dask-workers N`` (default
    min(4, CPUs)) and ``--dask-chunk-size SIZE`` (default 64MiB) on every skill
    (do not declare them); the skill receives them as the plain kwargs ``dask``
    (bool), ``dask_workers`` (int) and ``dask_chunk_size`` (str). With ``--dask``
    the run uses dask's threaded scheduler, inputs open in dask chunks, and a
    skill that must handle data larger than memory opens its remote sources with
    ``chunks="auto" if dask else None`` and keeps arrays lazy; the result is
    rechunked uniformly and written as a sharded Zarr v3 (shards near the chunk
    size, ~1 MiB inner chunks). Without ``--dask``, inputs are numpy-backed and
    the write has no sharding or rechunking. ``--dask`` needs the skill to list
    ``weather-skills-core[dask]`` in its script header (else a usage error); the
    other two flags are unused without it.

    Every Dataset is written to a sibling path renamed into place, so a failure
    leaves no partial output. A network error raised while writing (a lazy read
    of the source: aiohttp, gcsfs or requests) is reported as a ``DataError``
    naming the output.

    The skill also receives ``exit_stack``, a ``contextlib.ExitStack`` closed
    after every output is written (or the run fails), with or without ``--dask``.
    Lazy data that depends on a temp dir or session registers it there:
    ``exit_stack.enter_context(tempfile.TemporaryDirectory())``,
    ``exit_stack.callback(fs.close)``.

    Skills open precip rates and amounts alike. The operation that cannot take
    amounts is ``rate_to_total`` (used by ``convert-to-totals``): multiplying
    an amount by ``aggregation_period`` would double-count.

    On every Dataset write the decorator also normalizes GRIB unit strings,
    stamps precip-amount CF names when units are amounts, casts ``step`` to
    ``timedelta64[ns]``, rounds lat/lon to 5 decimal places as ``float32``,
    and fills data-var attrs stripped by the skill from the first input
    (same variable names). Value conversion (``to_standard_units``) stays
    skill-owned.
    """

    def decorator(fn):
        arguments = list(getattr(fn, ARGS_ATTR, []))
        for arg in arguments:
            if not isinstance(arg, Argument):
                raise TypeError(
                    f"skill {name!r}: stacked arguments must be Argument instances; "
                    f"got {type(arg).__name__}"
                )
            flags = set(arg.option_strings)
            if output and (flags & {"-o", "--output"} or arg.dest == "output"):
                raise ValueError(
                    f"skill {name!r}: do not declare -o/--output; the decorator owns it "
                    "(pass output=False to opt out)"
                )
            if set(standard_args.DASK_FLAGS) & flags or arg.dest in standard_args.DASK_DESTS:
                raise ValueError(
                    f"skill {name!r}: do not declare --dask/--dask-workers/--dask-chunk-size; "
                    "the decorator owns them"
                )

        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(fn).parameters.values()
        ):
            raise TypeError(
                f"skill {name!r} must accept **kwargs so the decorator can pass "
                "extra runtime information"
            )

        parser = argparse.ArgumentParser(
            description=fn.__doc__,
            epilog=f"skill version: {version}",
        )
        if output:
            parser.add_argument(
                "-o",
                "--output",
                dest="output",
                action="append",
                required=True,
                metavar="PATH",
                help="Output path (repeat once per returned artifact).",
            )
        parser.add_argument(
            "--dask",
            action="store_true",
            help="Process data larger than memory in chunks (threaded dask).",
        )
        parser.add_argument(
            "--dask-workers",
            type=int,
            default=min(4, os.cpu_count() or 1),
            metavar="N",
            help="Dask worker threads (default: %(default)s; needs --dask).",
        )
        parser.add_argument(
            "--dask-chunk-size",
            default="64MiB",
            metavar="SIZE",
            help="Target dask chunk size (default: %(default)s; needs --dask).",
        )
        for arg in arguments:
            kwargs = dict(arg.kwargs)
            if arg.dataset_type is not None:
                label = arg.dataset_type.help_label()
                kwargs.setdefault("metavar", "PATH")
                if "help" not in arg.kwargs:
                    kwargs["help"] = f"Input Zarr ({label})."
                elif label not in str(kwargs.get("help", "")):
                    kwargs["help"] = f"{kwargs['help']} [{label}]"
            if arg.dest in standard_args.STANDARD_HELP:
                kwargs = standard_args.add_standard_help(
                    kwargs, standard_args.STANDARD_HELP[arg.dest]
                )
            parser.add_argument(*arg.option_strings, **kwargs)

        strip_dests = {arg.dest for arg in arguments if arg.dataset_type is not None}
        strip_dests |= {"output", *standard_args.DASK_DESTS}
        # Options taking one value (store/append), for rewrite_negative_values.
        value_options = {
            opt
            for action in parser._actions
            if action.nargs is None
            for opt in action.option_strings
        }

        @functools.wraps(fn)
        def wrapper(argv=None):
            argv = sys.argv[1:] if argv is None else list(argv)
            argv = standard_args.rewrite_negative_values(argv, value_options)
            # Closed after every output is written, or the run fails.
            exit_stack = contextlib.ExitStack()

            try:
                probing = any(
                    arg.probe and argv_has_option(argv, arg.option_strings) for arg in arguments
                )
                saved_required = [(action, action.required) for action in parser._actions]
                try:
                    if probing:
                        for action in parser._actions:
                            action.required = False
                    args = parser.parse_args(argv)
                finally:
                    for action, required in saved_required:
                        action.required = required
                chunk_bytes = _enter_dask(args, exit_stack) if args.dask else None
                params = standard_args.convert_standard_args(args, arguments)
                params, input_paths, upstream, first_ds = open_dataset_params(
                    params, arguments, args.dask
                )
                if (
                    any(a.dataset_type is not None and a.dest == "input" for a in arguments)
                    and "input" in params
                ):
                    params["ds"] = params.pop("input")

                output_paths: list[Path] = []
                if output:
                    output_paths = [Path(p) for p in (args.output or [])]
                    # Check every --output now, before the skill runs. A bad path
                    # would otherwise only fail at write time, after a fetch that
                    # can take minutes, and the work would be lost.
                    for path in output_paths:
                        # A file somewhere above the output (e.g. -o rain.zip/out.zarr)
                        # means its folder can never be created, so to_zarr would fail.
                        blocker = next((p for p in path.parents if p.is_file()), None)
                        if blocker is not None:
                            raise UsageError(
                                f"--output {path} is inside {blocker}, which is a file, "
                                "not a folder; pass a path under a folder"
                            )
                        # Writing replaces whatever is at --output (write_output
                        # deletes it first). A mistyped -o could name the user's own
                        # folder or file, so only replace what is safe to lose: an
                        # empty folder or an earlier skill output, which carries
                        # weather_skills_history. See can_replace_output.
                        if not provenance_mod.can_replace_output(path):
                            raise UsageError(
                                f"--output {path} already exists and was not written by a "
                                "weather skill (it has no weather_skills_history), so it may "
                                "be your own data; pass a new path"
                            )
                    if not output_paths:
                        params["output"] = None
                    else:
                        params["output"] = (
                            output_paths[0] if len(output_paths) == 1 else output_paths
                        )

                params["dask"] = args.dask
                params["dask_workers"] = args.dask_workers
                params["dask_chunk_size"] = args.dask_chunk_size
                params["exit_stack"] = exit_stack
                result = fn(**params)
                if not output or probing or result is None:
                    return result

                results = list(result) if isinstance(result, (list, tuple)) else [result]
                if len(results) != len(output_paths):
                    raise SkillError(
                        f"skill returned {len(results)} value(s), "
                        f"but {len(output_paths)} --output path(s) were given"
                    )

                history = build_history(
                    name,
                    version,
                    args,
                    params,
                    input_paths,
                    upstream,
                    strip_dests,
                    revision=provenance_mod.resolve_skill_revision(fn),
                )
                for value, out_path in zip(results, output_paths, strict=True):
                    write_output(value, out_path, history, first_ds, chunk_bytes)
                return result
            except SkillError as exc:
                msg = str(exc)
                print(msg if not exc.prefix else f"Error: {msg}", file=sys.stderr)
                sys.exit(exc.exit_code)
            finally:
                exit_stack.close()

        wrapper.parser = parser
        return wrapper

    return decorator


weather_skill.argument = argument  # type: ignore[attr-defined]

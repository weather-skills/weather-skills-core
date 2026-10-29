"""Shared CLI flags (``--bbox``, dates, …) for the decorator and linting catalog."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
from dataclasses import dataclass

from weather_skills_core.errors import UsageError
from weather_skills_core.standard_utils import parse_bbox, parse_date

STANDARD_HELP = {
    "bbox": "N/W/S/E decimal degrees.",
    "date": "Absolute date YYYY-MM-DD.",
    "start_time": "Range start, inclusive. Absolute YYYY-MM-DD.",
    "end_time": "Range end, inclusive. Absolute YYYY-MM-DD.",
}


# Reserved flags the decorator adds to every skill (like -o/--output).
DASK_FLAGS = ("--dask", "--no-dask", "--dask-workers", "--dask-chunk-size")
DASK_DESTS = ("dask", "dask_workers", "dask_chunk_size")
# Target chunk size of a run without --dask is the same default, in bytes (no dask needed).
DEFAULT_DASK_CHUNK_SIZE = "64MiB"
DEFAULT_CHUNK_BYTES = 64 * 2**20


@dataclass(frozen=True)
class StandardParameter:
    name: str
    flags: tuple


@dataclass(frozen=True)
class DaskSettings:
    """Larger-than-memory mode, handed to the skill as ``dask`` (None without ``--dask``)."""

    workers: int
    chunk_size: str


def standard_parameters():
    """Shared flags (``--bbox``, dates, …) used by the decorator (and the linting package).

    Input/output paths are ordinary ``@weather_skill.argument`` declarations
    (``type=Dataset(...)`` / ``type=Path``); they are not catalogued here.
    """
    return (
        StandardParameter("start_time", ("--start-time",)),
        StandardParameter("end_time", ("--end-time",)),
        StandardParameter("date", ("--date",)),
        StandardParameter("bbox", ("--bbox",)),
        StandardParameter("variable", ("--variable", "-v")),
    )


def add_dask_arguments(parser, *, default: bool = False) -> None:
    """Add the reserved ``--dask`` / ``--no-dask`` flags to ``parser``; ``default`` is the state."""
    parser.add_argument(
        "--dask",
        action=argparse.BooleanOptionalAction,
        default=default,
        help="Process data larger than memory in chunks with threaded dask "
        f"(default: {'on' if default else 'off'}).",
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        metavar="N",
        help="Dask worker threads (default: min(4, CPU count)). Needs dask on.",
    )
    parser.add_argument(
        "--dask-chunk-size",
        metavar="SIZE",
        help=f"Target dask chunk size (default: {DEFAULT_DASK_CHUNK_SIZE}). Needs dask on.",
    )


def dask_settings(args) -> DaskSettings | None:
    """The ``DaskSettings`` for parsed ``args``, or None without ``--dask``.

    Raises UsageError for a dask option without ``--dask``, a bad value, or
    dask mode when dask is not installed.
    """
    if not args.dask:
        given = [
            flag
            for flag, value in (
                ("--dask-workers", args.dask_workers),
                ("--dask-chunk-size", args.dask_chunk_size),
            )
            if value is not None
        ]
        if given:
            raise UsageError(f"{' and '.join(given)} only apply when dask is on (--dask)")
        return None
    try:
        import dask  # noqa: F401
        from dask.utils import parse_bytes
    except ImportError:
        raise UsageError(
            "dask mode needs dask, which this skill does not have; the skill must list "
            "`weather-skills-core[dask]` in its script header dependencies"
        ) from None
    workers = args.dask_workers if args.dask_workers is not None else min(4, os.cpu_count() or 1)
    if workers < 1:
        raise UsageError(f"--dask-workers must be at least 1; got {workers}")
    chunk_size = args.dask_chunk_size or DEFAULT_DASK_CHUNK_SIZE
    try:
        if parse_bytes(chunk_size) < 1:
            raise ValueError(chunk_size)
    except ValueError:
        raise UsageError(f"--dask-chunk-size {chunk_size!r} is not a size like 100MiB") from None
    return DaskSettings(workers, chunk_size)


def chunk_bytes(settings: DaskSettings | None) -> int:
    """The target chunk size in bytes: the ``--dask-chunk-size``, else the default."""
    if settings is None:
        return DEFAULT_CHUNK_BYTES
    from dask.utils import parse_bytes

    return parse_bytes(settings.chunk_size)


def dask_config(settings: DaskSettings | None):
    """Context manager running dask's threaded scheduler under ``settings`` (a no-op if None)."""
    if settings is None:
        return contextlib.nullcontext()
    import dask

    return dask.config.set(
        scheduler="threads",
        num_workers=settings.workers,
        **{"array.chunk-size": settings.chunk_size},
    )


def rewrite_negative_values(argv, value_options):
    """Join ``--opt -1.28,36.82`` into ``--opt=-1.28,36.82``.

    argparse reads any token starting with ``-`` that isn't a plain number as
    a flag, so a southern-hemisphere ``--point`` or a negative-north ``--bbox``
    fails with "expected one argument". For options in ``value_options`` (those
    taking one value), a following token that starts with ``-`` and a digit or
    ``.`` is that option's value.
    """
    out = []
    i = 0
    while i < len(argv):
        nxt = argv[i + 1] if i + 1 < len(argv) else ""
        if argv[i] in value_options and re.match(r"-[\d.]", nxt):
            out.append(f"{argv[i]}={nxt}")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def add_standard_help(kwargs: dict, canonical: str) -> dict:
    """Attach standard help text to an ``add_argument`` kwargs dict."""
    out = dict(kwargs)
    existing = out.get("help")
    if existing:
        text = str(existing).rstrip()
        if canonical not in text:
            out["help"] = f"{text} {canonical}"
    else:
        out["help"] = canonical
    return out


def convert_standard_args(args, arguments) -> dict:
    """Build skill kwargs from parsed CLI.

    ``--bbox`` becomes an ``(N, W, S, E)`` float tuple. Dates become ``datetime.date``.
    Only absolute values are parsed: a ``N/W/S/E`` bbox and ``YYYY-MM-DD``
    dates. Place names and relative dates are resolved before the call.

    A skill that declares a date flag with ``action="append"`` gets a list of
    dates. Repeated ``--date`` gives one date per flag. Repeated
    ``--start-time`` / ``--end-time`` pair up: the Nth start with the Nth end,
    one window each.

    Raises UsageError if a ``start_time`` is after its ``end_time``, or the
    repeated starts and ends do not pair up.
    """
    params = {}
    for arg in arguments:
        dest = arg.dest
        raw = getattr(args, dest)
        if dest == "bbox":
            params[dest] = parse_bbox(raw) if raw is not None else None
        elif dest in ("date", "start_time", "end_time") and isinstance(raw, list):
            params[dest] = [parse_date(value) for value in raw]
        elif dest in ("date", "start_time", "end_time"):
            params[dest] = parse_date(raw) if raw is not None else None
        else:
            params[dest] = raw

    starts = params.get("start_time")
    ends = params.get("end_time")
    if isinstance(starts, list) or isinstance(ends, list):
        # One side may be a plain flag: a single date counts as one.
        starts = starts if isinstance(starts, list) else [] if starts is None else [starts]
        ends = ends if isinstance(ends, list) else [] if ends is None else [ends]
        if len(starts) != len(ends):
            raise UsageError(
                f"got {len(starts)} --start-time and {len(ends)} --end-time; "
                "pass them in pairs, one pair per window."
            )
    else:
        starts, ends = [starts], [ends]
    for start, end in zip(starts, ends, strict=True):
        if start is not None and end is not None and start > end:
            raise UsageError(
                f"--start-time {start.isoformat()} is after --end-time {end.isoformat()}."
            )
    return params

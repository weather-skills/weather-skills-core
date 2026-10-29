"""Chunk layout for writing a Dataset to Zarr, including dask-backed ones.

Zarr needs uniform chunks (only the last may be short), which ``isel`` /
``concat`` / ``sortby`` on dask arrays do not give. With ``--dask`` every output
is written sharded: shards near a target size (one per dask chunk), each cut into
small inner chunks for partial reads.
"""

import math

# Target size of one inner chunk (the unit a reader fetches from a shard).
INNER_CHUNK_BYTES = 1 << 20


def _divisors(n: int) -> list[int]:
    small = [d for d in range(1, math.isqrt(n) + 1) if n % d == 0]
    return sorted({*small, *(n // d for d in small)})


def _next_divisor(n: int, below: int) -> int:
    return max(d for d in _divisors(n) if d < below)


def inner_chunks(shard: tuple, itemsize: int) -> tuple:
    """Balanced chunk shape of about ``INNER_CHUNK_BYTES`` that divides ``shard`` per dim."""
    budget = max(1, INNER_CHUNK_BYTES // itemsize)
    inner = list(shard)
    while math.prod(inner) > budget:
        # Shrink the dim that loses least (to its next smaller divisor); on a tie the
        # larger one, so tiles stay balanced and a prime-ish dim is left whole.
        i = min(
            (i for i, c in enumerate(inner) if c > 1),
            key=lambda i: (inner[i] / _next_divisor(shard[i], inner[i]), -inner[i]),
        )
        inner[i] = _next_divisor(shard[i], inner[i])
    return tuple(inner)


def prepare_chunks(ds, chunk_bytes: int):
    """Return ``(ds, encoding)`` ready for ``to_zarr``: every data variable sharded.

    Shard shape is a uniform chunk shape near ``chunk_bytes`` (dask's ``"auto"``,
    keeping on-disk chunk multiples where it can), with inner chunks from
    ``inner_chunks`` (Zarr v3 sharding). Dask-backed variables are rechunked to the
    shard shape, so each dask chunk maps to whole shards. Dask-backed coordinates
    are made uniform too.
    """
    from dask.array.core import normalize_chunks

    updates, encoding = {}, {}
    for name, var in ds.variables.items():
        if name in ds.indexes or var.ndim == 0 or var.size == 0 or var.dtype.hasobject:
            continue
        lazy = var.chunks is not None
        if not (lazy or name in ds.data_vars):
            continue
        sizes = normalize_chunks(
            "auto", var.shape, dtype=var.dtype, limit=chunk_bytes, previous_chunks=var.chunks
        )
        shard = tuple(c[0] for c in sizes)
        if lazy:
            updates[name] = var.chunk(dict(zip(var.dims, shard, strict=True)))
        if name in ds.data_vars:
            encoding[name] = {"chunks": inner_chunks(shard, var.dtype.itemsize), "shards": shard}
    return ds.assign(updates), encoding

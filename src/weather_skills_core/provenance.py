"""Provenance graph handling for weather-skill artifacts.

``weather_skills_history`` is a JSON-encoded array of entries. A single-input
path stays oldest-first (a path DAG). A multi-input skill records a join: the
top-level array is just that skill's entry, and every parent subgraph lives
under ``input[].history``. Each new entry also records the git ``commit``
(and optional ``repo`` / ``dirty``) of the skill that ran.
"""

import hashlib
import html
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

HISTORY_ATTR = "weather_skills_history"
SOURCE_ATTR = "weather_skills_source"
# Tables keep their history in a sidecar file (see history_sidecar).
TABLE_SUFFIXES = (".csv", ".xlsx", ".xlsm")

_EXIF_USER_COMMENT = 0x9286  # EXIF UserComment tag
_HTML_META_RE = re.compile(
    rf'<meta\s+name=["\']{re.escape(HISTORY_ATTR)}["\']\s+content=["\'](.*?)["\']\s*/?>',
    re.IGNORECASE | re.DOTALL,
)
_SVG_META_RE = re.compile(
    rf'<metadata\s+id=["\']{re.escape(HISTORY_ATTR)}["\']\s*>(.*?)</metadata>', re.DOTALL
)
_SVG_STAMP_RE = re.compile(r'<g\s+id="weather_skills_stamp".*?</g>', re.DOTALL)
# The PDF stamp is a marked-content block, so a restamp can find and drop it.
_PDF_STAMP_RE = re.compile(rb"/weather_skills_stamp BMC\b.*?\bEMC\s*", re.DOTALL)

# The provenance stamp: a ring around a star, bottom-right, drawn on a figure
# whose history chain is intact. One shape, drawn as pixels (PNG, JPEG, GIF
# frames) or as vector shapes (SVG, PDF), in rubber-stamp crimson (RGBA).
_MARK_INK = (165, 15, 45, 235)


def hash_zarr(zarr_path: Path) -> str:
    """Stable sha256 of a zarr directory's relative paths + file bytes.

    A single file (a zipped Zarr, a table) hashes its bytes.
    """
    zarr_path = Path(zarr_path)
    if zarr_path.is_file():
        with zarr_path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    h = hashlib.sha256()
    for p in sorted(zarr_path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(zarr_path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def parse_chain(raw: str) -> list:
    """Strictly parse ``weather_skills_history`` JSON into a list."""
    try:
        chain = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("value is not valid JSON") from None
    if not isinstance(chain, list):
        raise ValueError("value is not a JSON array")  # noqa: TRY004
    return chain


def coerce_chain(raw: str, label: str) -> list | None:
    """Lenient parse for render paths; warns and returns None if malformed."""
    try:
        chain = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        chain = None
    if not isinstance(chain, list):
        print(
            f"ignoring malformed weather_skills_history on {label}; "
            "run `provenance --check` for details",
            file=sys.stderr,
        )
        return None
    return chain


_ENTRY_KNOWN_KEYS = {"skill", "version", "args", "input", "commit", "repo", "dirty"}
_INPUT_ITEM_KNOWN_KEYS = {"basename", "hash", "history"}
_GIT_TIMEOUT_SECONDS = 2
_GITHUB_REMOTE_RE = re.compile(
    r"^(?:git@github\.com:|https://github\.com/)([^/]+)/(.+?)(?:\.git)?/?$"
)


def _validate_input(value, loc: str, violations: list, notes: list) -> None:
    if value is None:
        return

    def _check_item(item, item_loc: str) -> None:
        if not isinstance(item, dict):
            violations.append(f"{item_loc}: input entry is not an object")
            return
        if "basename" not in item:
            violations.append(f"{item_loc}: missing required key 'basename'")
        elif not isinstance(item["basename"], str):
            violations.append(f"{item_loc}.basename: must be a string")
        if "hash" not in item:
            violations.append(f"{item_loc}: missing required key 'hash'")
        elif not isinstance(item["hash"], str):
            violations.append(f"{item_loc}.hash: must be a string")
        if "history" in item:
            _validate_chain(item["history"], f"{item_loc}.history", violations, notes)
        for key in item:
            if key not in _INPUT_ITEM_KNOWN_KEYS:
                notes.append(f"{item_loc}: unknown key {key!r}")

    if isinstance(value, list):
        for j, item in enumerate(value):
            _check_item(item, f"{loc}[{j}]")
        return
    if isinstance(value, dict):
        _check_item(value, loc)
        return
    violations.append(f"{loc}: must be null, an object, or an array of objects")


def _validate_chain(chain, loc: str, violations: list, notes: list) -> None:
    if not isinstance(chain, list):
        violations.append(f"{loc}: value is not a JSON array")
        return
    for i, entry in enumerate(chain):
        eloc = f"{loc}[{i}]"
        if not isinstance(entry, dict):
            violations.append(f"{eloc}: entry is not an object")
            continue
        if "skill" not in entry:
            violations.append(f"{eloc}: missing required key 'skill'")
        elif not isinstance(entry["skill"], str):
            violations.append(f"{eloc}.skill: must be a string")
        elif not entry["skill"]:
            violations.append(f"{eloc}.skill: must be a non-empty string")
        if "version" not in entry:
            violations.append(f"{eloc}: missing required key 'version'")
        elif not isinstance(entry["version"], str):
            violations.append(f"{eloc}.version: must be a string")
        if "args" not in entry:
            violations.append(f"{eloc}: missing required key 'args'")
        elif not isinstance(entry["args"], dict):
            violations.append(f"{eloc}.args: must be an object")
        if "input" not in entry:
            violations.append(f"{eloc}: missing required key 'input'")
        else:
            _validate_input(entry["input"], f"{eloc}.input", violations, notes)
        for key in ("commit", "repo"):
            if key in entry and not (isinstance(entry[key], str) and entry[key]):
                violations.append(f"{eloc}.{key}: must be a non-empty string")
        if "dirty" in entry and not isinstance(entry["dirty"], bool):
            violations.append(f"{eloc}.dirty: must be a boolean")
        for key in entry:
            if key not in _ENTRY_KNOWN_KEYS:
                notes.append(f"{eloc}: unknown key {key!r}")


def validate_chain(chain, loc: str) -> tuple[list, list]:
    """Validate a parsed history chain. Returns ``(violations, notes)``."""
    violations: list = []
    notes: list = []
    _validate_chain(chain, loc, violations, notes)
    return violations, notes


def chain_is_intact(history) -> bool:
    """True if ``history`` is a non-empty schema-valid provenance chain."""
    if not isinstance(history, list) or not history:
        return False
    violations, _notes = validate_chain(history, HISTORY_ATTR)
    return not violations


def _stamp_shape(width: float, height: float, px: float = 1.0):
    """Ring and star of the stamp in the figure's own units, y down.

    ``px`` is figure units per pixel (0.75 for PDF points). Returns
    ``(cx, cy, ring_radius, stroke, star_points)``, or ``None`` when the
    figure is under 96 px on its short side.
    """
    import math

    w, h = width / px, height / px
    if min(w, h) < 96:
        return None
    # About 3.5% of the short side, 24-28 px across.
    diameter = max(24, min(int(min(w, h) * 0.035), 28))
    margin = max(4, int(min(w, h) * 0.015))
    outer_r = diameter / 2 - 1
    stroke = max(2, diameter / 22)
    cx, cy = w - margin - diameter / 2, h - margin - diameter / 2
    star = []
    for i in range(10):
        r = outer_r * 0.48 if i % 2 == 0 else outer_r * 0.48 * 0.42
        # Turned 6 degrees clockwise so it looks hand-inked rather than UI chrome.
        a = -math.pi / 2 + i * math.pi / 5 + math.radians(6)
        star.append(((cx + r * math.cos(a)) * px, (cy + r * math.sin(a)) * px))
    return cx * px, cy * px, (outer_r - stroke / 2) * px, stroke * px, star


def _draw_official_mark(img):
    """Return a copy of ``img`` with the stamp composited on (unchanged if too small)."""
    from PIL import Image, ImageDraw

    shape = _stamp_shape(*img.size)
    if shape is None:
        return img.copy()
    cx, cy, r, stroke, star = shape

    # Draw at 4x on a small tile around the stamp, then shrink it, for smooth edges.
    scale = 4
    x0, y0 = int(cx - r - stroke), int(cy - r - stroke)
    size = int(2 * (r + stroke)) + 2
    tile = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    tx, ty, outer = (cx - x0) * scale, (cy - y0) * scale, (r + stroke / 2) * scale
    draw.ellipse(
        (tx - outer, ty - outer, tx + outer, ty + outer),
        outline=_MARK_INK,
        width=round(stroke * scale),
    )
    draw.polygon([((x - x0) * scale, (y - y0) * scale) for x, y in star], fill=_MARK_INK)
    tile = tile.resize((size, size), resample=Image.Resampling.LANCZOS)

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste(tile, (x0, y0), tile)
    marked = Image.alpha_composite(base, overlay)
    if img.mode == "RGBA":
        return marked
    # Stay in RGB after compositing. Converting a palette image back to ``P``
    # requantizes and can merge nearby fills (BoM IOD pink / blue both become
    # one purple).
    return marked.convert("RGB")


def zipped_zarr(path: Path):
    """``(store, group)`` for a zipped Zarr: the store at the zip root, or in one top folder.

    Zipping a ``name.zarr`` folder usually puts the store under ``name.zarr/``
    inside the archive, so a root without Zarr metadata looks one level down.
    """
    import zipfile

    import zarr

    from weather_skills_core.errors import UsageError

    markers = ("zarr.json", ".zgroup", ".zmetadata")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    if any(n in markers for n in names):
        return zarr.storage.ZipStore(path, mode="r"), None
    tops = sorted(
        {n.split("/")[0] for n in names if n.count("/") == 1 and n.split("/")[1] in markers}
    )
    if len(tops) == 1:
        return zarr.storage.ZipStore(path, mode="r"), tops[0]
    found = ", ".join(tops) if tops else "none"
    raise UsageError(
        f"{path} is a zip but holds no single Zarr store at its root or one folder down "
        f"(Zarr folders found: {found}); zip one .zarr folder, or upload NetCDF instead"
    )


def _svg_official_mark(text: str, mark: bool) -> str:
    """Drop any stamp from SVG ``text``; when ``mark``, add one as the last element."""
    text = _SVG_STAMP_RE.sub("", text)
    root = re.search(r"<svg\b[^>]*>", text)
    root = root[0] if root else ""
    size = re.search(r'viewBox="[-\d.]+[ ,]+[-\d.]+[ ,]+([\d.]+)[ ,]+([\d.]+)"', root)
    if size is None:
        # No viewBox: user units are the pixel width and height.
        width = re.search(r'\swidth="([\d.]+)(?:px)?"', root)
        height = re.search(r'\sheight="([\d.]+)(?:px)?"', root)
        size = (None, width[1], height[1]) if width and height else None
    shape = _stamp_shape(float(size[1]), float(size[2])) if mark and size else None
    if shape is None:
        return text
    cx, cy, r, stroke, star = shape
    ink = "rgb({}, {}, {})".format(*_MARK_INK[:3])
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in star)
    stamp = (
        f'<g id="weather_skills_stamp" opacity="{_MARK_INK[3] / 255:.2f}">'
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="none" '
        f'stroke="{ink}" stroke-width="{stroke:.2f}"/>'
        f'<polygon points="{points}" fill="{ink}"/></g>'
    )
    # Drawn last, so it sits on top of the figure.
    end = text.rindex("</svg>")
    return text[:end] + stamp + text[end:]


def _pdf_official_mark(page, mark: bool) -> None:
    """Drop any stamp from a pypdf ``page``; when ``mark``, draw one over it."""
    from pypdf import PageObject
    from pypdf.generic import DecodedStreamObject, NameObject

    contents = page.get_contents()
    if contents is not None:
        data = contents.get_data()
        stripped = _PDF_STAMP_RE.sub(b"", data)
        if stripped != data:
            contents.set_data(stripped)
            page.replace_contents(contents)

    box = page.mediabox
    shape = _stamp_shape(float(box.width), float(box.height), px=0.75) if mark else None
    if shape is None:
        return
    cx, cy, r, stroke, star = shape
    # Ring as four Bezier arcs and a filled star, in PDF's y-up page space.
    x0, top = float(box.left), float(box.top)
    cx, cy = x0 + cx, top - cy
    k = 0.5523 * r
    ink = " ".join(f"{c / 255:.3f}" for c in _MARK_INK[:3])
    ops = [
        f"/weather_skills_stamp BMC q {ink} RG {ink} rg {stroke:.2f} w",
        f"{cx + r:.2f} {cy:.2f} m",
        f"{cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c",
        f"{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c",
        f"{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c",
        f"{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c S",
    ]
    for i, (x, y) in enumerate(star):
        ops.append(f"{x0 + x:.2f} {top - y:.2f} {'m' if i == 0 else 'l'}")
    ops.append("h f Q EMC")
    content = DecodedStreamObject()
    content.set_data("\n".join(ops).encode())
    overlay = PageObject.create_blank_page(width=box.width, height=box.height)
    overlay[NameObject("/Contents")] = content
    page.merge_page(overlay)


def load_history(zarr_path: Path) -> list:
    """Read a zarr store's history chain; empty on miss or malformation."""
    zarr_path = Path(zarr_path)
    try:
        import xarray as xr

        store, group = zipped_zarr(zarr_path) if zarr_path.is_file() else (zarr_path, None)
        with xr.open_zarr(store, group=group, consolidated=False) as ds:
            raw = ds.attrs.get(HISTORY_ATTR)
    except (OSError, KeyError, ValueError):
        return []
    if not raw:
        return []
    parsed = coerce_chain(raw, str(zarr_path))
    return [] if parsed is None else parsed


def input_ref(path: Path, history=None) -> dict:
    """One parent ``input`` value: ``{basename, hash}`` plus optional ``history``."""
    path = Path(path)
    ref = {"basename": path.name, "hash": hash_zarr(path)}
    if history is not None:
        ref["history"] = list(history)
    return ref


def origin_skill(history) -> str | None:
    """Oldest skill name, walking into the first nested parent of a join."""
    if not isinstance(history, list) or not history or not isinstance(history[0], dict):
        return None
    first = history[0]
    if len(history) == 1:
        parents = first.get("input")
        for item in parents if isinstance(parents, list) else [parents]:
            if isinstance(item, dict) and (found := origin_skill(item.get("history"))):
                return found
    skill = first.get("skill")
    return skill.strip() if isinstance(skill, str) and skill.strip() else None


def normalize_repo_url(url: str) -> str:
    """Turn a git remote into an https clone URL when we recognize GitHub."""
    text = url.strip()
    match = _GITHUB_REMOTE_RE.match(text)
    if match:
        return f"https://github.com/{match[1]}/{match[2]}"
    return text.removesuffix(".git")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _revision_from_direct_url(source_file: Path) -> dict | None:
    """Read PEP 610 ``direct_url.json`` for the installed dist that owns ``source_file``."""
    from importlib.metadata import distributions

    for dist in distributions():
        try:
            owned = any(Path(f.locate()).resolve() == source_file for f in dist.files or [])
            payload = json.loads(dist.read_text("direct_url.json") or "null") if owned else None
        except (OSError, ValueError):
            continue
        commit = ((payload or {}).get("vcs_info") or {}).get("commit_id")
        if commit:
            revision = {"commit": commit}
            if payload.get("url"):
                revision["repo"] = normalize_repo_url(payload["url"])
            return revision
    return None


def resolve_skill_revision(source=None) -> dict | None:
    """Git identity of the skill that is running.

    ``source`` may be a skill function or a filesystem path. Prefers the git
    checkout that contains the file; falls back to the installed package's
    ``direct_url.json`` (``uvx --from git+…@sha``). Returns ``None`` when
    neither is available. Untracked files do not count as dirty.
    """
    if source is None:
        return None
    try:
        path = Path(source if isinstance(source, (str, Path)) else inspect.getfile(source))
        path = path.resolve()
    except (OSError, TypeError):
        return None
    root = _git(["rev-parse", "--show-toplevel"], path.parent if path.is_file() else path)
    if not root:
        return _revision_from_direct_url(path) if path.is_file() else None
    commit = _git(["rev-parse", "HEAD"], root)
    if not commit:
        return None
    revision = {"commit": commit}
    remote = _git(["config", "--get", "remote.origin.url"], root)
    if remote:
        revision["repo"] = normalize_repo_url(remote)
    if _git(["status", "--porcelain", "-uno"], root):
        revision["dirty"] = True
    return revision


def build_entry(skill: str, version: str, args: dict, input, revision=None) -> dict:
    """Assemble a provenance entry, attaching git identity (``commit``/``repo``/``dirty``)."""
    return {"skill": skill, "version": version, "args": args, "input": input, **(revision or {})}


def stamp_zarr(ds, history: list, *, source: str | None = None) -> None:
    """Stamp history (and optional source) on a dataset; clear encodings."""
    ds.attrs[HISTORY_ATTR] = json.dumps(history, sort_keys=True)
    if source is not None:
        ds.attrs[SOURCE_ATTR] = source
    for v in ds.variables:
        ds[v].encoding = {}


def restamp_zarr(zarr_path: Path, history: list) -> None:
    """Rewrite history on an already-written zarr store in place."""
    import zarr

    group = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    group.attrs[HISTORY_ATTR] = json.dumps(history, sort_keys=True)
    zarr.consolidate_metadata(str(zarr_path))


def stamp_figure(path: Path, history: list) -> None:
    """Embed ``weather_skills_history`` into a PNG, JPEG, GIF, HTML, PDF, SVG, or GeoJSON/JSON
    file, or a sidecar next to a CSV or Excel table.

    When the chain is intact (non-empty and schema-valid), also draw the
    provenance stamp (a ring around a star, bottom-right) on PNG/JPEG pixels,
    every GIF frame, and SVG and PDF as vector shapes. HTML gets metadata only.
    """
    from weather_skills_core.errors import SkillError

    path = Path(path)
    payload = json.dumps(history, sort_keys=True)
    suffix = path.suffix.lower()
    mark = chain_is_intact(history)

    if suffix == ".png":
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        with Image.open(path) as img:
            out = _draw_official_mark(img) if mark else img.copy()
            info = PngInfo()
            for key, value in img.info.items():
                if isinstance(value, str) and key not in (HISTORY_ATTR, "Software"):
                    info.add_text(key, value)
            info.add_text(HISTORY_ATTR, payload)
            info.add_text("Software", "weather-skills")
            out.save(path, pnginfo=info)
        return

    if suffix in (".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(path) as img:
            out = _draw_official_mark(img) if mark else img.copy()
            if out.mode not in ("RGB", "L"):
                out = out.convert("RGB")
            exif = img.getexif()
            # ASCII UserComment: 8-byte charset header + payload
            exif[_EXIF_USER_COMMENT] = b"ASCII\x00\x00\x00" + payload.encode("ascii")
            out.save(path, exif=exif, quality=95)
        return

    if suffix == ".gif":
        import numpy as np
        from PIL import Image, ImageSequence

        # The history goes in the GIF comment block; each frame keeps its duration.
        with Image.open(path) as img:
            frames, durations = [], []
            for frame in ImageSequence.Iterator(img):
                durations.append(frame.info.get("duration", 0))
                rgb = frame.convert("RGB")
                marked = _draw_official_mark(rgb) if mark else rgb
                # Saving RGB would let Pillow pick a new 256-colour palette and recolour
                # the whole frame. Keep the frame's own colours, so every pixel off the
                # mark stays as it was; the mark gets the free palette slots, and
                # otherwise the nearest colour the frame has.
                colours = [c for _, c in rgb.getcolors(256)]
                ink = sorted(marked.getcolors(1 << 16), reverse=True)
                colours += [c for _, c in ink if c not in colours][: 256 - len(colours)]
                # Pillow's quantize(palette=...) matches colours only roughly and merges
                # close shades, so look each pixel's colour up exactly here.
                palette = np.array(colours, dtype=np.int32)
                pixels = np.asarray(marked, dtype=np.int32)
                seen, where = np.unique(pixels.reshape(-1, 3), axis=0, return_inverse=True)
                nearest = ((seen[:, None] - palette[None]) ** 2).sum(-1).argmin(1)
                out = Image.fromarray(nearest[where].reshape(pixels.shape[:2]).astype(np.uint8))
                out.putpalette(palette.astype(np.uint8).ravel().tolist())
                frames.append(out)
            loop = img.info.get("loop", 0)
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            comment=payload.encode("utf-8"),
        )
        return

    if suffix in (".html", ".htm"):
        text = path.read_text(encoding="utf-8")
        meta = f'<meta name="{HISTORY_ATTR}" content="{html.escape(payload, quote=True)}">'
        # Replace via a function: the JSON payload carries backslash escapes
        # (e.g. \u2014 for an em dash) that re.sub would parse as a template.
        if _HTML_META_RE.search(text):
            text = _HTML_META_RE.sub(lambda m: meta, text, count=1)
        elif re.search(r"<head[^>]*>", text, re.IGNORECASE):
            text = re.sub(
                r"(<head[^>]*>)", lambda m: f"{m.group(1)}\n{meta}", text, count=1, flags=re.I
            )
        else:
            text = meta + "\n" + text
        path.write_text(text, encoding="utf-8")
        return

    if suffix == ".pdf":
        from pypdf import PdfReader, PdfWriter

        # A custom key in the document info dictionary; viewers ignore it.
        writer = PdfWriter(clone_from=path)
        writer.pdf_header = PdfReader(path).pdf_header
        writer.add_metadata({f"/{HISTORY_ATTR}": payload})
        _pdf_official_mark(writer.pages[0], mark)
        writer.write(path)
        return

    if suffix == ".svg":
        text = path.read_text(encoding="utf-8")
        meta = f'<metadata id="{HISTORY_ATTR}">{html.escape(payload, quote=False)}</metadata>'
        if _SVG_META_RE.search(text):
            text = _SVG_META_RE.sub(lambda m: meta, text, count=1)
        else:
            text = re.sub(r"(<svg\b[^>]*>)", lambda m: f"{m.group(1)}{meta}", text, count=1)
        text = _svg_official_mark(text, mark)
        path.write_text(text, encoding="utf-8")
        return

    if suffix in (".geojson", ".json"):
        # A top-level key; in GeoJSON a foreign member, which RFC 7946 allows.
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise SkillError(
                f"{path} holds a JSON {type(doc).__name__}, not an object; "
                "history can only be stamped on a JSON object"
            )
        doc[HISTORY_ATTR] = payload
        path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        return

    if suffix in TABLE_SUFFIXES:
        # A table has nowhere to keep it, so the history goes in a sidecar with the
        # table's hash. A table edited after the skill wrote it no longer matches.
        sidecar = {"sha256": hash_zarr(path), HISTORY_ATTR: payload}
        history_sidecar(path).write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
        return

    raise SkillError(
        f"unsupported output type {suffix!r} for {path}; expected .png, .jpg/.jpeg, .gif, "
        ".html/.htm, .pdf, .svg, .geojson/.json, .csv, or .xlsx/.xlsm"
    )


def history_sidecar(path: Path) -> Path:
    """The ``<name>.history.json`` file next to a table that holds its history."""
    path = Path(path)
    return path.with_name(f"{path.name}.history.json")


def read_raw_history(path: Path) -> str | None:
    """The raw ``weather_skills_history`` JSON text stamped on an output file, or None."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as img:
            return img.info.get(HISTORY_ATTR) or None

    if suffix in (".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(path) as img:
            raw = img.getexif().get(_EXIF_USER_COMMENT)
        if isinstance(raw, bytes):
            if raw.startswith(b"ASCII\x00\x00\x00"):
                raw = raw[8:].decode("ascii")
            else:
                raw = raw.decode("utf-8", errors="replace")
        return raw or None

    if suffix == ".gif":
        from PIL import Image

        with Image.open(path) as img:
            raw = img.info.get("comment")
        return raw.decode("utf-8") if raw else None

    if suffix in (".html", ".htm"):
        m = _HTML_META_RE.search(path.read_text(encoding="utf-8"))
        return html.unescape(m.group(1)) if m else None

    if suffix == ".pdf":
        from pypdf import PdfReader

        raw = (PdfReader(path).metadata or {}).get(f"/{HISTORY_ATTR}")
        return str(raw) if raw else None

    if suffix == ".svg":
        m = _SVG_META_RE.search(path.read_text(encoding="utf-8"))
        return html.unescape(m.group(1)) if m else None

    if suffix in (".geojson", ".json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        return (doc.get(HISTORY_ATTR) or None) if isinstance(doc, dict) else None

    if suffix in TABLE_SUFFIXES:
        sidecar = history_sidecar(path)
        if not sidecar.is_file():
            return None
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("sha256") != hash_zarr(path):
            return None
        return meta.get(HISTORY_ATTR) or None

    return None


def load_figure_history(path: Path) -> list | None:
    """Read history from a stamped output file, or None if absent."""
    raw = read_raw_history(path)
    return coerce_chain(raw, Path(path).name) if raw else None


def can_replace_output(path: Path) -> bool:
    """True when a skill may write over ``path``: nothing there, an empty folder,
    or an earlier skill output (it carries a ``weather_skills_history``).

    Anything else may be the user's own data and is left alone.
    """
    path = Path(path)
    if not path.exists():
        return True
    try:
        if path.is_dir():
            return not any(path.iterdir()) or bool(load_history(path))
        return read_raw_history(path) is not None
    except Exception:  # noqa: BLE001
        # Unreadable as its type (e.g. a .png that is not an image): not ours.
        return False

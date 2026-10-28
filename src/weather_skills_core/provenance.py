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

_EXIF_USER_COMMENT = 0x9286  # EXIF UserComment tag
_HTML_META_RE = re.compile(
    rf'<meta\s+name=["\']{re.escape(HISTORY_ATTR)}["\']\s+content=["\'](.*?)["\']\s*/?>',
    re.IGNORECASE | re.DOTALL,
)
_SVG_META_RE = re.compile(
    rf'<metadata\s+id=["\']{re.escape(HISTORY_ATTR)}["\']\s*>(.*?)</metadata>', re.DOTALL
)


def hash_zarr(zarr_path: Path) -> str:
    """Stable sha256 of a zarr directory's relative paths + file bytes."""
    zarr_path = Path(zarr_path)
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


# The provenance stamp: a ring around a star, bottom-right, drawn on a figure
# whose history chain is intact. One shape, drawn as pixels (PNG, JPEG, GIF
# frames) or as vector shapes (SVG, PDF), in rubber-stamp crimson (RGBA).
_MARK_INK = (165, 15, 45, 235)


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


_SVG_STAMP_RE = re.compile(r'<g\s+id="weather_skills_stamp".*?</g>', re.DOTALL)


def _svg_official_mark(text: str, mark: bool) -> str:
    """Drop any stamp from SVG ``text``; when ``mark``, add one as the last element."""
    text = _SVG_STAMP_RE.sub("", text)
    size = re.search(r'<svg\b[^>]*?viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"', text)
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


def _pdf_official_mark(page) -> None:
    """Draw the stamp over a pypdf ``page`` (unchanged if too small)."""
    from pypdf import PageObject
    from pypdf.generic import DecodedStreamObject, NameObject

    box = page.mediabox
    shape = _stamp_shape(float(box.width), float(box.height), px=0.75)
    if shape is None:
        return
    cx, cy, r, stroke, star = shape
    # Ring as four Bezier arcs and a filled star, in PDF's y-up page space.
    x0, top = float(box.left), float(box.top)
    cx, cy = x0 + cx, top - cy
    k = 0.5523 * r
    ink = " ".join(f"{c / 255:.3f}" for c in _MARK_INK[:3])
    ops = [
        f"q {ink} RG {ink} rg {stroke:.2f} w",
        f"{cx + r:.2f} {cy:.2f} m",
        f"{cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c",
        f"{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c",
        f"{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c",
        f"{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c S",
    ]
    for i, (x, y) in enumerate(star):
        ops.append(f"{x0 + x:.2f} {top - y:.2f} {'m' if i == 0 else 'l'}")
    ops.append("h f Q")
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

        with xr.open_zarr(zarr_path, consolidated=False) as ds:
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
    """Embed ``weather_skills_history`` into a PNG, JPEG, GIF, HTML, PDF, or SVG file.

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
        from PIL import Image, ImageSequence

        # The history goes in the GIF comment block; each frame keeps its duration.
        with Image.open(path) as img:
            frames, durations = [], []
            for frame in ImageSequence.Iterator(img):
                durations.append(frame.info.get("duration", 0))
                rgb = frame.convert("RGB")
                frames.append(_draw_official_mark(rgb) if mark else rgb)
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
        if mark:
            _pdf_official_mark(writer.pages[0])
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

    raise SkillError(
        f"unsupported figure type {suffix!r} for {path}; expected .png, .jpg/.jpeg, .gif, "
        ".html/.htm, .pdf, or .svg"
    )


def load_figure_history(path: Path) -> list | None:
    """Read history from a stamped figure file, or None if absent."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as img:
            raw = img.info.get(HISTORY_ATTR)
        return coerce_chain(raw, path.name) if raw else None

    if suffix in (".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(path) as img:
            raw = img.getexif().get(_EXIF_USER_COMMENT)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            if raw.startswith(b"ASCII\x00\x00\x00"):
                raw = raw[8:].decode("ascii")
            else:
                raw = raw.decode("utf-8", errors="replace")
        return coerce_chain(raw, path.name)

    if suffix == ".gif":
        from PIL import Image

        with Image.open(path) as img:
            raw = img.info.get("comment")
        return coerce_chain(raw.decode("utf-8"), path.name) if raw else None

    if suffix in (".html", ".htm"):
        text = path.read_text(encoding="utf-8")
        m = _HTML_META_RE.search(text)
        if not m:
            return None
        return coerce_chain(html.unescape(m.group(1)), path.name)

    if suffix == ".pdf":
        from pypdf import PdfReader

        raw = (PdfReader(path).metadata or {}).get(f"/{HISTORY_ATTR}")
        return coerce_chain(str(raw), path.name) if raw else None

    if suffix == ".svg":
        m = _SVG_META_RE.search(path.read_text(encoding="utf-8"))
        return coerce_chain(html.unescape(m.group(1)), path.name) if m else None

    return None

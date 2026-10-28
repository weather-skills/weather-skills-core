"""Provenance helpers: history load/stamp and figure metadata."""

import json
import subprocess
from pathlib import Path

import pytest
from conftest import make_gridded
from PIL import Image

from weather_skills_core import provenance
from weather_skills_core.errors import SkillError


def write_store(path, history=None, fill=1.0):
    ds = make_gridded(fill=fill)
    if history is not None:
        provenance.stamp_zarr(ds, history)
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def entry(**overrides):
    base = {"skill": "demo", "version": "0.1.0", "args": {"x": 1}, "input": None}
    base.update(overrides)
    return base


def test_stamp_and_load_roundtrip(tmp_path):
    out = write_store(tmp_path / "a.zarr", history=[entry()])
    assert provenance.load_history(out) == [entry()]


def test_stamp_and_load_restamp(tmp_path):
    out = write_store(tmp_path / "a.zarr", history=[entry()])
    new = [entry(), entry(skill="next", version="0.2.0")]
    provenance.restamp_zarr(out, new)
    assert provenance.load_history(out) == new


def test_stamp_and_load_hash_stable(tmp_path):
    a = write_store(tmp_path / "a.zarr")
    b = write_store(tmp_path / "b.zarr")
    assert provenance.hash_zarr(a) == provenance.hash_zarr(b)


def test_stamp_and_load_build_entry():
    e = provenance.build_entry("s", "1.0.0", {"k": "v"}, None)
    assert e == {"skill": "s", "version": "1.0.0", "args": {"k": "v"}, "input": None}


def test_build_entry_records_revision():
    e = provenance.build_entry(
        "s",
        "1.0.0",
        {"k": "v"},
        None,
        revision={
            "commit": "abc123def456",
            "repo": "https://github.com/rhiza-research/weather-skills",
            "dirty": True,
        },
    )
    assert e["commit"] == "abc123def456"
    assert e["repo"] == "https://github.com/rhiza-research/weather-skills"
    assert e["dirty"] is True


def test_chain_validation_accepts_commit():
    e = entry(commit="abc123", repo="https://github.com/org/repo", dirty=False)
    violations, notes = provenance.validate_chain([e], "h")
    assert violations == []
    assert notes == []


def test_chain_validation_rejects_empty_commit():
    violations, _ = provenance.validate_chain([entry(commit="")], "h")
    assert any("commit" in v for v in violations)


def test_origin_skill_walks_join():
    history = [
        {
            "skill": "concat",
            "version": "0.1.0",
            "args": {},
            "input": [
                {
                    "basename": "a.zarr",
                    "hash": "aa",
                    "history": [entry(skill="chirps-fetch")],
                },
                {
                    "basename": "b.zarr",
                    "hash": "bb",
                    "history": [entry(skill="dynamical-fetch")],
                },
            ],
        }
    ]
    assert provenance.origin_skill(history) == "chirps-fetch"


def test_normalize_repo_url_github_ssh():
    assert (
        provenance.normalize_repo_url("git@github.com:rhiza-research/weather-skills.git")
        == "https://github.com/rhiza-research/weather-skills"
    )


def test_resolve_skill_revision_from_this_checkout():
    revision = provenance.resolve_skill_revision(Path(__file__))
    assert revision is not None
    assert len(revision["commit"]) >= 7
    # Whatever this checkout's origin is (rhiza-research or weather-skills).
    origin = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert revision["repo"] == provenance.normalize_repo_url(origin)


def test_resolve_skill_revision_marks_dirty(monkeypatch, tmp_path):
    script = tmp_path / "clip.py"
    script.write_text("x = 1\n", encoding="utf-8")

    def fake_git(args, cwd):
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(tmp_path)
        if args[:2] == ["rev-parse", "HEAD"]:
            return "abc123def456"
        if args[:3] == ["config", "--get", "remote.origin.url"]:
            return "git@github.com:example/skills.git"
        if args[0] == "status":
            return " M clip.py"
        return None

    monkeypatch.setattr(provenance, "_git", fake_git)
    revision = provenance.resolve_skill_revision(script)
    assert revision == {
        "commit": "abc123def456",
        "repo": "https://github.com/example/skills",
        "dirty": True,
    }


def test_chain_validation_ok():
    violations, _notes = provenance.validate_chain([entry()], "h")
    assert violations == []


def test_chain_validation_missing_skill():
    bad = entry()
    del bad["skill"]
    violations, _ = provenance.validate_chain([bad], "h")
    assert any("skill" in v for v in violations)


def test_chain_validation_chain_is_intact():
    assert provenance.chain_is_intact([entry()])
    assert not provenance.chain_is_intact([])
    assert not provenance.chain_is_intact(None)
    bad = entry()
    del bad["skill"]
    assert not provenance.chain_is_intact([bad])


def test_visualization_png(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(path)
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain


def test_visualization_html(tmp_path):
    path = tmp_path / "x.html"
    path.write_text("<html><head></head><body></body></html>", encoding="utf-8")
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain
    assert "weather_skills_history" in path.read_text(encoding="utf-8")


def test_visualization_html_non_ascii_payload(tmp_path):
    """A title like 'Rain — week 1' is JSON-escaped to \\u2014 in the payload."""
    path = tmp_path / "x.html"
    path.write_text("<html><head></head><body></body></html>", encoding="utf-8")
    chain = [{**entry(), "args": {"title": "Rain \u2014 week 1", "path": "C:\\data"}}]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain
    # Restamping replaces the existing meta tag through the same path.
    chain2 = [*chain, entry()]
    provenance.stamp_figure(path, chain2)
    assert provenance.load_figure_history(path) == chain2
    assert path.read_text(encoding="utf-8").count('name="weather_skills_history"') == 1


def test_visualization_pdf(tmp_path):
    path = tmp_path / "x.pdf"
    Image.new("RGB", (40, 30), color=(200, 200, 200)).save(path)
    chain = [{**entry(), "args": {"title": "Rain — week 1"}}]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain
    # Restamping replaces the key rather than adding a second one.
    chain2 = [*chain, entry()]
    provenance.stamp_figure(path, chain2)
    assert provenance.load_figure_history(path) == chain2


def test_visualization_svg(tmp_path):
    path = tmp_path / "x.svg"
    path.write_text(
        '<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4">'
        '<rect width="4" height="4"/></svg>',
        encoding="utf-8",
    )
    chain = [{**entry(), "args": {"title": "Rain <b>—</b> & week 1"}}]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain
    chain2 = [*chain, entry()]
    provenance.stamp_figure(path, chain2)
    assert provenance.load_figure_history(path) == chain2
    text = path.read_text(encoding="utf-8")
    assert text.count("<metadata") == 1
    import xml.etree.ElementTree as ET

    ET.fromstring(text.split("?>", 1)[1])  # still well-formed XML


def test_visualization_pdf_official_mark_when_intact(tmp_path):
    from pypdf import PdfReader

    path = tmp_path / "marked.pdf"
    Image.new("RGB", (400, 300), color=(220, 220, 220)).save(path)
    header = PdfReader(path).pdf_header
    provenance.stamp_figure(path, [entry()])
    reader = PdfReader(path)
    # The stamp is drawn over the page in the stamp ink; the PDF version is kept.
    ink = " ".join(f"{c / 255:.3f}" for c in provenance._MARK_INK[:3]).encode()
    assert ink + b" RG" in reader.pages[0].get_contents().get_data()
    assert reader.pdf_header == header

    path = tmp_path / "invalid.pdf"
    Image.new("RGB", (400, 300), color=(220, 220, 220)).save(path)
    provenance.stamp_figure(path, [{"skill": "", "version": "0", "args": {}, "input": None}])
    assert ink not in PdfReader(path).pages[0].get_contents().get_data()


def test_visualization_pdf_restamp_replaces_mark(tmp_path):
    from pypdf import PdfReader

    path = tmp_path / "restamp.pdf"
    Image.new("RGB", (400, 300), color=(220, 220, 220)).save(path)
    for _ in range(3):
        provenance.stamp_figure(path, [entry()])
    data = PdfReader(path).pages[0].get_contents().get_data()
    assert data.count(b"/weather_skills_stamp BMC") == 1
    # A broken chain removes the stamp but keeps the history.
    provenance.stamp_figure(path, [{"skill": "", "version": "0", "args": {}, "input": None}])
    assert b"weather_skills_stamp" not in PdfReader(path).pages[0].get_contents().get_data()
    assert provenance.load_figure_history(path) is not None


def test_visualization_svg_official_mark_when_intact(tmp_path):
    path = tmp_path / "marked.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" '
        'viewBox="0 0 400 300"><rect width="400" height="300"/></svg>',
        encoding="utf-8",
    )
    provenance.stamp_figure(path, [entry()])
    provenance.stamp_figure(path, [entry()])
    text = path.read_text(encoding="utf-8")
    # One stamp after a restamp, drawn last so it sits on top.
    assert text.count('id="weather_skills_stamp"') == 1
    assert text.index('id="weather_skills_stamp"') > text.index("<rect")
    # A broken chain removes the stamp but keeps the history.
    provenance.stamp_figure(path, [{"skill": "", "version": "0", "args": {}, "input": None}])
    text = path.read_text(encoding="utf-8")
    assert "weather_skills_stamp" not in text
    assert provenance.load_figure_history(path) is not None


def test_visualization_svg_official_mark_without_viewbox(tmp_path):
    path = tmp_path / "plain.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400px" height="300">'
        '<rect width="400" height="300"/></svg>',
        encoding="utf-8",
    )
    provenance.stamp_figure(path, [entry()])
    assert path.read_text(encoding="utf-8").count('id="weather_skills_stamp"') == 1


def test_visualization_jpeg(tmp_path):
    path = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(path, quality=90)
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain


def test_visualization_gif_keeps_frames_and_timing(tmp_path):
    path = tmp_path / "x.gif"
    frames = [Image.new("RGB", (400, 300), color=(c, c, c)) for c in (40, 200)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=300, loop=0)
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain
    with Image.open(path) as img:
        assert img.n_frames == 2
        assert img.info["duration"] == 300
        assert img.info["loop"] == 0
        # The seal sits on every frame.
        for i in range(2):
            img.seek(i)
            corner = img.convert("RGB").crop((340, 250, 400, 300)).getcolors(10_000)
            assert len(corner) > 1


def test_visualization_gif_stamp_changes_only_the_mark(tmp_path):
    import numpy as np
    from PIL import ImageChops

    path = tmp_path / "x.gif"
    # A 200-colour ramp (a colorbar) beside 50 colours of each frame's own: nearly
    # the whole palette, so re-picking it would recolour the ramp.
    ramp = np.repeat(np.arange(200, dtype=np.uint8), 2)[:, None]
    frames = []
    for f in range(3):
        rgb = np.full((400, 300, 3), 255, np.uint8)
        rgb[:, :100] = np.stack([ramp, 255 - ramp, ramp // 2], -1)
        own = np.repeat(np.arange(50, dtype=np.uint8), 8)[:, None]
        rgb[:, 100:200] = np.stack([own + 60 * f, own, 255 - own], -1)
        frames.append(Image.fromarray(rgb))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=300, loop=0)
    with Image.open(path) as img:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.convert("RGB"))
    provenance.stamp_figure(path, [entry()])
    cx, cy, r, stroke, _ = provenance._stamp_shape(300, 400)
    with Image.open(path) as img:
        for i, before in enumerate(frames):
            img.seek(i)
            x0, y0, x1, y1 = ImageChops.difference(before, img.convert("RGB")).getbbox()
            # Only pixels under the mark changed.
            assert cx - r - stroke - 1 <= x0 and x1 <= cx + r + stroke + 2
            assert cy - r - stroke - 1 <= y0 and y1 <= cy + r + stroke + 2


def test_visualization_png_official_mark_when_intact(tmp_path):
    path = tmp_path / "marked.png"
    Image.new("RGB", (400, 300), color=(220, 220, 220)).save(path)
    before = list(Image.open(path).crop((320, 250, 400, 300)).get_flattened_data())
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain
    after = list(Image.open(path).crop((320, 250, 400, 300)).get_flattened_data())
    assert before != after


def test_visualization_png_no_mark_when_empty_or_invalid(tmp_path):
    for name, history in (
        ("empty.png", []),
        ("invalid.png", [{"skill": "", "version": "0", "args": {}, "input": None}]),
    ):
        path = tmp_path / name
        Image.new("RGB", (400, 300), color=(220, 220, 220)).save(path)
        before = list(Image.open(path).crop((320, 250, 400, 300)).get_flattened_data())
        provenance.stamp_figure(path, history)
        after = list(Image.open(path).crop((320, 250, 400, 300)).get_flattened_data())
        assert before == after


def test_visualization_png_palette_keeps_distinct_fills(tmp_path):
    """BoM IOD graphs are 8-bit colormaps; stamping must not merge pastel fills."""
    pink = (255, 234, 234)
    blue = (226, 226, 255)
    img = Image.new("P", (400, 300))
    palette = [0] * (256 * 3)
    palette[0:3] = pink
    palette[3:6] = blue
    palette[6:9] = (255, 255, 255)
    img.putpalette(palette)
    pixels = img.load()
    for y in range(300):
        index = 0 if y < 150 else 1
        for x in range(400):
            pixels[x, y] = index
    path = tmp_path / "palette.png"
    img.save(path)

    provenance.stamp_figure(path, [entry()])

    out = Image.open(path).convert("RGB")
    assert out.getpixel((20, 20)) == pink
    assert out.getpixel((20, 280)) == blue


def test_visualization_jpeg_official_mark_when_intact(tmp_path):
    path = tmp_path / "marked.jpg"
    Image.new("RGB", (400, 300), color=(220, 220, 220)).save(path, quality=95)
    before = list(Image.open(path).crop((320, 250, 400, 300)).get_flattened_data())
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    assert provenance.load_figure_history(path) == chain
    after = list(Image.open(path).crop((320, 250, 400, 300)).get_flattened_data())
    assert before != after


def test_geojson_history_is_a_top_level_member(tmp_path):
    path = tmp_path / "region.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    doc = json.loads(path.read_text())
    assert doc["type"] == "FeatureCollection"
    assert json.loads(doc[provenance.HISTORY_ATTR]) == chain
    assert provenance.load_figure_history(path) == chain


def test_json_list_output_is_refused(tmp_path):
    path = tmp_path / "values.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(SkillError, match="not an object"):
        provenance.stamp_figure(path, [entry()])
    assert path.read_text() == "[1, 2, 3]"


@pytest.mark.parametrize("name", ["table.csv", "table.xlsx"])
def test_table_history_lives_in_a_sidecar_tied_to_its_hash(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"time,precip\n2026-01-01,1.0\n")
    chain = [entry()]
    provenance.stamp_figure(path, chain)
    assert path.read_bytes() == b"time,precip\n2026-01-01,1.0\n"
    assert provenance.history_sidecar(path).name == f"{name}.history.json"
    assert provenance.load_figure_history(path) == chain

    # Edited after the skill wrote it: the history no longer describes it.
    path.write_bytes(b"time,precip\n2026-01-01,2.0\n")
    assert provenance.load_figure_history(path) is None
    assert not provenance.can_replace_output(path)


def test_can_replace_output(tmp_path):
    assert provenance.can_replace_output(tmp_path / "missing.png")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert provenance.can_replace_output(empty)

    mine = tmp_path / "notes.csv"
    mine.write_text("a,b\n")
    assert not provenance.can_replace_output(mine)
    provenance.stamp_figure(mine, [entry()])
    assert provenance.can_replace_output(mine)

    not_an_image = tmp_path / "fake.png"
    not_an_image.write_text("hello")
    assert not provenance.can_replace_output(not_an_image)

    bad_geojson = tmp_path / "bad.geojson"
    bad_geojson.write_text("{")
    assert not provenance.can_replace_output(bad_geojson)

    folder = tmp_path / "Documents"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep")
    assert not provenance.can_replace_output(folder)

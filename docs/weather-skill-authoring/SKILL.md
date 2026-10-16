---
name: weather-skill-authoring
description: Playbook for writing a weather skill on the @weather_skill decorator. Covers script layout, declaring flags and Zarr inputs, what to return, and where the dataset, naming, and units contracts live.
---

# weather-skill-authoring

A skill is a directory `skills/<name>/` with a `SKILL.md` and one script in
`scripts/` (e.g. `scripts/fetch.py`, `scripts/clip.py`). Do not name the
script after a stdlib module (`inspect.py`, `json.py`): `uv run` puts that
directory on `sys.path` and shadows the real module.

The `@weather_skill` decorator owns the CLI, opening and checking Zarr
inputs, provenance, and writing outputs. The script body is domain logic
only.

## References

One question per file:

- [`references/CONVENTIONS.md`](references/CONVENTIONS.md) — what should
  this flag be called, and which flags are parsed for me?
- [`references/STANDARD_DATASET.md`](references/STANDARD_DATASET.md) — what
  dims must this Zarr have, and what happens on write?
- [`references/UNITS.md`](references/UNITS.md) — rate or amount, and which
  attrs to stamp?

## Declaration

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core @ git+https://github.com/rhiza-research/weather-skills-core",
# ]
# ///
"""Smooth a gridded Zarr over a lat/lon box and date range."""

from weather_skills_core import Dataset, weather_skill

# Kept in lockstep with SKILL.md metadata.version by CI; do not edit by hand.
_SKILL_VERSION = "0.1.0"


@weather_skill(name="my-skill", version=_SKILL_VERSION)
@weather_skill.argument("-i", "--input", type=Dataset("spatial"), required=True)
@weather_skill.argument("--bbox", required=True)
@weather_skill.argument("--start-time", required=True)
@weather_skill.argument("--end-time", required=True)
@weather_skill.argument("--smoothing", type=int, help="Window width.")
def my_skill(ds, output, bbox, start_time, end_time, smoothing, **kwargs):
    """Smooth a gridded Zarr over a lat/lon box and date range."""
    return ds


if __name__ == "__main__":
    my_skill()
```

`SKILL.md` frontmatter carries the same version (CI bumps both):

```yaml
metadata:
  version: "0.1.0"
```

- `@weather_skill.argument(...)` takes the same kwargs as
  `argparse.ArgumentParser.add_argument`. Stack one per flag.
- The function is called with keyword arguments only. Each flag becomes a
  parameter with hyphens turned into underscores (`--start-time` →
  `start_time`); `--input` arrives as `ds`.
- The function **must** accept `**kwargs`. The decorator may pass keys you
  did not name, and a skill without `**kwargs` refuses to load.
- A handful of flags (`--bbox`, `--date`, `--start-time`, `--end-time`,
  `--variable`) arrive already parsed. Do not re-parse them; see
  [CONVENTIONS.md](references/CONVENTIONS.md#canonical-specials).

## Inputs

Use `type=Dataset(...)` for Zarr inputs. The decorator opens the path, checks
the required dims, attaches pint units, and passes the opened
`xarray.Dataset`. `Dataset("spatial")`, `Dataset("forecast")`, and the rest
of the grammar are in
[STANDARD_DATASET.md](references/STANDARD_DATASET.md#declaring-dataset).

- Several Zarrs of the same role: `action="append"` and repeat `-i`; `ds`
  is then a list.
- Different roles: one flag each (`--forecast`, `--obs`).
- Other files (GeoJSON, a PNG you read): `type=Path`. Those stay paths.

## Outputs

The decorator adds `-o/--output`; do not declare it. One `-o` arrives as
`output: Path`, several as `output: list[Path]`. What you return decides what
happens to that path:

| Return | The decorator |
| --- | --- |
| `xr.Dataset` | stamps provenance and writes the Zarr to `output` |
| `Path` | stamps provenance on that file (you wrote it, e.g. a PNG; it must be the `output` path) |
| a sequence | does one of the above per `-o`; counts must match |
| `None` | writes nothing (the skill already did) |

Skills that only print (inspect, resolve-*) pass
`@weather_skill(..., output=False)` and get no `-o`.

## Skill shapes

| Kind | Inputs | Returns |
| --- | --- | --- |
| Fetcher | none | Dataset |
| Transform | Dataset(s) | Dataset |
| Figure | Dataset(s) | Path (write the figure yourself) |
| Inspect | Dataset or Path; `output=False` | nothing (prints) |

Fetchers also implement `--probe-latest` and list their `-v` tokens in
`metadata.variables` ([CONVENTIONS.md](references/CONVENTIONS.md#fetchers)),
set `weather_skills_source`
([STANDARD_DATASET.md](references/STANDARD_DATASET.md#provenance)), write
precipitation as rates, and stamp native spacing
([UNITS.md](references/UNITS.md)).

## Writing SKILL.md

Agents pick a skill from its `description`, then call it from the
`## Usage` section. Some hosts show only those two parts, so everything
needed to run the skill lives in them.

**`description`** (≤ 800 chars, usually 2–3 sentences): what the skill
does (input → output) and when to use it, written for someone seeing it for
the first time. Agents see every description at once and pick the most
specific match, so do not point to other skills; when two skills look alike,
the more specialised one says what sets it apart. Caveats, flag details, and
pipeline rules go in Usage.

**`## Usage`** (≤ 3500 chars, up to the next `##` heading):

````markdown
# my-skill

One sentence on what it does.

## Usage

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/x.py -i IN.zarr -o OUT.zarr --bbox N/W/S/E [--smoothing N]
```

### Arguments
- `--bbox` — `N/W/S/E` decimal degrees.
- `--smoothing` — window width in steps. Default: 1.

### Output
Dims, variables, units, and the attrs the skill stamps.

### Notes
- Only rules that change what the caller does.

### Example
```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/x.py -i chirps.zarr -o chirps_ken.zarr --bbox 5.5/33.5/-5/42.5
```
One line on what that produces.

## Details

Optional: background a caller rarely needs. Longer material (dataset
catalogs, recipes) goes in `references/*.md`, linked from here.
````

- Describe how the skill works now, in the present tense. No history
  ("now", "no longer", "previously", "fixed").
- Every flag appears once under Arguments with its default; lint checks
  both directions against the script.
- Examples use real values (a real bbox, dates, dataset ids), not
  placeholders.
- Say each thing once: the body does not repeat the description.
- The decorated function's docstring is the `--help` description: one
  plain sentence starting with a verb, saying what the skill does, in line
  with the `description`. The module docstring is the same sentence.
  Implementation notes go in `#` comments.

## Checklist

- Script under `scripts/`, PEP 723 block declaring `weather-skills-core`.
- `_SKILL_VERSION` in the script matches `metadata.version` in `SKILL.md`.
- `SKILL.md` follows [Writing SKILL.md](#writing-skillmd); every flag is under Arguments.
- `uv run weather-skills-core lint skills/<name>` is clean.

"""Tests for region resolution (ISO3, country name, admin-1/admin-2)."""

import sys

import pytest

from weather_skills_core import region as region_mod
from weather_skills_core.errors import DataError, UsageError
from weather_skills_core.region import (
    bbox_from_feature,
    bbox_from_geometry,
    clean_region_name,
    geocode_nominatim,
    level_features,
    lookup_region,
    resolve_region,
    should_geocode,
)

_NAIROBI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "shapeName": "Nairobi",
                "shapeGroup": "KEN",
                "shapeType": "ADM1",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [36.6, -1.4],
                        [37.0, -1.4],
                        [37.0, -1.1],
                        [36.6, -1.1],
                        [36.6, -1.4],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "shapeName": "Elgeyo-Marakwet",
                "shapeGroup": "KEN",
                "shapeType": "ADM1",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[35.0, 0.5], [35.5, 0.5], [35.5, 1.0], [35.0, 1.0], [35.0, 0.5]]],
            },
        },
    ],
}

_WESTLANDS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "shapeName": "Westlands",
                "shapeGroup": "KEN",
                "shapeType": "ADM2",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [36.7, -1.3],
                        [36.9, -1.3],
                        [36.9, -1.2],
                        [36.7, -1.2],
                        [36.7, -1.3],
                    ]
                ],
            },
        }
    ],
}


@pytest.fixture(autouse=True)
def _clear_admin_cache():
    region_mod._admin_collection.cache_clear()
    region_mod._nominatim_collection.cache_clear()
    yield
    region_mod._admin_collection.cache_clear()
    region_mod._nominatim_collection.cache_clear()


@pytest.fixture(autouse=True)
def _offline_country_outlines(monkeypatch):
    """Country lookups use the bundled outline unless a test supplies ADM0."""

    def offline(iso3, level):
        raise region_mod._AdminMissing(f"offline test: no ADM{level} for {iso3}")

    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", offline)


def _fake_admin(iso3, level):
    if level == 0:
        raise region_mod._AdminMissing("offline test: no ADM0")
    if iso3 != "KEN":
        raise AssertionError(f"unexpected fetch {iso3} admin-{level}")
    if level == 1:
        return _NAIROBI
    if level == 2:
        return _WESTLANDS
    raise AssertionError(f"unexpected admin level {level}")


def test_clean_region_name():
    """Aliases, ASCII folding, and spaces-to-underscores land on the lookup key."""
    assert clean_region_name("United States") == "united_states_of_america"
    assert clean_region_name("South Korea") == "south_korea"
    assert clean_region_name("São Tomé") == "sao_tome"
    assert clean_region_name("Côte d'Ivoire") == "ivory_coast"


def test_resolve_kenya_iso3():
    """ISO3 KEN returns Kenya's bbox and a one-row GeoDataFrame."""
    bbox, gdf = resolve_region("KEN")
    n, w, s, e = bbox
    assert n > s and w < e
    assert 4.0 < n < 6.0
    assert 33.0 < w < 35.0
    assert -5.5 < s < -4.0
    assert 41.0 < e < 43.0
    assert list(gdf["iso3"]) == ["KEN"]
    assert list(gdf["name"]) == ["Kenya"]


def test_country_outline_from_geoboundaries(monkeypatch):
    """A country takes the geoBoundaries ADM0 outline when it is available."""
    adm0 = {"type": "FeatureCollection", "features": _NAIROBI["features"]}
    monkeypatch.setattr(
        "weather_skills_core.region._load_admin_geojson",
        lambda iso3, level: adm0 if (iso3, level) == ("KEN", 0) else None,
    )
    feature = lookup_region("Kenya")
    assert feature["properties"]["boundary_source"] == "geoBoundaries gbOpen ADM0"
    assert feature["geometry"]["type"] == "MultiPolygon"
    assert len(feature["geometry"]["coordinates"]) == 2
    assert bbox_from_feature(feature) == (1.0, 35.0, -1.4, 37.0)


def test_country_outline_falls_back_with_warning(monkeypatch, capsys):
    """When geoBoundaries cannot be reached, the bundled outline is used and named."""

    def unreachable(iso3, level):
        raise DataError("geoBoundaries ADM0 metadata for KEN failed: timed out")

    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", unreachable)
    feature = lookup_region("KEN")
    assert feature["properties"]["boundary_source"] == "Natural Earth 1:50m"
    assert "coarse Natural Earth 1:50m outline for Kenya" in capsys.readouterr().err


def test_resolve_by_name_case_insensitive():
    """Country name and ISO3 resolve to the same bbox."""
    bbox_code, _ = resolve_region("KEN")
    bbox_name, gdf = resolve_region("Kenya")
    assert bbox_code == bbox_name
    assert list(gdf["iso3"]) == ["KEN"]


def test_resolve_unknown():
    """An unknown ISO3 is a DataError, not a Nominatim fallthrough."""
    with pytest.raises(DataError, match="not a known"):
        resolve_region("ZZZ")


def test_lookup_empty_is_usage_error():
    """Whitespace-only queries are UsageError."""
    with pytest.raises(UsageError, match="non-empty"):
        lookup_region("  ")


def test_bbox_wraps_antimeridian():
    """A polygon that crosses 180° reports W > E instead of a full-width box."""
    geometry = {
        "type": "Polygon",
        "coordinates": [[[170.0, 0.0], [175.0, 0.0], [-170.0, 0.0], [170.0, 0.0]]],
    }
    _n, w, _s, e = bbox_from_geometry(geometry)
    assert w > e
    assert w == 170.0
    assert e == -170.0


def test_lookup_admin1_name(monkeypatch):
    """country-admin1 resolves Nairobi as admin_1 with its polygon bbox."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-nairobi")
    assert feature["properties"]["level"] == "admin_1"
    assert feature["properties"]["name"] == "Nairobi"
    assert feature["properties"]["iso3"] == "KEN"
    assert feature["properties"]["region_name"] == "kenya-nairobi"

    n, w, s, e = bbox_from_geometry(feature["geometry"])
    assert n == pytest.approx(-1.1)
    assert s == pytest.approx(-1.4)
    assert w == pytest.approx(36.6)
    assert e == pytest.approx(37.0)


def test_lookup_admin1_iso3_prefix(monkeypatch):
    """KEN-nairobi is the same admin_1 lookup as kenya-nairobi."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("KEN-nairobi")
    assert feature["properties"]["region_name"] == "kenya-nairobi"
    assert feature["properties"]["level"] == "admin_1"


def test_lookup_admin2(monkeypatch):
    """country-admin1-admin2 resolves Westlands as admin_2."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-nairobi-westlands")
    assert feature["properties"]["level"] == "admin_2"
    assert feature["properties"]["name"] == "Westlands"
    assert feature["properties"]["region_name"] == "kenya-nairobi-westlands"


def test_lookup_admin2_without_parent(monkeypatch):
    """An admin-2 name is found even without its parent admin-1 in the key."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-westlands")
    assert feature["properties"]["level"] == "admin_2"
    assert feature["properties"]["name"] == "Westlands"


def test_lookup_hyphenated_admin1(monkeypatch):
    """A hyphen inside the admin-1 name (Elgeyo-Marakwet) is not a new hierarchy level."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    feature = lookup_region("kenya-elgeyo-marakwet")
    assert feature["properties"]["level"] == "admin_1"
    assert feature["properties"]["name"] == "Elgeyo-Marakwet"


def test_lookup_unknown_admin_unit(monkeypatch):
    """A country-prefixed key that matches no admin unit is a DataError."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin)

    with pytest.raises(DataError, match="admin-1 or admin-2"):
        lookup_region("kenya-not-a-county")


def test_hyphenated_country_is_not_split_into_admin():
    """Guinea-Bissau is a country, not country 'Guinea' plus admin 'Bissau'."""
    feature = lookup_region("Guinea-Bissau")
    assert feature["properties"]["level"] == "country"
    assert feature["properties"]["iso3"] == "GNB"


_MOUNT_KENYA_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [37.2, -0.25],
            [37.4, -0.25],
            [37.4, -0.05],
            [37.2, -0.05],
            [37.2, -0.25],
        ]
    ],
}


def _mount_kenya_hit(*, geojson=None):
    return [
        {
            "display_name": "Mount Kenya, Kenya",
            "name": "Mount Kenya",
            "category": "natural",
            "type": "volcano",
            "boundingbox": ["-0.25", "-0.05", "37.2", "37.4"],
            "lat": "-0.15",
            "lon": "37.3",
            "geojson": geojson,
        }
    ]


def test_should_geocode_landmarks_not_admin_keys():
    """Landmarks may hit Nominatim; ISO3, admin keys, NE regions, and custom boxes must not."""
    assert should_geocode("Mount Kenya") is True
    assert should_geocode("Mount Kenya, Kenya") is True
    assert should_geocode("kenya-nairbi") is False
    assert should_geocode("KEN-nairbi") is False
    assert should_geocode("ZZZ") is False
    assert should_geocode("  ") is False
    assert should_geocode("East Africa") is False
    assert should_geocode("Eastern Africa") is False
    assert should_geocode("Western Africa") is False
    assert should_geocode("Sub-Saharan Africa") is False
    assert should_geocode("Indian Ocean") is False
    assert should_geocode("Indian Ocean basin") is False
    assert should_geocode("IOB") is False


def test_lookup_ne_eastern_africa_not_nominatim(monkeypatch):
    """East Africa / Eastern Africa dissolve bundled countries, never Nominatim."""

    def _fail_nominatim(query):
        raise AssertionError(f"Nominatim should not run for named region; got {query!r}")

    monkeypatch.setattr("weather_skills_core.region._load_nominatim", _fail_nominatim)

    kenya = lookup_region("KEN")
    kn, kw, ks, ke = bbox_from_feature(kenya)
    east = lookup_region("East Africa")
    eastern = lookup_region("Eastern Africa")
    # "East Africa" is the directional alias of the UN Eastern Africa grouping.
    assert east == eastern
    assert east["properties"]["name"] == "Eastern Africa"
    assert east["properties"]["level"] == "region"
    assert east["geometry"]["type"] == "MultiPolygon"
    n, w, s, e = bbox_from_feature(east)
    assert s <= ks <= kn <= n
    assert w <= kw <= ke <= e
    assert n > 10
    assert s < -20  # keeps Madagascar


def test_lookup_indian_ocean_basin_not_nominatim(monkeypatch):
    """Indian Ocean is the conventional basin box, never a Nominatim centroid."""

    def _fail_nominatim(query):
        raise AssertionError(f"Nominatim should not run for Indian Ocean; got {query!r}")

    monkeypatch.setattr("weather_skills_core.region._load_nominatim", _fail_nominatim)

    feature = lookup_region("Indian Ocean basin")
    props = feature["properties"]
    assert props["name"] == "Indian Ocean basin"
    assert props["level"] == "custom"
    assert props["iso3"] is None
    n, w, s, e = bbox_from_feature(feature)
    assert (n, w, s, e) == (30.0, 20.0, -40.0, 120.0)
    assert feature["geometry"]["type"] == "Polygon"

    aliases = (
        lookup_region("Indian Ocean"),
        lookup_region("Indian Ocean Basin"),
        lookup_region("Indian Ocean basin region"),
        lookup_region("IOB"),
    )
    for other in aliases:
        assert bbox_from_feature(other) == (n, w, s, e)
        assert other["properties"]["name"] == "Indian Ocean basin"


def test_custom_regions_file_is_well_formed():
    """Every bundled custom region has a polygon, a source, and names no other lookup takes."""
    import json
    from importlib.resources import files

    payload = json.loads(
        files("weather_skills_core.data").joinpath("custom_regions.geojson").read_text()
    )
    _, by_clean = region_mod._country_indexes()
    regions = region_mod._region_indexes()
    seen = set()
    for feature in payload["features"]:
        props = feature["properties"]
        assert props["name"] and props["source"]
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
        bbox_from_geometry(feature["geometry"])
        for alias in (props["name"], *props.get("aliases", ())):
            cleaned = clean_region_name(alias)
            assert cleaned not in by_clean, f"{alias!r} is a country name"
            assert cleaned not in regions, f"{alias!r} is a Natural Earth region"
            assert cleaned not in seen, f"{alias!r} is listed twice"
            seen.add(cleaned)


def test_south_africa_is_the_country_not_the_subregion():
    """South Africa matches the country, not a Natural Earth subregion of that name."""
    feature = lookup_region("South Africa")
    assert feature["properties"]["level"] == "country"
    assert feature["properties"]["iso3"] == "ZAF"


def test_lookup_ne_western_africa():
    """West Africa is the UN Western Africa grouping and contains Ghana."""
    ghana = lookup_region("GHA")
    gn, gw, gs, ge = bbox_from_feature(ghana)
    west = lookup_region("West Africa")
    assert west["properties"]["name"] == "Western Africa"
    n, w, s, e = bbox_from_feature(west)
    assert s <= gs <= gn <= n
    assert w <= gw <= ge <= e


def test_geocode_nominatim_polygon(monkeypatch):
    """A Nominatim polygon hit is stored as geometry with bbox on properties."""
    monkeypatch.setattr(
        "weather_skills_core.region._load_nominatim",
        lambda query: _mount_kenya_hit(geojson=_MOUNT_KENYA_POLYGON),
    )

    feature = geocode_nominatim("Mount Kenya")
    props = feature["properties"]
    assert props["level"] == "nominatim"
    assert props["name"] == "Mount Kenya"
    assert props["display_name"] == "Mount Kenya, Kenya"
    assert (props["osm_category"], props["osm_type"]) == ("natural", "volcano")
    assert props["bbox"] == pytest.approx([-0.05, 37.2, -0.25, 37.4])
    n, w, s, e = bbox_from_feature(feature)
    assert (n, w, s, e) == pytest.approx((-0.05, 37.2, -0.25, 37.4))
    assert feature["geometry"]["type"] == "Polygon"


def test_geocode_nominatim_point_uses_bbox_rectangle(monkeypatch):
    """A Nominatim Point is replaced by a rectangle from the boundingbox."""
    monkeypatch.setattr(
        "weather_skills_core.region._load_nominatim",
        lambda query: _mount_kenya_hit(geojson={"type": "Point", "coordinates": [37.3, -0.15]}),
    )

    feature = geocode_nominatim("Mount Kenya")
    assert feature["geometry"]["type"] == "Polygon"
    n, w, s, e = bbox_from_feature(feature)
    assert (n, w, s, e) == pytest.approx((-0.05, 37.2, -0.25, 37.4))


def test_geocode_nominatim_empty(monkeypatch):
    """Zero Nominatim hits is a DataError."""
    monkeypatch.setattr("weather_skills_core.region._load_nominatim", lambda query: [])

    with pytest.raises(DataError, match="Nominatim found no matching place"):
        geocode_nominatim("Not A Real Volcano 12345")


def test_bbox_from_feature_prefers_stored_bbox():
    """properties.bbox wins over deriving a box from a Point geometry."""
    feature = {
        "type": "Feature",
        "properties": {"bbox": [1.0, 2.0, -1.0, 4.0]},
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }
    assert bbox_from_feature(feature) == (1.0, 2.0, -1.0, 4.0)


def test_http_json_read_timeout_is_data_error(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(region_mod.urllib.request, "urlopen", timeout)
    with pytest.raises(DataError, match="timed out"):
        region_mod._http_json("https://example.invalid/x.json", what="test fetch")


def _fake_admin_kenya_only(iso3, level):
    if iso3 != "KEN":
        raise region_mod._AdminMissing(f"offline test: no ADM{level} for {iso3}")
    return _fake_admin(iso3, level)


def test_level_features_admin1_keys(monkeypatch):
    """admin_1 units carry lookup_region keys; countries without a layer are skipped."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin_kenya_only)

    features = level_features("admin_1", (2.0, 34.0, -2.0, 38.0))
    keys = [f["properties"]["region_name"] for f in features]
    assert keys == ["kenya-nairobi", "kenya-elgeyo-marakwet"]
    assert lookup_region(keys[0])["properties"]["name"] == "Nairobi"


def test_level_features_bbox_filters_units(monkeypatch):
    """Units outside the bbox are left out."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_admin_kenya_only)

    features = level_features("admin_1", (-1.0, 36.5, -1.5, 37.5))
    assert [f["properties"]["region_name"] for f in features] == ["kenya-nairobi"]


def test_level_features_duplicate_admin_names(monkeypatch):
    """A repeated admin name gets a numbered key so each key is one polygon."""
    twice = {"type": "FeatureCollection", "features": _NAIROBI["features"][:1] * 2}

    def fake(iso3, level):
        if iso3 != "KEN" or level != 1:
            raise region_mod._AdminMissing("offline test")
        return twice

    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", fake)
    features = level_features("admin_1", (-1.0, 36.5, -1.5, 37.5))
    assert [f["properties"]["region_name"] for f in features] == [
        "kenya-nairobi",
        "kenya-nairobi_2",
    ]


def test_level_features_duplicate_keys_do_not_depend_on_bbox(monkeypatch):
    """A namesake outside the bbox still counts, so each polygon keeps one key."""

    def square(west, south):
        ring = [[west, south], [west + 1, south], [west + 1, south + 1], [west, south + 1]]
        return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}

    central = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"shapeName": "Central"}, "geometry": square(35, 1)},
            {"type": "Feature", "properties": {"shapeName": "Central"}, "geometry": square(38, -3)},
        ],
    }

    def fake(iso3, level):
        if iso3 != "KEN" or level != 1:
            raise region_mod._AdminMissing("offline test")
        return central

    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", fake)
    wide = level_features("admin_1", (2.5, 34.0, -3.5, 39.5))
    south_only = level_features("admin_1", (-2.0, 37.5, -3.5, 39.5))
    assert [f["properties"]["region_name"] for f in wide] == ["kenya-central", "kenya-central_2"]
    assert [f["properties"]["region_name"] for f in south_only] == ["kenya-central_2"]


def test_level_features_country_and_subregion():
    """Countries and subregions overlapping East Africa come from bundled data."""
    bbox = (5.0, 34.0, -4.0, 41.0)
    countries = {f["properties"]["iso3"] for f in level_features("country", bbox)}
    assert {"KEN", "UGA", "TZA", "ETH", "SOM"} <= countries
    assert "NGA" not in countries
    subregions = [f["properties"]["region_name"] for f in level_features("subregion", bbox)]
    assert subregions == ["eastern_africa"]
    continents = [f["properties"]["name"] for f in level_features("continent", bbox)]
    assert continents == ["Africa"]


def test_level_features_antimeridian_bbox():
    """A bbox crossing 180 (west > east) finds Fiji."""
    countries = {
        f["properties"]["iso3"] for f in level_features("country", (-15.0, 175.0, -20.0, -178.0))
    }
    assert "FJI" in countries


def test_level_features_unknown_level():
    with pytest.raises(UsageError, match="admin_1"):
        level_features("county")


def _square(west, south, size=1.0):
    ring = [[west, south], [west + size, south], [west + size, south + size], [west, south + size]]
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}


def _unit(name, geometry):
    return {"type": "Feature", "properties": {"shapeName": name}, "geometry": geometry}


# Two admin_1 units, each holding an admin_2 "Central"; admin_2 "Kiambu" shares its
# admin_1's name.
_TWO_COUNTIES = {
    1: [_unit("Kiambu", _square(36.0, -1.5, 2.0)), _unit("Meru", _square(38.0, -1.5, 2.0))],
    2: [
        _unit("Central", _square(36.2, -1.3)),
        _unit("Kiambu", _square(36.8, -0.3)),
        _unit("Central", _square(38.5, -1.0)),
    ],
}


def _fake_two_counties(iso3, level):
    if iso3 != "KEN" or level not in _TWO_COUNTIES:
        raise region_mod._AdminMissing("offline test")
    return {"type": "FeatureCollection", "features": _TWO_COUNTIES[level]}


def test_level_features_admin2_keys_name_their_admin1(monkeypatch):
    """admin_2 keys are country-admin1-admin2 and lookup_region resolves each to its unit."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_two_counties)
    features = level_features("admin_2", (1.0, 35.5, -2.0, 40.0))
    keys = [f["properties"]["region_name"] for f in features]
    assert keys == ["kenya-kiambu-central", "kenya-kiambu-kiambu", "kenya-meru-central"]
    for feature in features:
        found = lookup_region(feature["properties"]["region_name"])
        assert found["properties"]["level"] == "admin_2"
        assert found["geometry"] == feature["geometry"]


def test_level_features_admin2_needs_shapely(monkeypatch):
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _fake_two_counties)
    monkeypatch.setitem(sys.modules, "shapely", None)
    with pytest.raises(UsageError, match=r"weather-skills-core\[geo\]"):
        level_features("admin_2", (1.0, 35.5, -2.0, 40.0))


def test_bundled_countries_include_small_states():
    """The 1:50m country file holds island states the 1:110m file left out."""
    by_iso3, _ = region_mod._country_indexes()
    assert {"MUS", "SYC", "COM", "CPV", "STP", "MDV"} <= set(by_iso3)


def _no_adm0(iso3, level):
    raise region_mod._AdminMissing("offline test")


def test_level_features_overlap_is_per_polygon(monkeypatch):
    """France's outline holds Mayotte: France is in a Mayotte box, not in a Kenya box."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _no_adm0)
    kenya = {f["properties"]["iso3"] for f in level_features("country", (5.0, 34.0, -4.0, 41.0))}
    mayotte = {
        f["properties"]["iso3"] for f in level_features("country", (-12.6, 44.9, -13.1, 45.4))
    }
    assert "FRA" not in kenya
    assert "FRA" in mayotte


def test_level_features_subregion_from_country_outlines(monkeypatch):
    """A subregion is its member countries in the bbox, labelled with the real level."""
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _no_adm0)
    (feature,) = level_features("subregion", (-1.0, 36.5, -1.5, 37.5))
    assert feature["properties"]["region_name"] == "eastern_africa"
    assert feature["properties"]["level"] == "subregion"
    # Kenya's parts, and only parts of countries whose own parts overlap the bbox.
    by_iso3 = region_mod._country_indexes()[0]
    kenya = region_mod._polygon_parts(by_iso3["KEN"]["geometry"])
    ethiopia = region_mod._polygon_parts(by_iso3["ETH"]["geometry"])
    parts = region_mod._polygon_parts(feature["geometry"])
    assert all(part in parts for part in kenya)
    assert not any(part in parts for part in ethiopia)


def test_level_features_warns_on_many_countries(monkeypatch, capsys):
    monkeypatch.setattr("weather_skills_core.region._load_admin_geojson", _no_adm0)
    monkeypatch.setattr("weather_skills_core.region._MANY_COUNTRIES", 2)
    level_features("country", (5.0, 34.0, -4.0, 41.0))
    assert "fetching geoBoundaries outlines for" in capsys.readouterr().err

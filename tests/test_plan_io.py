import pytest

from race_planners.models import PacingConfig
from race_planners.plan_io import ensure_gpx_exists_for_plan, export_plan_json, import_plan_json


def test_plan_json_roundtrip() -> None:
    config = PacingConfig(race_model="technical_trail_ultra", input_mode="effort_anchor")
    exported = export_plan_json(
        course_id="sample-course",
        gpx_filename="sample.gpx",
        config=config,
    )
    payload = import_plan_json(exported)
    assert payload["course_id"] == "sample-course"
    assert payload["gpx_filename"] == "sample.gpx"
    assert payload["config"]["race_model"] == "technical_trail_ultra"


def test_missing_gpx_error_message() -> None:
    payload = {
        "schema_version": 1,
        "course_id": "missing-course",
        "gpx_filename": "ABC123.gpx",
        "config": {"race_model": "fire_road_ultra", "input_mode": "effort_anchor"},
    }

    with pytest.raises(FileNotFoundError, match="Missing course file: ABC123.gpx"):
        ensure_gpx_exists_for_plan(payload)


def test_plan_import_resolves_legacy_curated_gpx_basename() -> None:
    payload = {
        "schema_version": 1,
        "course_id": "semi-marathon-finistere",
        "gpx_filename": "semi-marathon-du-finistere.gpx",
        "config": {"race_model": "half_marathon", "input_mode": "finish_time"},
    }

    gpx_path = ensure_gpx_exists_for_plan(payload)

    assert gpx_path.name == payload["gpx_filename"]
    assert gpx_path.is_file()

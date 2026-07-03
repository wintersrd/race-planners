from pathlib import Path

from race_planners.event_catalog import get_curated_event
from race_planners.models import PacingConfig
from race_planners.plan_io import export_plan_json
from race_planners.streamlit_general import (
    _course_overview_rows,
    _default_config_for_event,
    load_plan_into_state,
)


def test_load_plan_into_state_returns_user_facing_missing_gpx_error(tmp_path: Path) -> None:
    payload_json = export_plan_json(
        course_id="library:missing-course",
        gpx_filename="missing-course.gpx",
        config=PacingConfig(race_model="road_marathon", input_mode="finish_time"),
    )

    state, error = load_plan_into_state(
        payload_json, tmp_path, {"general_config": {}, "general_course_id": "x"}
    )

    assert error is not None
    assert "Missing course file: missing-course.gpx" in error
    assert state["general_course_id"] == "x"


def test_load_plan_into_state_restores_course_and_config(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir(parents=True)
    (course_dir / "sample.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="sample.gpx",
        config=PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.8,
            hike_pace_min_km=13.2,
        ),
    )

    state, error = load_plan_into_state(
        payload_json, tmp_path, {"general_config": {}, "general_course_id": "x"}
    )

    assert error is None
    assert state["general_course_id"] == "semi-marathon-finistere"
    assert state["general_config"]["race_model"] == "technical_trail_ultra"


def test_load_plan_into_state_restores_curated_event_id(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir(parents=True)
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="semi-marathon-du-finistere.gpx",
        config=PacingConfig(race_model="half_marathon", input_mode="finish_time"),
    )

    state, error = load_plan_into_state(
        payload_json,
        tmp_path,
        {"general_config": {}, "general_course_id": "x", "general_event_id": "y"},
    )

    assert error is None
    assert state["general_event_id"] == "semi-marathon-finistere"


def test_load_plan_into_state_restores_athlete_profile(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir(parents=True)
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="semi-marathon-du-finistere.gpx",
        config=PacingConfig(race_model="half_marathon", input_mode="finish_time"),
        athlete_profile={"lt1_hr": 152, "lt1_pace_min_km": 5.0},
    )

    state, error = load_plan_into_state(
        payload_json,
        tmp_path,
        {"general_config": {}, "general_course_id": "x", "general_athlete_profile": {}},
    )

    assert error is None
    assert state["general_athlete_profile"]["lt1_hr"] == 152
    assert state["general_athlete_profile"]["lt1_pace_min_km"] == 5.0


def test_course_overview_rows_reflect_event_metadata(tmp_path: Path) -> None:
    (tmp_path / "semi-marathon-finistere").mkdir(parents=True)
    (tmp_path / "semi-marathon-finistere" / "semi-marathon-du-finistere.gpx").write_text(
        "<gpx></gpx>", encoding="utf-8"
    )
    event = get_curated_event(tmp_path, "semi-marathon-finistere")

    rows = _course_overview_rows(21.06, event)

    assert rows[0] == {"label": "Distance", "value": "21.06 km"}
    assert any(row == {"label": "Terrain", "value": "Road"} for row in rows)
    assert any(row == {"label": "Aid Stations", "value": "3 configured"} for row in rows)
    assert all(row["label"] != "Race Model" for row in rows)


def test_default_config_for_trail_event_uses_athlete_profile_defaults(tmp_path: Path) -> None:
    (tmp_path / "semi-marathon-finistere").mkdir(parents=True)
    (tmp_path / "semi-marathon-finistere" / "2026-grf92.gpx").write_text(
        "<gpx></gpx>", encoding="utf-8"
    )
    event = get_curated_event(tmp_path, "grf92")

    config = _default_config_for_event(
        event,
        {
            "lt1_pace_min_km": 5.0,
            "technical_trail_slowdown_sec_km": 45.0,
        },
    )

    assert config["fade_profile_preset"] == "progressive_fade"
    assert config["flat_pace_min_km"] == 5.75
    assert config["hike_pace_min_km"] == 9.75

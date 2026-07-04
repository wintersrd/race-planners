from pathlib import Path

from race_planners.event_catalog import get_curated_event
from race_planners.models import AthleteProfile, PacingConfig, PaceSplit, TrackPoint
from race_planners.plan_io import export_plan_json
from race_planners.streamlit_general import (
    _aggregate_split_rows,
    _course_overview_rows,
    _default_config_for_event,
    _derived_hr_guardrail_cap,
    _format_clock_time,
    _format_duration_minutes,
    _format_pace_minutes,
    _modeled_road_best_likely_time_min,
    _road_capability_sources,
    _selected_road_capability,
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


def test_load_plan_into_state_restores_expanded_athlete_profile_fields(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir(parents=True)
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="semi-marathon-du-finistere.gpx",
        config=PacingConfig(race_model="half_marathon", input_mode="finish_time"),
        athlete_profile={
            "best_likely_half_time_min": 86.0,
            "best_likely_marathon_time_min": 195.0,
            "predictor_half_time_min": 88.0,
            "predictor_marathon_time_min": 198.0,
            "predictor_source": "COROS",
            "durability_factor": 0.4,
            "heat_tolerance": -0.2,
            "hill_tolerance": 0.1,
        },
    )

    state, error = load_plan_into_state(
        payload_json,
        tmp_path,
        {"general_config": {}, "general_course_id": "x", "general_athlete_profile": {}},
    )

    assert error is None
    assert state["general_athlete_profile"]["best_likely_half_time_min"] == 86.0
    assert state["general_athlete_profile"]["best_likely_marathon_time_min"] == 195.0
    assert state["general_athlete_profile"]["predictor_half_time_min"] == 88.0
    assert state["general_athlete_profile"]["predictor_marathon_time_min"] == 198.0
    assert state["general_athlete_profile"]["predictor_source"] == "COROS"
    assert state["general_athlete_profile"]["durability_factor"] == 0.4
    assert state["general_athlete_profile"]["heat_tolerance"] == -0.2
    assert state["general_athlete_profile"]["hill_tolerance"] == 0.1


def test_athlete_profile_dataclass_round_trips_new_fields() -> None:
    profile = AthleteProfile(
        best_likely_half_time_min=86.0,
        best_likely_marathon_time_min=195.0,
        predictor_half_time_min=88.0,
        predictor_marathon_time_min=198.0,
        predictor_source="COROS",
        durability_factor=0.3,
        heat_tolerance=-0.1,
        hill_tolerance=0.2,
    )

    assert profile.best_likely_half_time_min == 86.0
    assert profile.best_likely_marathon_time_min == 195.0
    assert profile.predictor_half_time_min == 88.0
    assert profile.predictor_marathon_time_min == 198.0
    assert profile.predictor_source == "COROS"
    assert profile.durability_factor == 0.3
    assert profile.heat_tolerance == -0.1
    assert profile.hill_tolerance == 0.2


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
    assert any(row == {"label": "Start Time", "value": "Unknown"} for row in rows)
    assert all(row["label"] != "Race Model" for row in rows)


def test_time_format_helpers_render_human_readable_values(tmp_path: Path) -> None:
    (tmp_path / "semi-marathon-finistere").mkdir(parents=True)
    (tmp_path / "semi-marathon-finistere" / "2026-grf92.gpx").write_text(
        "<gpx></gpx>", encoding="utf-8"
    )
    event = get_curated_event(tmp_path, "grf92")

    assert _format_pace_minutes(4.5) == "4:30"
    assert _format_duration_minutes(88.5) == "1:28:30"
    assert _format_duration_minutes(4.5) == "4:30"
    assert _format_clock_time(event, 88.5) == "7:58 AM"


def test_aggregate_split_rows_supports_multi_kilometer_blocks(tmp_path: Path) -> None:
    (tmp_path / "semi-marathon-finistere").mkdir(parents=True)
    (tmp_path / "semi-marathon-finistere" / "2026-grf92.gpx").write_text(
        "<gpx></gpx>", encoding="utf-8"
    )
    event = get_curated_event(tmp_path, "grf92")
    rows = _aggregate_split_rows(
        splits=[
            PaceSplit(1.0, 5.0, 2.0, 5.0, 5.0),
            PaceSplit(2.0, 7.0, 4.0, 7.0, 12.0),
            PaceSplit(3.0, 6.0, -1.0, 6.0, 18.0),
        ],
        trackpoints=[
            TrackPoint(48.0, -4.0, 0.0, "", 0.0),
            TrackPoint(48.0, -3.99, 10.0, "", 1000.0),
            TrackPoint(48.0, -3.98, 25.0, "", 2000.0),
            TrackPoint(48.0, -3.97, 20.0, "", 3000.0),
        ],
        event=event,
        block_size_km=2,
    )

    assert len(rows) == 2
    assert rows[0]["split"] == "0.0-2.0 km"
    assert rows[0]["pace"] == "6:00"
    assert rows[0]["elev_gain_m"] == 25.0
    assert rows[1]["split"] == "2.0-3.0 km"


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
            "default_trail_fade_preset": "late_fade",
            "default_trail_effort_policy": "conservative",
            "lt1_hr": 152,
            "lt2_hr": 170,
        },
    )

    assert config["fade_profile_preset"] == "late_fade"
    assert config["effort_policy"] == "conservative"
    assert config["use_hr_guardrail"] is True
    assert config["flat_pace_min_km"] == 5.75
    assert config["hike_pace_min_km"] == 9.75


def test_default_config_for_road_event_uses_profile_split_bias(tmp_path: Path) -> None:
    (tmp_path / "semi-marathon-finistere").mkdir(parents=True)
    (tmp_path / "semi-marathon-finistere" / "semi-marathon-du-finistere.gpx").write_text(
        "<gpx></gpx>", encoding="utf-8"
    )
    event = get_curated_event(tmp_path, "semi-marathon-finistere")

    config = _default_config_for_event(
        event,
        {
            "lt1_pace_min_km": 5.0,
            "lt2_pace_min_km": 4.25,
            "default_road_split_bias": -2.5,
        },
    )

    assert config["marathon_pace_min_km"] == 4.78
    assert config["pacing_bias"] == -2.5


def test_selected_road_capability_prefers_manual_then_predictor_then_model() -> None:
    profile = {
        "best_likely_marathon_time_min": 205.0,
        "predictor_marathon_time_min": 210.0,
        "predictor_source": "COROS",
        "lt1_pace_min_km": 5.0,
    }

    selected_time_min, selected_source = _selected_road_capability(profile, "road_marathon")

    assert selected_time_min == 205.0
    assert selected_source == "Manual Profile"

    del profile["best_likely_marathon_time_min"]
    selected_time_min, selected_source = _selected_road_capability(profile, "road_marathon")

    assert selected_time_min == 210.0
    assert selected_source == "COROS"

    del profile["predictor_marathon_time_min"]
    selected_time_min, selected_source = _selected_road_capability(profile, "road_marathon")

    assert selected_time_min is not None
    assert selected_source == "LT-Derived Model"


def test_modeled_road_best_likely_time_uses_current_profile_heuristic() -> None:
    profile = {
        "lt1_pace_min_km": 5.0,
        "lt2_pace_min_km": 4.25,
    }

    half_time_min = _modeled_road_best_likely_time_min(profile, "half_marathon")
    marathon_time_min = _modeled_road_best_likely_time_min(profile, "road_marathon")

    assert half_time_min == 100.85
    assert marathon_time_min == 210.97


def test_road_capability_sources_marks_selected_source() -> None:
    sources = _road_capability_sources(
        {
            "predictor_half_time_min": 92.0,
            "predictor_source": "Strava",
            "lt1_pace_min_km": 5.0,
            "lt2_pace_min_km": 4.25,
        },
        "half_marathon",
    )

    assert [row["source"] for row in sources] == [
        "Manual Profile",
        "Strava",
        "LT-Derived Model",
    ]
    assert sources[1]["selected"] is True
    assert sources[2]["selected"] is False


def test_default_config_for_road_event_uses_selected_capability_time(tmp_path: Path) -> None:
    (tmp_path / "semi-marathon-finistere").mkdir(parents=True)
    (tmp_path / "semi-marathon-finistere" / "marathon-des-etoiles-de-la-baie.gpx").write_text(
        "<gpx></gpx>", encoding="utf-8"
    )
    event = get_curated_event(tmp_path, "marathon-etoiles-baie")

    config = _default_config_for_event(
        event,
        {
            "best_likely_marathon_time_min": 198.0,
            "lt1_pace_min_km": 5.0,
        },
    )

    assert config["target_finish_time_min"] == 198.0


def test_derived_hr_guardrail_cap_uses_profile_and_policy() -> None:
    guardrail = _derived_hr_guardrail_cap(
        {"lt1_hr": 152, "lt2_hr": 170},
        "technical_trail_ultra",
        "conservative",
    )

    assert guardrail == 153


def test_derived_hr_guardrail_cap_varies_by_event_type() -> None:
    profile = {"lt1_hr": 152, "lt2_hr": 170}

    technical = _derived_hr_guardrail_cap(profile, "technical_trail_ultra", "steady")
    fire_road = _derived_hr_guardrail_cap(profile, "fire_road_ultra", "steady")
    road = _derived_hr_guardrail_cap(profile, "road_marathon", "steady")

    assert technical is not None
    assert fire_road is not None
    assert road is not None
    assert technical < fire_road < road

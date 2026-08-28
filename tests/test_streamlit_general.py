from race_planners.event_catalog import get_curated_event
from race_planners.formatting import format_clock_time, format_duration_minutes, format_pace_minutes
from race_planners.i18n import t
from race_planners.models import AthleteProfile, PacingConfig, PaceSplit, TrackPoint
from race_planners.plan_io import export_plan_json, load_plan_into_state
from race_planners.profile import (
    default_config_for_event,
    derived_hr_guardrail_cap,
    modeled_road_best_likely_time_min,
    road_capability_sources,
    selected_road_capability,
)
from race_planners.splits import aggregate_split_rows, course_overview_rows


def test_load_plan_into_state_returns_user_facing_missing_gpx_error() -> None:
    payload_json = export_plan_json(
        course_id="missing-course",
        gpx_filename="missing-course.gpx",
        config=PacingConfig(race_model="road_marathon", input_mode="finish_time"),
    )

    state, error = load_plan_into_state(
        payload_json, {"general_config": {}, "general_course_id": "x"}
    )

    assert error is not None
    assert "Missing course file: missing-course.gpx" in error
    assert state["general_course_id"] == "x"


def test_load_plan_into_state_rejects_mismatched_curated_gpx() -> None:
    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="mismatched.gpx",
        config=PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.8,
            hike_pace_min_km=13.2,
        ),
    )

    state, error = load_plan_into_state(
        payload_json, {"general_config": {}, "general_course_id": "x"}
    )

    assert error is not None
    assert state["general_course_id"] == "x"


def test_load_plan_into_state_restores_curated_event_id() -> None:
    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="semi-marathon-du-finistere.gpx",
        config=PacingConfig(race_model="half_marathon", input_mode="finish_time"),
    )

    state, error = load_plan_into_state(
        payload_json,
        {"general_config": {}, "general_course_id": "x", "general_event_id": "y"},
    )

    assert error is None
    assert state["general_event_id"] == "semi-marathon-finistere"


def test_load_plan_into_state_restores_athlete_profile() -> None:
    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="semi-marathon-du-finistere.gpx",
        config=PacingConfig(race_model="half_marathon", input_mode="finish_time"),
        athlete_profile={"lt1_hr": 152, "lt1_pace_min_km": 5.0},
    )

    state, error = load_plan_into_state(
        payload_json,
        {"general_config": {}, "general_course_id": "x", "general_athlete_profile": {}},
    )

    assert error is None
    assert state["general_athlete_profile"]["lt1_hr"] == 152
    assert state["general_athlete_profile"]["lt1_pace_min_km"] == 5.0


def test_load_plan_into_state_restores_expanded_athlete_profile_fields() -> None:
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


def testcourse_overview_rows_reflect_event_metadata() -> None:
    event = get_curated_event("semi-marathon-finistere")

    rows = course_overview_rows(21.06, event)

    assert rows[0] == {"label": "Distance", "value": "21.06 km"}
    assert any(row == {"label": "Terrain", "value": "Road"} for row in rows)
    assert any(row == {"label": "Aid Stations", "value": "3 configured"} for row in rows)
    assert any(row == {"label": "Start Time", "value": "Unknown"} for row in rows)
    assert all(row["label"] != "Race Model" for row in rows)


def test_time_format_helpers_render_human_readable_values() -> None:
    event = get_curated_event("grf92")

    assert format_pace_minutes(4.5) == "4:30"
    assert format_duration_minutes(88.5) == "1:28:30"
    assert format_duration_minutes(4.5) == "4:30"
    assert format_clock_time(event, 88.5) == "7:58 AM"
    assert format_clock_time(event, 88.5, "fr") == "7:58"
    assert format_clock_time(event, 0.0, "fr") == "6:30"
    assert format_clock_time(event, 0.0, "en") == "6:30 AM"


def testaggregate_split_rows_supports_multi_kilometer_blocks() -> None:
    event = get_curated_event("grf92")
    rows = aggregate_split_rows(
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
    assert rows[0][t("col.split_range")] == "0.0-2.0 km"
    assert rows[0][t("col.pace")] == "6:00"
    assert rows[0][t("col.elev_gain_m")] == 25.0
    assert rows[1][t("col.split_range")] == "2.0-3.0 km"


def test_default_config_for_trail_event_uses_athlete_profile_defaults() -> None:
    event = get_curated_event("grf92")

    config = default_config_for_event(
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


def test_default_config_for_road_event_uses_profile_split_bias() -> None:
    event = get_curated_event("semi-marathon-finistere")

    config = default_config_for_event(
        event,
        {
            "lt1_pace_min_km": 5.0,
            "lt2_pace_min_km": 4.25,
            "default_road_split_bias": -2.5,
        },
    )

    assert config["marathon_pace_min_km"] is not None
    assert 4.25 <= config["marathon_pace_min_km"] <= 5.0
    assert config["pacing_bias"] == -2.5


def testselected_road_capability_prefers_manual_then_predictor_then_model() -> None:
    profile = {
        "best_likely_marathon_time_min": 205.0,
        "predictor_marathon_time_min": 210.0,
        "predictor_source": "COROS",
        "lt1_pace_min_km": 5.0,
    }

    selected_time_min, selected_source = selected_road_capability(profile, "road_marathon")

    assert selected_time_min == 205.0
    assert selected_source == "Manual Profile"

    del profile["best_likely_marathon_time_min"]
    selected_time_min, selected_source = selected_road_capability(profile, "road_marathon")

    assert selected_time_min == 210.0
    assert selected_source == "COROS"

    del profile["predictor_marathon_time_min"]
    selected_time_min, selected_source = selected_road_capability(profile, "road_marathon")

    assert selected_time_min is not None
    assert selected_source == "LT-Derived Model"


def test_modeled_road_best_likely_time_uses_current_profile_heuristic() -> None:
    profile = {
        "lt1_pace_min_km": 5.0,
        "lt2_pace_min_km": 4.25,
    }

    half_time_min = modeled_road_best_likely_time_min(profile, "half_marathon")
    marathon_time_min = modeled_road_best_likely_time_min(profile, "road_marathon")

    assert half_time_min is not None
    assert marathon_time_min is not None
    assert 89.65 <= half_time_min <= 105.49
    assert 179.33 <= marathon_time_min <= 210.97
    assert marathon_time_min > half_time_min


def testroad_capability_sources_marks_selected_source() -> None:
    sources = road_capability_sources(
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


def test_default_config_for_road_event_uses_selected_capability_time() -> None:
    event = get_curated_event("marathon-etoiles-baie")

    config = default_config_for_event(
        event,
        {
            "best_likely_marathon_time_min": 198.0,
            "lt1_pace_min_km": 5.0,
        },
    )

    assert config["target_finish_time_min"] == 198.0
    assert config["race_intent"] == "controlled"


def test_default_config_for_road_event_uses_modeled_capability_when_no_override() -> None:
    event = get_curated_event("semi-marathon-finistere")

    config = default_config_for_event(
        event,
        {
            "lt1_pace_min_km": 5.0,
            "lt2_pace_min_km": 4.25,
        },
    )

    assert config["target_finish_time_min"] is not None
    assert config["marathon_pace_min_km"] is not None
    assert config["target_finish_time_min"] < 105.5


def testcourse_overview_rows_include_known_start_time() -> None:
    event = get_curated_event("marathon-etoiles-baie")

    rows = course_overview_rows(42.06, event)

    assert any(row == {"label": "Start Time", "value": "9:00 AM"} for row in rows)


def testderived_hr_guardrail_cap_uses_profile_and_policy() -> None:
    guardrail = derived_hr_guardrail_cap(
        {"lt1_hr": 152, "lt2_hr": 170},
        "technical_trail_ultra",
        "conservative",
    )

    assert guardrail == 153


def testderived_hr_guardrail_cap_varies_by_event_type() -> None:
    profile = {"lt1_hr": 152, "lt2_hr": 170}

    technical = derived_hr_guardrail_cap(profile, "technical_trail_ultra", "steady")
    fire_road = derived_hr_guardrail_cap(profile, "fire_road_ultra", "steady")
    road = derived_hr_guardrail_cap(profile, "road_marathon", "steady")

    assert technical is not None
    assert fire_road is not None
    assert road is not None
    assert technical < fire_road < road

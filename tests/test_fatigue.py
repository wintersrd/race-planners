from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.fatigue import fatigue_multiplier
from race_planners.models import PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


def testfatigue_multiplier_progresses_by_model() -> None:
    assert fatigue_multiplier("road_marathon", 0.5) == 1.0
    assert fatigue_multiplier("road_marathon", 1.0) > 1.0
    assert fatigue_multiplier("technical_trail_ultra", 1.0) > fatigue_multiplier(
        "road_marathon", 1.0
    )


def test_fade_profile_presets_change_trail_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    stable = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            fade_profile_preset="stable",
        ),
    )
    blow_up = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            fade_profile_preset="blow_up_risk",
        ),
    )

    assert blow_up.total_time_min > stable.total_time_min
    assert "Fade profile progressively slows pace across the event." in blow_up.assumptions


def test_trail_durability_factor_reduces_fade_cost() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    short_course = get_course_by_id(repo_root, "semi-marathon-finistere")
    short_loaded = load_course_trackpoints(short_course)
    long_course = get_course_by_id(repo_root, "grf92")
    long_loaded = load_course_trackpoints(long_course)
    very_long_course = get_course_by_id(repo_root, "grf166")
    very_long_loaded = load_course_trackpoints(very_long_course)

    short_durable = calculate_plan(
        short_loaded,
        PacingConfig(
            race_model="half_marathon",
            input_mode="effort_anchor",
            marathon_pace_min_km=5.0,
            athlete_durability_factor=1.0,
        ),
    )
    short_fragile = calculate_plan(
        short_loaded,
        PacingConfig(
            race_model="half_marathon",
            input_mode="effort_anchor",
            marathon_pace_min_km=5.0,
            athlete_durability_factor=-1.0,
        ),
    )

    long_durable = calculate_plan(
        long_loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            fade_profile_preset="progressive_fade",
            athlete_durability_factor=1.0,
        ),
    )
    long_fragile = calculate_plan(
        long_loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            fade_profile_preset="progressive_fade",
            athlete_durability_factor=-1.0,
        ),
    )

    very_long_durable = calculate_plan(
        very_long_loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=9.0,
            hike_pace_min_km=14.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            fade_profile_preset="progressive_fade",
            athlete_durability_factor=1.0,
        ),
    )
    very_long_fragile = calculate_plan(
        very_long_loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=9.0,
            hike_pace_min_km=14.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            fade_profile_preset="progressive_fade",
            athlete_durability_factor=-1.0,
        ),
    )

    short_delta = short_fragile.total_time_min - short_durable.total_time_min
    long_delta = long_fragile.total_time_min - long_durable.total_time_min
    very_long_delta = very_long_fragile.total_time_min - very_long_durable.total_time_min

    assert short_delta > 0
    assert long_delta > short_delta * 3
    assert very_long_delta > long_delta * 1.25

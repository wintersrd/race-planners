from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.models import PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


def test_heat_penalty_materially_slows_grf92_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    cool = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            peak_temperature_c=10.0,
            event_start_time_local="06:30",
        ),
    )
    hot = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            peak_temperature_c=30.0,
            event_start_time_local="06:30",
        ),
    )

    assert hot.total_time_min > cool.total_time_min
    assert hot.total_time_min - cool.total_time_min > 30.0
    assert any("assumption.weather_heat" in a for a in hot.assumptions)


def test_heat_curve_handles_multi_day_ultra_without_crash() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf166")
    loaded = load_course_trackpoints(course)

    result = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=9.0,
            hike_pace_min_km=14.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            peak_temperature_c=25.0,
            event_start_time_local="17:00",
        ),
    )

    assert result.total_time_min > 0
    assert result.moving_time_min > 0
    assert any("assumption.weather_heat" in a for a in result.assumptions)


def test_trail_heat_and_hill_tolerance_reduce_penalties() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    tolerant = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            peak_temperature_c=28.0,
            event_start_time_local="06:30",
            athlete_heat_tolerance=1.0,
            athlete_hill_tolerance=1.0,
        ),
    )
    intolerant = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            peak_temperature_c=28.0,
            event_start_time_local="06:30",
            athlete_heat_tolerance=-1.0,
            athlete_hill_tolerance=-1.0,
        ),
    )

    assert tolerant.total_time_min < intolerant.total_time_min

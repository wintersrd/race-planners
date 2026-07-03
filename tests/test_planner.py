from pathlib import Path

import pytest

from race_planners.course_library import get_course_by_id
from race_planners.models import PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


def test_calculate_plan_for_marathon_pace_model() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    config = PacingConfig(
        race_model="road_marathon",
        input_mode="effort_anchor",
        marathon_pace_min_km=5.5,
    )
    result = calculate_plan(loaded, config)

    assert result.total_distance_km > 20.0
    assert result.total_time_min > 0
    assert len(result.aid_arrival_times_min) == len(course.aid_stops_km)
    assert len(result.aid_station_etas) == len(course.aid_stops_km)
    assert result.total_time_min == pytest.approx(
        result.moving_time_min + result.total_rest_time_min
    )


def test_fire_road_requires_all_effort_inputs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    config = PacingConfig(
        race_model="fire_road_ultra",
        input_mode="effort_anchor",
        z1_pace_min_km=7.0,
    )

    with pytest.raises(ValueError, match="requires Z1 pace, Z2 pace, and hike pace"):
        calculate_plan(loaded, config)


def test_technical_trail_finish_time_mode_derives_effort_anchors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    config = PacingConfig(
        race_model="technical_trail_ultra",
        input_mode="finish_time",
        target_finish_time_min=150.0,
        climb_hike_threshold_percent=12.0,
        descent_caution="medium",
    )

    result = calculate_plan(loaded, config)

    assert abs(result.total_time_min - 150.0) < 2.0


def test_fire_road_finish_time_mode_derives_effort_anchors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    config = PacingConfig(
        race_model="fire_road_ultra",
        input_mode="finish_time",
        target_finish_time_min=145.0,
        climb_hike_threshold_percent=12.0,
    )

    result = calculate_plan(loaded, config)

    assert abs(result.total_time_min - 145.0) < 2.0


def test_rest_stops_are_modeled_as_additive_timing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    config = PacingConfig(
        race_model="half_marathon",
        input_mode="finish_time",
        target_finish_time_min=110.0,
        rest_duration_sec=30,
    )

    result = calculate_plan(loaded, config)

    assert result.total_rest_time_min == pytest.approx(1.5)
    assert result.total_time_min == pytest.approx(
        result.moving_time_min + result.total_rest_time_min
    )
    assert result.total_time_min - result.moving_time_min == pytest.approx(1.5)
    assert result.aid_station_etas[0].departure_elapsed_time_min == pytest.approx(
        result.aid_station_etas[0].arrival_elapsed_time_min + 0.5
    )


def test_pacing_bias_progressively_changes_total_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    conservative = calculate_plan(
        loaded,
        PacingConfig(
            race_model="road_marathon",
            input_mode="effort_anchor",
            marathon_pace_min_km=5.5,
            pacing_bias=5.0,
        ),
    )
    aggressive = calculate_plan(
        loaded,
        PacingConfig(
            race_model="road_marathon",
            input_mode="effort_anchor",
            marathon_pace_min_km=5.5,
            pacing_bias=-5.0,
        ),
    )

    assert conservative.total_time_min > aggressive.total_time_min
    assert "Pacing bias progressively shifts pace across the course." in conservative.assumptions


def test_climb_hike_threshold_meaningfully_changes_grf92_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    low_threshold = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=5.0,
            descent_caution="medium",
        ),
    )
    high_threshold = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=15.0,
            descent_caution="medium",
        ),
    )

    assert low_threshold.total_time_min > high_threshold.total_time_min
    assert low_threshold.total_time_min - high_threshold.total_time_min > 3.0


def test_rpe_aggressiveness_changes_grf92_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    conservative = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            rpe_target=3.0,
        ),
    )
    aggressive = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            rpe_target=9.0,
        ),
    )

    assert conservative.total_time_min > aggressive.total_time_min
    assert (
        "RPE aggressiveness nudges pacing more conservatively or aggressively."
        in conservative.assumptions
    )


def test_hr_guardrail_changes_grf92_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    lower_cap = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            hr_cap=130,
        ),
    )
    higher_cap = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            hr_cap=180,
        ),
    )

    assert lower_cap.total_time_min > higher_cap.total_time_min
    assert (
        "HR guardrail tempers pacing on steeper or later-course segments." in lower_cap.assumptions
    )


def test_descent_caution_changes_grf92_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    lower_caution = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="low",
            rpe_target=6.0,
            hr_cap=155,
        ),
    )
    higher_caution = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
            climb_hike_threshold_percent=12.0,
            descent_caution="high",
            rpe_target=6.0,
            hr_cap=155,
        ),
    )

    assert higher_caution.total_time_min > lower_caution.total_time_min
    assert higher_caution.total_time_min - lower_caution.total_time_min > 1.0


def test_segment_summaries_include_elevation_gain_and_loss() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    result = calculate_plan(
        loaded,
        PacingConfig(
            race_model="road_marathon",
            input_mode="effort_anchor",
            marathon_pace_min_km=5.5,
        ),
    )

    assert result.segments
    assert any(segment.elevation_gain_m > 0 for segment in result.segments)
    assert any(segment.elevation_loss_m > 0 for segment in result.segments)

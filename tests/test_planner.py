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

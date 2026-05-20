from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.models import Course, PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


def test_calculate_plan_ignores_out_of_range_aid_stops() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base = get_course_by_id(repo_root, "semi-marathon-finistere")
    course = Course(
        course_id=base.course_id,
        name=base.name,
        gpx_path=base.gpx_path,
        aid_stops_km=[-1.0, 5.3, 999.0],
        terrain=base.terrain,
    )
    loaded = load_course_trackpoints(course)
    config = PacingConfig(race_model="road_marathon", input_mode="effort_anchor", marathon_pace_min_km=5.5)

    result = calculate_plan(loaded, config)

    assert len(result.aid_arrival_times_min) == 1


def test_calculate_plan_supports_course_without_aid_stops() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base = get_course_by_id(repo_root, "semi-marathon-finistere")
    course = Course(
        course_id=base.course_id,
        name=base.name,
        gpx_path=base.gpx_path,
        aid_stops_km=[],
        terrain=base.terrain,
    )
    loaded = load_course_trackpoints(course)
    config = PacingConfig(race_model="half_marathon", input_mode="finish_time", target_finish_time_min=110.0)

    result = calculate_plan(loaded, config)

    assert result.aid_arrival_times_min == []
    assert result.total_time_min > 0

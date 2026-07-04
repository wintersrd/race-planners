from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.models import AidStation, PacingConfig, PaceSplit
from race_planners.planner import calculate_plan, load_course_trackpoints
from race_planners.segments import build_segment_summaries


def test_segment_summary_groups_adjacent_split_types() -> None:
    splits = [
        PaceSplit(1.0, 6.0, 3.1, 6.0, 6.0),
        PaceSplit(2.0, 6.2, 2.8, 6.2, 12.2),
        PaceSplit(3.0, 5.5, 0.1, 5.5, 17.7),
        PaceSplit(4.0, 5.2, -3.3, 5.2, 22.9),
    ]

    segments = build_segment_summaries(splits)

    assert [s.segment_type for s in segments] == ["climb", "flat", "descent"]
    assert segments[0].distance_km == 2.0
    assert segments[-1].end_km == 4.0
    assert segments[0].block_label == "Start to Finish"


def test_segment_summary_respects_aid_station_boundaries() -> None:
    splits = [
        PaceSplit(1.0, 6.0, 3.0, 6.0, 6.0),
        PaceSplit(2.0, 6.0, 3.2, 6.0, 12.0),
        PaceSplit(3.0, 5.0, -3.0, 5.0, 17.0),
    ]

    segments = build_segment_summaries(
        splits,
        aid_stations=[AidStation(distance_km=1.5, label="Aid 1")],
        total_distance_km=3.0,
    )

    assert [(segment.start_km, segment.end_km) for segment in segments] == [
        (0.0, 1.5),
        (1.5, 2.0),
        (2.0, 3.0),
    ]
    assert [segment.block_label for segment in segments] == [
        "Start to Aid 1",
        "Aid 1 to Finish",
        "Aid 1 to Finish",
    ]
    assert segments[0].section_name == "Start to Aid 1: climb"


def test_climb_hike_threshold_meaningfully_changes_grf92_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    low_threshold = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=5.0,
            descent_caution="medium",
        ),
    )
    high_threshold = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=15.0,
            descent_caution="medium",
        ),
    )

    assert low_threshold.total_time_min > high_threshold.total_time_min
    assert low_threshold.total_time_min - high_threshold.total_time_min > 3.0


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

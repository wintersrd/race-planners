from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.planner import load_course_trackpoints
from race_planners.road_capability import (
    classify_road_effort_band,
    classify_road_feasibility,
    classify_road_recovery_cost,
    estimate_road_adjusted_best_likely,
    estimate_road_best_likely_pace_min_km,
    estimate_road_best_likely_time_min,
    estimate_road_intent_target_time_min,
)


def test_road_best_likely_solver_stays_between_lt1_and_lt2() -> None:
    pace_min_km = estimate_road_best_likely_pace_min_km("half_marathon", 5.0, 4.25)
    time_min = estimate_road_best_likely_time_min("road_marathon", 5.0, 4.25)

    assert 4.25 <= pace_min_km <= 5.0
    assert (42.195 * 4.25) <= time_min <= (42.195 * 5.0)
    assert time_min >= 189.5


def test_road_best_likely_solver_places_faster_half_runner_nearer_lt2() -> None:
    fast_runner_pace = estimate_road_best_likely_pace_min_km("half_marathon", 5.0, 4.25)
    slower_runner_pace = estimate_road_best_likely_pace_min_km("half_marathon", 7.0, 6.0)

    assert (fast_runner_pace - 4.25) < (slower_runner_pace - 6.0)


def test_road_best_likely_solver_is_less_aggressive_for_known_profile_example() -> None:
    half_time_min = estimate_road_best_likely_time_min("half_marathon", 5.25, 4.25)
    marathon_time_min = estimate_road_best_likely_time_min("road_marathon", 5.25, 4.25)

    assert 91.5 <= half_time_min <= 95.0
    assert 193.0 <= marathon_time_min <= 201.0


def test_road_best_likely_solver_returns_more_conservative_marathon_than_half() -> None:
    half_pace = estimate_road_best_likely_pace_min_km("half_marathon", 5.0, 4.25)
    marathon_pace = estimate_road_best_likely_pace_min_km("road_marathon", 5.0, 4.25)

    assert marathon_pace > half_pace


def test_road_adjusted_best_likely_slows_with_heat_and_hills() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "marathon-etoiles-baie")
    loaded = load_course_trackpoints(course)

    adjusted = estimate_road_adjusted_best_likely(
        loaded,
        base_time_min=198.0,
        peak_temperature_c=20.0,
        start_time_local="09:00",
        hill_tolerance=0.0,
        heat_tolerance=0.0,
    )

    assert adjusted["adjusted_time_min"] > 198.0
    assert adjusted["course_multiplier"] >= 1.0
    assert adjusted["weather_multiplier"] >= 1.0


def test_road_adjusted_best_likely_respects_tolerance_modifiers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "marathon-etoiles-baie")
    loaded = load_course_trackpoints(course)

    tolerant = estimate_road_adjusted_best_likely(
        loaded,
        base_time_min=198.0,
        peak_temperature_c=20.0,
        start_time_local="09:00",
        hill_tolerance=1.0,
        heat_tolerance=1.0,
    )
    intolerant = estimate_road_adjusted_best_likely(
        loaded,
        base_time_min=198.0,
        peak_temperature_c=20.0,
        start_time_local="09:00",
        hill_tolerance=-1.0,
        heat_tolerance=-1.0,
    )

    assert tolerant["adjusted_time_min"] < intolerant["adjusted_time_min"]


def test_road_intent_target_time_and_labels_shift_with_intent() -> None:
    adjusted_best_likely_time_min = 200.0

    best_effort = estimate_road_intent_target_time_min(adjusted_best_likely_time_min, "best_effort")
    controlled = estimate_road_intent_target_time_min(adjusted_best_likely_time_min, "controlled")
    easy = estimate_road_intent_target_time_min(adjusted_best_likely_time_min, "easy_durable")

    assert best_effort < controlled < easy
    assert classify_road_feasibility(adjusted_best_likely_time_min, easy) == "Very High"
    assert classify_road_feasibility(adjusted_best_likely_time_min, best_effort) == "Stretch"
    assert classify_road_effort_band(adjusted_best_likely_time_min, easy) == "Controlled"
    assert classify_road_effort_band(adjusted_best_likely_time_min, best_effort) == "Near Limit"
    assert classify_road_recovery_cost(adjusted_best_likely_time_min, easy) == "Low"
    assert classify_road_recovery_cost(adjusted_best_likely_time_min, best_effort) == "High"

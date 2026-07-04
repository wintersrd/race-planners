from pathlib import Path

import pytest

from race_planners.course_library import get_course_by_id
from race_planners.models import PacingConfig
from race_planners.planner import (
    calculate_plan,
    estimate_road_best_likely_pace_min_km,
    estimate_road_best_likely_time_min,
    load_course_trackpoints,
)


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


def test_effort_policy_changes_grf92_finish_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    conservative = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            effort_policy="conservative",
            rpe_target=3.0,
        ),
    )
    aggressive = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            effort_policy="aggressive",
            rpe_target=9.0,
        ),
    )

    assert conservative.total_time_min > aggressive.total_time_min
    assert (
        "Effort policy nudges pacing more conservatively or aggressively."
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
            use_hr_guardrail=True,
            athlete_lt1_hr=152,
            athlete_lt2_hr=170,
            athlete_lt1_pace_min_km=5.0,
            athlete_lt2_pace_min_km=4.25,
            athlete_flat_trail_slowdown_sec_km=25.0,
            athlete_technical_trail_slowdown_sec_km=45.0,
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
            use_hr_guardrail=True,
            athlete_lt1_hr=152,
            athlete_lt2_hr=170,
            athlete_lt1_pace_min_km=5.0,
            athlete_lt2_pace_min_km=4.25,
            athlete_flat_trail_slowdown_sec_km=25.0,
            athlete_technical_trail_slowdown_sec_km=45.0,
            hr_cap=180,
        ),
    )

    assert lower_cap.total_time_min > higher_cap.total_time_min
    assert (
        "Derived HR guardrail tempers pacing on steeper or later-course segments."
        in lower_cap.assumptions
    )
    assert any(
        "Derived HR strategy targets roughly" in assumption for assumption in lower_cap.assumptions
    )


def test_hr_guardrail_uses_profile_pace_relationships() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)

    lower_technical_tax = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            effort_policy="steady",
            use_hr_guardrail=True,
            athlete_lt1_hr=152,
            athlete_lt2_hr=170,
            athlete_lt1_pace_min_km=5.0,
            athlete_lt2_pace_min_km=4.25,
            athlete_flat_trail_slowdown_sec_km=20.0,
            athlete_technical_trail_slowdown_sec_km=20.0,
            hr_cap=156,
        ),
    )
    higher_technical_tax = calculate_plan(
        loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=6.2,
            hike_pace_min_km=10.5,
            climb_hike_threshold_percent=12.0,
            descent_caution="medium",
            effort_policy="steady",
            use_hr_guardrail=True,
            athlete_lt1_hr=152,
            athlete_lt2_hr=170,
            athlete_lt1_pace_min_km=5.0,
            athlete_lt2_pace_min_km=4.25,
            athlete_flat_trail_slowdown_sec_km=20.0,
            athlete_technical_trail_slowdown_sec_km=60.0,
            hr_cap=156,
        ),
    )

    assert higher_technical_tax.total_time_min > lower_technical_tax.total_time_min


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
    assert any("Weather model applies heat penalty" in a for a in hot.assumptions)


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
    assert any("Weather model applies heat penalty" in a for a in result.assumptions)


def test_road_best_likely_solver_stays_between_lt1_and_lt2() -> None:
    pace_min_km = estimate_road_best_likely_pace_min_km("half_marathon", 5.0, 4.25)
    time_min = estimate_road_best_likely_time_min("road_marathon", 5.0, 4.25)

    assert 4.25 <= pace_min_km <= 5.0
    assert (42.195 * 4.25) <= time_min <= (42.195 * 5.0)


def test_road_best_likely_solver_places_faster_half_runner_nearer_lt2() -> None:
    fast_runner_pace = estimate_road_best_likely_pace_min_km("half_marathon", 5.0, 4.25)
    slower_runner_pace = estimate_road_best_likely_pace_min_km("half_marathon", 7.0, 6.0)

    assert (fast_runner_pace - 4.25) < (slower_runner_pace - 6.0)


def test_road_best_likely_solver_returns_more_conservative_marathon_than_half() -> None:
    half_pace = estimate_road_best_likely_pace_min_km("half_marathon", 5.0, 4.25)
    marathon_pace = estimate_road_best_likely_pace_min_km("road_marathon", 5.0, 4.25)

    assert marathon_pace > half_pace

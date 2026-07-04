from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.models import PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


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

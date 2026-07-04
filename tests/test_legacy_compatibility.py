"""Stability regression test for the Finistere course planning output.

Replaces the former legacy-compatibility test that imported ``app.py`` as an
oracle.  The new engine has been validated against real-world results, so this
test now guards against unintended changes to the canonical output for a known
course and configuration.
"""

from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.models import PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


def test_finistere_half_marathon_plan_stability() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)

    config = PacingConfig(
        race_model="half_marathon",
        input_mode="effort_anchor",
        marathon_pace_min_km=5.31,
    )
    result = calculate_plan(loaded, config)

    assert round(result.total_distance_km, 2) == 21.14
    assert len(result.splits) == 22
    assert abs(result.moving_time_min - 112.83) < 1.0

import importlib.util
from pathlib import Path
from typing import Any, cast

from race_planners.course_library import get_course_by_id
from race_planners.models import PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


def _load_legacy_app_module() -> Any:
    app_path = Path(__file__).resolve().parents[1] / "semi-marathon-finistere" / "app.py"
    spec = importlib.util.spec_from_file_location("legacy_semi_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_planner_gap_model_stays_close_to_legacy_for_same_base_gap_pace() -> None:
    legacy = cast(Any, _load_legacy_app_module())
    trackpoints, total_distance_km, gap_adjusted_distance_m = legacy.load_gpx_data.__wrapped__(
        legacy.GPX_FILE,
        legacy.SMOOTHING_WINDOW,
    )
    legacy_result = legacy.calculate_pacing(
        trackpoints=trackpoints,
        target_finish_time_min=110.0,
        rest_stops=legacy.REST_STOPS,
        total_distance_km=total_distance_km,
        gap_adjusted_distance_m=gap_adjusted_distance_m,
        power_fade=0,
        rest_duration_sec=30,
    )

    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)
    new_config = PacingConfig(
        race_model="half_marathon",
        input_mode="effort_anchor",
        marathon_pace_min_km=legacy_result["target_gap_pace"],
    )
    new_result = calculate_plan(loaded, new_config)

    legacy_finish = legacy_result["calculated_finish_time_min"]
    assert abs(new_result.moving_time_min - legacy_finish) < 2.0
    assert len(new_result.splits) == len(legacy_result["km_splits"])

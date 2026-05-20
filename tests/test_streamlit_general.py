from pathlib import Path

from race_planners.models import PacingConfig
from race_planners.plan_io import export_plan_json
from race_planners.streamlit_general import load_plan_into_state


def test_load_plan_into_state_returns_user_facing_missing_gpx_error(tmp_path: Path) -> None:
    payload_json = export_plan_json(
        course_id="library:missing-course",
        gpx_filename="missing-course.gpx",
        config=PacingConfig(race_model="road_marathon", input_mode="finish_time"),
    )

    state, error = load_plan_into_state(
        payload_json, tmp_path, {"general_config": {}, "general_course_id": "x"}
    )

    assert error is not None
    assert "Missing course file: missing-course.gpx" in error
    assert state["general_course_id"] == "x"


def test_load_plan_into_state_restores_course_and_config(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir(parents=True)
    (course_dir / "sample.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    payload_json = export_plan_json(
        course_id="semi-marathon-finistere",
        gpx_filename="sample.gpx",
        config=PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.8,
            hike_pace_min_km=13.2,
        ),
    )

    state, error = load_plan_into_state(
        payload_json, tmp_path, {"general_config": {}, "general_course_id": "x"}
    )

    assert error is None
    assert state["general_course_id"] == "semi-marathon-finistere"
    assert state["general_config"]["race_model"] == "technical_trail_ultra"

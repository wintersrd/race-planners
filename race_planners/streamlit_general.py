from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from race_planners.course_library import get_course_by_id, list_courses, save_uploaded_gpx
from race_planners.models import PacingConfig
from race_planners.plan_io import (
    ensure_gpx_exists_for_plan,
    export_plan_json,
    import_plan_json,
)
from race_planners.planner import calculate_plan, load_course_trackpoints


def _default_config() -> dict[str, Any]:
    return {
        "race_model": "road_marathon",
        "input_mode": "finish_time",
        "target_finish_time_min": 240.0,
        "marathon_pace_min_km": None,
        "z1_pace_min_km": None,
        "z2_pace_min_km": None,
        "flat_pace_min_km": None,
        "hike_pace_min_km": None,
        "climb_hike_threshold_percent": 12.0,
        "descent_caution": "medium",
        "rest_duration_sec": 30,
        "rpe_target": None,
        "hr_cap": None,
    }


def load_plan_into_state(
    plan_json: str,
    repo_root: Path,
    current_state: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Parse + validate plan JSON and return updated state or UI-safe error."""
    try:
        payload = import_plan_json(plan_json)
        ensure_gpx_exists_for_plan(
            payload,
            [repo_root / "semi-marathon-finistere", repo_root / "courses", repo_root],
        )
    except (ValueError, FileNotFoundError) as exc:
        return current_state, str(exc)

    updated_state = dict(current_state)
    updated_state["general_course_id"] = str(payload["course_id"])
    updated_state["general_config"] = dict(payload["config"])
    return updated_state, None


def render_general_planner(repo_root: Path) -> None:
    st.title("General Race Planner (Beta)")
    st.caption("Pluggable pacing models with local course library + GPX upload + JSON plans.")

    st.session_state.setdefault("general_config", _default_config())
    st.session_state.setdefault("general_course_id", "semi-marathon-finistere")

    with st.expander("Load Saved Plan"):
        loaded_plan = st.file_uploader("Plan JSON", type=["json"], key="plan_json_uploader")
        if loaded_plan is not None:
            updated_state, load_error = load_plan_into_state(
                loaded_plan.getvalue().decode("utf-8"),
                repo_root,
                {str(key): st.session_state[key] for key in st.session_state.keys()},
            )
            if load_error is not None:
                st.error(f"Could not load plan: {load_error}")
            else:
                st.session_state["general_course_id"] = updated_state["general_course_id"]
                st.session_state["general_config"] = updated_state["general_config"]
                st.success("Plan loaded. Review values and click Calculate.")

    courses = list_courses(repo_root)
    course_index = 0
    for idx, course in enumerate(courses):
        if course.course_id == st.session_state["general_course_id"]:
            course_index = idx
            break

    col_a, col_b = st.columns([3, 2])
    with col_a:
        selected_course = st.selectbox(
            "Course Library",
            options=courses,
            index=course_index,
            format_func=lambda c: f"{c.name} ({c.course_id})",
            help="Choose a built-in route or any GPX from the local repository library.",
        )

    with col_b:
        uploaded_gpx = st.file_uploader(
            "Upload GPX to Local Library",
            type=["gpx"],
            key="general_gpx_uploader",
            help="Uploaded files are stored under courses/uploads and become selectable courses.",
        )
        if uploaded_gpx is not None:
            upload_course = save_uploaded_gpx(
                uploaded_gpx.name,
                uploaded_gpx.getvalue(),
                repo_root / "courses" / "uploads",
            )
            st.session_state["general_course_id"] = upload_course.course_id
            st.success(f"Stored upload as {upload_course.gpx_path.name}. Select it in course list.")

    st.session_state["general_course_id"] = selected_course.course_id

    cfg = st.session_state["general_config"]

    race_model = st.selectbox(
        "Race Model",
        options=[
            "road_marathon",
            "half_marathon",
            "fire_road_ultra",
            "technical_trail_ultra",
        ],
        index=["road_marathon", "half_marathon", "fire_road_ultra", "technical_trail_ultra"].index(
            cfg.get("race_model", "road_marathon")
        ),
    )

    input_mode = st.radio(
        "Input Mode",
        options=["finish_time", "effort_anchor"],
        index=0 if cfg.get("input_mode") == "finish_time" else 1,
        horizontal=True,
        help="Finish-time derives base pace from target finish. Effort-anchor uses your known pace anchor.",
    )

    target_finish_time_min: float | None = None
    marathon_pace_min_km: float | None = None
    z1_pace_min_km: float | None = None
    z2_pace_min_km: float | None = None
    flat_pace_min_km: float | None = None
    hike_pace_min_km: float | None = None

    if race_model in {"road_marathon", "half_marathon"}:
        if input_mode == "finish_time":
            target_finish_time_min = st.number_input(
                "Target Finish Time (minutes)",
                min_value=30.0,
                max_value=2400.0,
                value=float(cfg.get("target_finish_time_min") or 240.0),
                step=5.0,
            )
        else:
            marathon_pace_min_km = st.number_input(
                "Marathon/Half Anchor Pace (min/km)",
                min_value=3.0,
                max_value=20.0,
                value=float(cfg.get("marathon_pace_min_km") or 5.5),
                step=0.1,
            )
    elif race_model == "fire_road_ultra":
        z1_pace_min_km = st.number_input(
            "Z1 Pace (min/km)",
            min_value=4.0,
            max_value=25.0,
            value=float(cfg.get("z1_pace_min_km") or 8.0),
            step=0.1,
        )
        z2_pace_min_km = st.number_input(
            "Z2 Pace (min/km)",
            min_value=3.0,
            max_value=20.0,
            value=float(cfg.get("z2_pace_min_km") or 7.0),
            step=0.1,
        )
        hike_pace_min_km = st.number_input(
            "Hike Pace (min/km)",
            min_value=5.0,
            max_value=40.0,
            value=float(cfg.get("hike_pace_min_km") or 12.0),
            step=0.1,
        )
    else:
        flat_pace_min_km = st.number_input(
            "Flat Pace (min/km)",
            min_value=4.0,
            max_value=25.0,
            value=float(cfg.get("flat_pace_min_km") or 8.5),
            step=0.1,
        )
        hike_pace_min_km = st.number_input(
            "Hike Pace (min/km)",
            min_value=5.0,
            max_value=40.0,
            value=float(cfg.get("hike_pace_min_km") or 13.0),
            step=0.1,
        )

    climb_hike_threshold_percent = st.slider(
        "Climb Hike Threshold (%)",
        min_value=5.0,
        max_value=25.0,
        value=float(cfg.get("climb_hike_threshold_percent", 12.0)),
        step=0.5,
    )
    descent_caution = st.selectbox(
        "Descent Caution",
        options=["low", "medium", "high"],
        index=["low", "medium", "high"].index(cfg.get("descent_caution", "medium")),
    )
    col_anchor_a, col_anchor_b = st.columns(2)
    with col_anchor_a:
        rpe_target = st.number_input(
            "RPE Aggressiveness",
            min_value=1.0,
            max_value=10.0,
            value=6.0,
            step=0.5,
            help="Secondary policy input: lower is conservative, higher is aggressive.",
        )
    with col_anchor_b:
        hr_cap = st.number_input(
            "HR Guardrail Cap",
            min_value=80,
            max_value=210,
            value=155,
            step=1,
            help="Safety ceiling used as a guardrail when validating pacing choices.",
        )

    new_config = PacingConfig(
        race_model=race_model,
        input_mode=input_mode,
        target_finish_time_min=target_finish_time_min,
        marathon_pace_min_km=marathon_pace_min_km,
        z1_pace_min_km=z1_pace_min_km,
        z2_pace_min_km=z2_pace_min_km,
        flat_pace_min_km=flat_pace_min_km,
        hike_pace_min_km=hike_pace_min_km,
        climb_hike_threshold_percent=climb_hike_threshold_percent,
        descent_caution=descent_caution,
        rest_duration_sec=30,
        rpe_target=rpe_target,
        hr_cap=hr_cap,
    )
    st.session_state["general_config"] = asdict(new_config)

    if st.button("Calculate Plan", type="primary"):
        if selected_course.course_id.startswith("upload:"):
            selected_course = get_course_by_id(repo_root, selected_course.course_id)

        loaded = load_course_trackpoints(selected_course)
        result = calculate_plan(loaded, new_config)
        st.session_state["general_result"] = result
        st.session_state["general_selected_course"] = selected_course

    if "general_result" in st.session_state:
        result = st.session_state["general_result"]
        chosen_course = st.session_state["general_selected_course"]
        st.subheader("Plan Output")
        st.write(
            f"Distance: {result.total_distance_km:.2f} km | Time: {result.total_time_min:.1f} min"
        )
        st.write(
            "Aid arrivals (min): "
            + ", ".join(f"{value:.1f}" for value in result.aid_arrival_times_min)
        )
        st.dataframe(
            [
                {
                    "km": split.km,
                    "pace": round(split.actual_pace_min_km, 2),
                    "grade": round(split.grade_percent, 2),
                    "segment_min": round(split.segment_time_min, 2),
                    "cum_min": round(split.cumulative_time_min, 2),
                }
                for split in result.splits
            ],
            width="stretch",
            hide_index=True,
        )

        plan_json = export_plan_json(
            course_id=chosen_course.course_id,
            gpx_filename=chosen_course.gpx_path.name,
            config=PacingConfig(**st.session_state["general_config"]),
        )
        st.download_button(
            "Download plan JSON",
            data=plan_json,
            file_name="race-plan.json",
            mime="application/json",
        )

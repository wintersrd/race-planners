from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from race_planners.course_library import get_course_by_id
from race_planners.event_catalog import (
    get_curated_event_by_course_id,
    get_event_template,
    list_curated_events,
)
from race_planners.grade import elevation_changes
from race_planners.models import CuratedEvent, PacingConfig
from race_planners.plan_io import (
    ensure_gpx_exists_for_plan,
    export_plan_json,
    import_plan_json,
)
from race_planners.planner import calculate_plan, load_course_trackpoints


def _course_overview_rows(total_distance_km: float, event: CuratedEvent) -> list[dict[str, str]]:
    aid_mode = "Configured" if event.aid_stops_km else "From GPX"
    return [
        {"label": "Distance", "value": f"{total_distance_km:.2f} km"},
        {"label": "Terrain", "value": event.terrain.title()},
        {"label": "Race Model", "value": event.race_model},
        {"label": "Aid Stations", "value": aid_mode},
    ]


def _plot_course_profile(trackpoints: list[Any], aid_distances_km: list[float]) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    distances_km = [point.distance_from_start / 1000 for point in trackpoints]
    elevations = [point.elevation for point in trackpoints]
    ax.fill_between(distances_km, elevations, color="#2E86AB", alpha=0.28)
    ax.plot(distances_km, elevations, color="#2E86AB", linewidth=2.2)
    for aid_distance_km in aid_distances_km:
        ax.axvline(aid_distance_km, color="#E94F37", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Course Elevation Profile", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.18)
    plt.tight_layout()
    return fig


def _plot_pace_profile(result: Any) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    kms = [split.km for split in result.splits]
    paces = [split.actual_pace_min_km for split in result.splits]
    colors = ["#E94F37" if split.grade_percent > 1.5 else "#2E86AB" for split in result.splits]
    ax.bar(kms, paces, width=0.85, color=colors, alpha=0.75)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pace (min/km)")
    ax.set_title("Planned Pace Profile", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    return fig


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
        "pacing_bias": 0.0,
        "rpe_target": None,
        "hr_cap": None,
    }


def _default_config_for_event(event: CuratedEvent) -> dict[str, Any]:
    config = _default_config()
    config["race_model"] = event.race_model
    config["input_mode"] = event.default_input_mode

    if event.race_model == "half_marathon":
        config["target_finish_time_min"] = 105.0
    elif event.race_model == "road_marathon":
        config["target_finish_time_min"] = 240.0
    else:
        config["input_mode"] = "effort_anchor"
        config["target_finish_time_min"] = None
        config["flat_pace_min_km"] = 8.5
        config["hike_pace_min_km"] = 13.0

    return config


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
    matched_event = get_curated_event_by_course_id(repo_root, str(payload["course_id"]))
    if matched_event is not None:
        updated_state["general_event_id"] = matched_event.event_id
    return updated_state, None


def render_general_planner(repo_root: Path) -> None:
    st.title("Unified Event Planner")
    st.caption("Choose a curated event and plan it through one event-first pacing flow.")

    st.session_state.setdefault("general_config", _default_config())
    st.session_state.setdefault("general_event_id", "semi-marathon-finistere")
    st.session_state.setdefault("general_course_id", "semi-marathon-finistere")

    events = list_curated_events(repo_root)
    if not events:
        st.error("No curated events are currently available in the repository.")
        return

    event_index = 0
    for idx, event in enumerate(events):
        if event.event_id == st.session_state["general_event_id"]:
            event_index = idx
            break

    previous_event_id = st.session_state["general_event_id"]
    selected_event = st.selectbox(
        "Event",
        options=events,
        index=event_index,
        format_func=lambda event: event.name,
        help="Choose a curated event. The planner model and course are selected automatically.",
    )
    selected_course = get_course_by_id(repo_root, selected_event.course_id)
    selected_template = get_event_template(selected_event.template_id)

    if selected_event.event_id != previous_event_id:
        st.session_state["general_event_id"] = selected_event.event_id
        st.session_state["general_course_id"] = selected_event.course_id
        st.session_state["general_config"] = _default_config_for_event(selected_event)
        st.session_state.pop("general_result", None)
        st.session_state.pop("general_selected_course", None)
        st.session_state.pop("general_loaded_course", None)
    else:
        st.session_state["general_course_id"] = selected_event.course_id

    overview_course = load_course_trackpoints(selected_course)
    total_gain_m, total_loss_m = elevation_changes(
        overview_course.trackpoints,
        0.0,
        overview_course.total_distance_km * 1000,
    )

    cfg = st.session_state["general_config"]
    race_model = selected_event.race_model
    cfg["race_model"] = race_model

    target_finish_time_min: float | None = None
    marathon_pace_min_km: float | None = None
    z1_pace_min_km: float | None = None
    z2_pace_min_km: float | None = None
    flat_pace_min_km: float | None = None
    hike_pace_min_km: float | None = None

    with st.sidebar:
        st.header("Planner Controls")
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
                    st.info(
                        "If this is a missing GPX, restore the curated course file in the repository and retry loading the plan."
                    )
                else:
                    st.session_state["general_course_id"] = updated_state["general_course_id"]
                    st.session_state["general_config"] = updated_state["general_config"]
                    st.success("Plan loaded. Review values and click Calculate.")

        st.markdown("### Event Setup")
        st.caption(
            f"Template: {selected_template.label} | Model: {selected_event.race_model} | Course: {selected_course.gpx_path.name}"
        )

        input_mode = st.radio(
            "Input Mode",
            options=["finish_time", "effort_anchor"],
            index=0
            if cfg.get("input_mode", selected_event.default_input_mode) == "finish_time"
            else 1,
            help="Finish-time derives base pace from target finish. Effort-anchor uses your known pace anchor.",
        )

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
            if input_mode == "finish_time":
                target_finish_time_min = st.number_input(
                    "Target Finish Time (minutes)",
                    min_value=30.0,
                    max_value=4000.0,
                    value=float(cfg.get("target_finish_time_min") or 720.0),
                    step=10.0,
                )
            else:
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
            if input_mode == "finish_time":
                target_finish_time_min = st.number_input(
                    "Target Finish Time (minutes)",
                    min_value=30.0,
                    max_value=4000.0,
                    value=float(cfg.get("target_finish_time_min") or 720.0),
                    step=10.0,
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

        st.markdown("### Strategy")
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
        rpe_target = st.number_input(
            "RPE Aggressiveness",
            min_value=1.0,
            max_value=10.0,
            value=float(cfg.get("rpe_target") or 6.0),
            step=0.5,
            help="Secondary policy input: lower is conservative, higher is aggressive.",
        )
        hr_cap = st.number_input(
            "HR Guardrail Cap",
            min_value=80,
            max_value=210,
            value=int(cfg.get("hr_cap") or 155),
            step=1,
            help="Safety ceiling used as a guardrail when validating pacing choices.",
        )
        pacing_bias = st.slider(
            "Pacing Bias",
            min_value=-10.0,
            max_value=10.0,
            value=float(cfg.get("pacing_bias", 0.0)),
            step=0.5,
            help="Negative values bias earlier aggression. Positive values bias later caution.",
        )
        rest_duration_sec = st.slider(
            "Rest Duration Per Aid Station (sec)",
            min_value=0,
            max_value=300,
            value=int(cfg.get("rest_duration_sec", 30)),
            step=15,
            help="Applied as fixed additive elapsed time at each aid station.",
        )

        calculate_plan_clicked = st.button(
            "Calculate Plan", type="primary", use_container_width=True
        )

    st.markdown("### Course Overview")
    st.caption(
        f"Template: {selected_template.label} | Model: {selected_event.race_model} | Course: {selected_course.gpx_path.name}"
    )
    overview_a, overview_b = st.columns(2)
    with overview_a:
        for row in _course_overview_rows(overview_course.total_distance_km, selected_event):
            st.markdown(f"**{row['label']}:** {row['value']}")
    with overview_b:
        st.markdown(f"**Elevation Gain / Loss:** +{total_gain_m:.0f}m / -{total_loss_m:.0f}m")
        if selected_course.aid_stations:
            st.caption(
                "Aid points: "
                + ", ".join(f"{aid.distance_km:.1f} km" for aid in selected_course.aid_stations)
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
        rest_duration_sec=rest_duration_sec,
        pacing_bias=pacing_bias,
        rpe_target=rpe_target,
        hr_cap=hr_cap,
    )
    st.session_state["general_config"] = asdict(new_config)

    if calculate_plan_clicked:
        loaded = load_course_trackpoints(selected_course)
        result = calculate_plan(loaded, new_config)
        st.session_state["general_result"] = result
        st.session_state["general_selected_course"] = selected_course
        st.session_state["general_loaded_course"] = loaded

    if "general_result" in st.session_state:
        result = st.session_state["general_result"]
        chosen_course = st.session_state["general_selected_course"]
        loaded_course = st.session_state.get("general_loaded_course", overview_course)
        st.subheader("Plan Output")
        summary_a, summary_b, summary_c, summary_d = st.columns(4)
        summary_a.metric("Distance", f"{result.total_distance_km:.2f} km")
        summary_b.metric("Elapsed", f"{result.total_time_min:.1f} min")
        summary_c.metric("Moving", f"{result.moving_time_min:.1f} min")
        summary_d.metric("Rest", f"{result.total_rest_time_min:.1f} min")
        if result.assumptions:
            st.caption("Assumptions: " + " | ".join(result.assumptions))
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

        tab_summary, tab_profile, tab_aid, tab_sections, tab_splits = st.tabs(
            ["Summary", "Course Profile", "Aid Stations", "Sections", "Splits"]
        )

        with tab_summary:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": "Terrain", "Value": selected_event.terrain.title()},
                        {"Metric": "Race Model", "Value": selected_event.race_model},
                        {"Metric": "Elevation Gain", "Value": f"+{total_gain_m:.0f}m"},
                        {"Metric": "Elevation Loss", "Value": f"-{total_loss_m:.0f}m"},
                        {"Metric": "Aid Stations", "Value": str(len(result.aid_station_etas))},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        with tab_profile:
            st.pyplot(_plot_course_profile(loaded_course.trackpoints, chosen_course.aid_stops_km))
            st.pyplot(_plot_pace_profile(result))

        with tab_aid:
            if result.aid_station_etas:
                st.markdown("#### Aid station timing")
                st.dataframe(
                    [
                        {
                            "label": aid_eta.label or f"Aid {idx + 1}",
                            "distance_km": round(aid_eta.distance_km, 2),
                            "arrival_elapsed_min": round(aid_eta.arrival_elapsed_time_min, 2),
                            "departure_elapsed_min": round(aid_eta.departure_elapsed_time_min, 2),
                            "split_min": round(aid_eta.split_from_prev_min, 2),
                            "split_pace": round(aid_eta.actual_pace_min_km, 2),
                            "rest_min": round(aid_eta.suggested_rest_min, 2),
                            "source": aid_eta.source,
                        }
                        for idx, aid_eta in enumerate(result.aid_station_etas)
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No in-range aid stations were available for this course.")

        with tab_sections:
            st.markdown("#### Segment pacing")
            st.dataframe(
                [
                    {
                        "section": segment.section_name or segment.segment_type,
                        "block": segment.block_label,
                        "type": segment.segment_type,
                        "start_km": round(segment.start_km, 2),
                        "end_km": round(segment.end_km, 2),
                        "distance_km": round(segment.distance_km, 2),
                        "elev_gain_m": round(segment.elevation_gain_m, 1),
                        "elev_loss_m": round(segment.elevation_loss_m, 1),
                        "start_min": round(segment.start_time_min, 2),
                        "end_min": round(segment.end_time_min, 2),
                        "avg_grade": round(segment.avg_grade_percent, 2),
                        "avg_pace": round(segment.avg_pace_min_km, 2),
                        "segment_min": round(segment.segment_time_min, 2),
                    }
                    for segment in result.segments
                ],
                width="stretch",
                hide_index=True,
            )

        with tab_splits:
            st.markdown("#### Kilometer pacing")
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

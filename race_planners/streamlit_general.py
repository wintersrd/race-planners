from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from race_planners.course_library import get_course_by_id
from race_planners.event_catalog import (
    get_curated_event_by_course_id,
    list_curated_events,
)
from race_planners.grade import elevation_changes
from race_planners.models import AthleteProfile, CuratedEvent, PacingConfig
from race_planners.plan_io import (
    ensure_gpx_exists_for_plan,
    export_plan_json,
    import_plan_json,
)
from race_planners.planner import calculate_plan, load_course_trackpoints


def _course_overview_rows(total_distance_km: float, event: CuratedEvent) -> list[dict[str, str]]:
    aid_mode = (
        f"{len(event.aid_stops_km)} configured"
        if event.aid_stops_km
        else "Derived from course file"
    )
    return [
        {"label": "Distance", "value": f"{total_distance_km:.2f} km"},
        {"label": "Terrain", "value": event.terrain.title()},
        {"label": "Aid Stations", "value": aid_mode},
    ]


FADE_PROFILE_PRESETS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "stable": ("Stable", (0.0, 0.75, 1.5)),
    "late_fade": ("Late Fade", (0.0, 1.25, 3.5)),
    "progressive_fade": ("Progressive Fade", (0.5, 2.0, 4.5)),
    "blow_up_risk": ("Blow-Up Risk", (1.5, 4.0, 7.0)),
}

EFFORT_POLICY_PRESETS: dict[str, tuple[str, float]] = {
    "conservative": ("Conservative", 4.5),
    "steady": ("Steady", 6.0),
    "aggressive": ("Aggressive", 7.5),
}


def _default_athlete_profile() -> dict[str, Any]:
    return asdict(AthleteProfile())


def _normalized_athlete_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _default_athlete_profile()
    if profile is None:
        return normalized
    for key in normalized:
        if key in profile:
            normalized[key] = profile[key]
    return normalized


def _is_road_event(event: CuratedEvent) -> bool:
    return event.race_model in {"half_marathon", "road_marathon"}


def _is_trail_event(event: CuratedEvent) -> bool:
    return not _is_road_event(event)


def _profile_text(profile: dict[str, Any], key: str) -> str:
    value = profile.get(key)
    return "" if value is None else str(value)


def _parse_optional_number(raw_value: str) -> float | None:
    stripped = raw_value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _fade_profile_values(config: dict[str, Any]) -> tuple[float, float, float]:
    if all(
        config.get(key) is not None
        for key in ("fade_early_bias", "fade_mid_bias", "fade_late_bias")
    ):
        return (
            float(config.get("fade_early_bias") or 0.0),
            float(config.get("fade_mid_bias") or 0.0),
            float(config.get("fade_late_bias") or 0.0),
        )

    preset_key = str(config.get("fade_profile_preset") or "stable")
    return FADE_PROFILE_PRESETS.get(preset_key, FADE_PROFILE_PRESETS["stable"])[1]


def _trail_anchor_defaults_from_profile(
    athlete_profile: dict[str, Any], race_model: str
) -> tuple[float | None, float | None]:
    lt1_pace_min_km = athlete_profile.get("lt1_pace_min_km")
    if lt1_pace_min_km is None:
        return None, None

    slowdown_key = (
        "technical_trail_slowdown_sec_km"
        if race_model == "technical_trail_ultra"
        else "flat_trail_slowdown_sec_km"
    )
    slowdown_sec = athlete_profile.get(slowdown_key)
    if slowdown_sec is None and race_model == "technical_trail_ultra":
        slowdown_sec = athlete_profile.get("flat_trail_slowdown_sec_km")

    flat_pace_min_km = float(lt1_pace_min_km) + (float(slowdown_sec or 0.0) / 60.0)
    hike_pace_min_km = flat_pace_min_km + 4.0
    return round(flat_pace_min_km, 2), round(hike_pace_min_km, 2)


def _road_anchor_default_from_profile(
    athlete_profile: dict[str, Any], race_model: str
) -> float | None:
    lt1_pace = athlete_profile.get("lt1_pace_min_km")
    if lt1_pace is None:
        return None

    lt1 = float(lt1_pace)
    lt2_pace = athlete_profile.get("lt2_pace_min_km")
    if race_model == "half_marathon" and lt2_pace is not None:
        lt2 = float(lt2_pace)
        return round((lt1 * 0.7) + (lt2 * 0.3), 2)

    return round(lt1, 2)


def _derived_effort_policy(profile: dict[str, Any]) -> str:
    return str(profile.get("default_trail_effort_policy") or "steady")


def _derived_fade_preset(profile: dict[str, Any]) -> str:
    return str(profile.get("default_trail_fade_preset") or "progressive_fade")


def _derived_split_bias(profile: dict[str, Any]) -> float:
    return float(profile.get("default_road_split_bias") or 0.0)


def _derived_hr_guardrail_cap(
    athlete_profile: dict[str, Any], race_model: str, effort_policy: str
) -> int | None:
    lt1_hr = athlete_profile.get("lt1_hr")
    lt2_hr = athlete_profile.get("lt2_hr")
    if lt1_hr is None or lt2_hr is None:
        return None

    lt1 = int(lt1_hr)
    lt2 = int(lt2_hr)
    delta = max(lt2 - lt1, 0)
    if race_model == "technical_trail_ultra":
        base = lt1 + (delta * 0.2)
    elif race_model == "fire_road_ultra":
        base = lt1 + (delta * 0.3)
    else:
        base = lt1 + (delta * 0.45)

    adjustment = {
        "conservative": -3,
        "steady": 0,
        "aggressive": 3,
    }.get(effort_policy, 0)
    return int(round(base + adjustment))


def _athlete_profile_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, sort_keys=True)


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
        "fade_profile_preset": "stable",
        "fade_early_bias": None,
        "fade_mid_bias": None,
        "fade_late_bias": None,
        "effort_policy": None,
        "use_hr_guardrail": False,
        "rpe_target": None,
        "hr_cap": None,
    }


def _default_config_for_event(
    event: CuratedEvent, athlete_profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    athlete_profile = athlete_profile or {}
    config = _default_config()
    config["race_model"] = event.race_model
    config["input_mode"] = event.default_input_mode

    if event.race_model == "half_marathon":
        config["target_finish_time_min"] = 105.0
        config["rest_duration_sec"] = 10
        config["marathon_pace_min_km"] = _road_anchor_default_from_profile(
            athlete_profile,
            event.race_model,
        )
        config["pacing_bias"] = _derived_split_bias(athlete_profile)
    elif event.race_model == "road_marathon":
        config["target_finish_time_min"] = 240.0
        config["rest_duration_sec"] = 15
        config["marathon_pace_min_km"] = _road_anchor_default_from_profile(
            athlete_profile,
            event.race_model,
        )
        config["pacing_bias"] = _derived_split_bias(athlete_profile)
    else:
        config["input_mode"] = "effort_anchor"
        config["target_finish_time_min"] = None
        config["rest_duration_sec"] = 180
        config["fade_profile_preset"] = _derived_fade_preset(athlete_profile)
        config["effort_policy"] = _derived_effort_policy(athlete_profile)
        config["use_hr_guardrail"] = (
            _derived_hr_guardrail_cap(
                athlete_profile,
                event.race_model,
                config["effort_policy"],
            )
            is not None
        )
        if event.race_model == "fire_road_ultra":
            config["z1_pace_min_km"] = athlete_profile.get("lt1_pace_min_km") or 8.0
            config["z2_pace_min_km"] = athlete_profile.get("lt2_pace_min_km") or 7.0
            config["hike_pace_min_km"] = 12.0
        else:
            flat_pace_min_km, hike_pace_min_km = _trail_anchor_defaults_from_profile(
                athlete_profile, event.race_model
            )
            config["flat_pace_min_km"] = flat_pace_min_km or 8.5
            config["hike_pace_min_km"] = hike_pace_min_km or 13.0

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
    if "athlete_profile" in payload:
        updated_state["general_athlete_profile"] = dict(payload["athlete_profile"])
    matched_event = get_curated_event_by_course_id(repo_root, str(payload["course_id"]))
    if matched_event is not None:
        updated_state["general_event_id"] = matched_event.event_id
    return updated_state, None


def render_general_planner(repo_root: Path) -> None:
    st.title("Unified Event Planner")
    st.caption("Choose a curated event and plan it through one event-first pacing flow.")

    st.session_state.setdefault("general_config", _default_config())
    st.session_state.setdefault("general_athlete_profile", _default_athlete_profile())
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
    selected_event: CuratedEvent

    selected_course: Any | None = None
    cfg = st.session_state["general_config"]
    athlete_profile = _normalized_athlete_profile(st.session_state["general_athlete_profile"])
    race_model = ""

    target_finish_time_min: float | None = None
    marathon_pace_min_km: float | None = None
    z1_pace_min_km: float | None = None
    z2_pace_min_km: float | None = None
    flat_pace_min_km: float | None = None
    hike_pace_min_km: float | None = None
    climb_hike_threshold_percent = float(cfg.get("climb_hike_threshold_percent", 12.0))
    descent_caution = str(cfg.get("descent_caution", "medium"))
    pacing_bias = float(cfg.get("pacing_bias", 0.0))
    fade_profile_preset = str(cfg.get("fade_profile_preset") or "stable")
    fade_early_bias, fade_mid_bias, fade_late_bias = _fade_profile_values(cfg)
    effort_policy = str(cfg.get("effort_policy") or _derived_effort_policy(athlete_profile))
    use_hr_guardrail = bool(cfg.get("use_hr_guardrail", False))
    derived_hr_cap: int | None = None
    rest_duration_sec = int(cfg.get("rest_duration_sec", 30))

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
                    if "general_event_id" in updated_state:
                        st.session_state["general_event_id"] = updated_state["general_event_id"]
                    if "general_athlete_profile" in updated_state:
                        st.session_state["general_athlete_profile"] = updated_state[
                            "general_athlete_profile"
                        ]
                    st.success("Plan loaded. Review values and click Calculate.")
                    cfg = st.session_state["general_config"]
                    athlete_profile = _normalized_athlete_profile(
                        st.session_state["general_athlete_profile"]
                    )

        selected_event = st.selectbox(
            "Event",
            options=events,
            index=event_index,
            format_func=lambda event: event.name,
            help="Choose a curated event. The planner model and course are selected automatically.",
        )

        selected_course = get_course_by_id(repo_root, selected_event.course_id)
        if selected_event.event_id != previous_event_id:
            st.session_state["general_event_id"] = selected_event.event_id
            st.session_state["general_course_id"] = selected_event.course_id
            st.session_state["general_config"] = _default_config_for_event(
                selected_event,
                athlete_profile,
            )
            st.session_state.pop("general_result", None)
            st.session_state.pop("general_selected_course", None)
            st.session_state.pop("general_loaded_course", None)
            cfg = st.session_state["general_config"]
        else:
            st.session_state["general_course_id"] = selected_event.course_id

        race_model = selected_event.race_model
        cfg["race_model"] = race_model

        st.markdown("### Event Setup")
        st.caption(
            "Road events focus on target pace and split shape. Trail and ultra events focus on terrain handling, fade, and aid-station time."
        )

        input_mode = st.radio(
            "Target Mode",
            options=["finish_time", "effort_anchor"],
            index=0
            if cfg.get("input_mode", selected_event.default_input_mode) == "finish_time"
            else 1,
            format_func=lambda mode: "Finish Time" if mode == "finish_time" else "Effort Anchor",
            help="Finish time derives a base plan from your goal time. Effort anchor uses your known pace anchor.",
        )

        if _is_road_event(selected_event):
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
                    "Anchor Pace (min/km)",
                    min_value=3.0,
                    max_value=20.0,
                    value=float(cfg.get("marathon_pace_min_km") or 5.5),
                    step=0.1,
                    help="Your realistic event anchor pace before split shaping is applied.",
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
                    help="Your conservative all-day runnable pace on smoother trail terrain.",
                )
                z2_pace_min_km = st.number_input(
                    "Z2 Pace (min/km)",
                    min_value=3.0,
                    max_value=20.0,
                    value=float(cfg.get("z2_pace_min_km") or 7.0),
                    step=0.1,
                    help="Your stronger but still sustainable pace when terrain and effort allow.",
                )
                hike_pace_min_km = st.number_input(
                    "Hike Pace (min/km)",
                    min_value=5.0,
                    max_value=40.0,
                    value=float(cfg.get("hike_pace_min_km") or 12.0),
                    step=0.1,
                    help="Expected pace once climbing becomes more efficient to hike than run.",
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
                    "Flat Trail Pace (min/km)",
                    min_value=4.0,
                    max_value=25.0,
                    value=float(cfg.get("flat_pace_min_km") or 8.5),
                    step=0.1,
                    help="Expected pace on runnable flat trail before fade and terrain penalties.",
                )
                hike_pace_min_km = st.number_input(
                    "Hike Pace (min/km)",
                    min_value=5.0,
                    max_value=40.0,
                    value=float(cfg.get("hike_pace_min_km") or 13.0),
                    step=0.1,
                    help="Expected uphill hiking pace once the climb threshold is exceeded.",
                )

        if _is_road_event(selected_event):
            st.markdown("### Race Strategy")
            pacing_bias = st.slider(
                "Split Bias",
                min_value=-10.0,
                max_value=10.0,
                value=float(cfg.get("pacing_bias", 0.0)),
                step=0.5,
                help="Negative values hold back a little early for a stronger finish. Positive values front-load the effort.",
            )
            rest_duration_sec = st.slider(
                "Aid Stop Time (sec)",
                min_value=0,
                max_value=90,
                value=int(cfg.get("rest_duration_sec", 10)),
                step=5,
                help="Optional slowdown per aid station for grabbing water or brief walking.",
            )
        else:
            st.markdown("### Terrain & Fade")
            climb_hike_threshold_percent = st.slider(
                "Climb-to-Hike Threshold (%)",
                min_value=5.0,
                max_value=25.0,
                value=float(cfg.get("climb_hike_threshold_percent", 12.0)),
                step=0.5,
                help="Grade where hiking becomes more efficient than running.",
            )
            if race_model == "technical_trail_ultra":
                descent_caution = st.selectbox(
                    "Descent Caution",
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(cfg.get("descent_caution", "medium")),
                    help="How conservatively to descend steep technical terrain.",
                )
            fade_profile_preset = st.selectbox(
                "Fade Profile",
                options=list(FADE_PROFILE_PRESETS.keys()),
                index=list(FADE_PROFILE_PRESETS.keys()).index(
                    str(cfg.get("fade_profile_preset") or "progressive_fade")
                ),
                format_func=lambda key: FADE_PROFILE_PRESETS[key][0],
                help="How much pace is expected to fade across the day. Presets are backed by early, mid, and late-race phase values.",
            )
            fade_early_bias, fade_mid_bias, fade_late_bias = FADE_PROFILE_PRESETS[
                fade_profile_preset
            ][1]
            st.caption(
                f"Fade phases: early {fade_early_bias:.1f}, mid {fade_mid_bias:.1f}, late {fade_late_bias:.1f}"
            )
            rest_duration_min = st.slider(
                "Aid Stop Time (min)",
                min_value=0.0,
                max_value=20.0,
                value=round(float(cfg.get("rest_duration_sec", 180)) / 60.0, 1),
                step=0.5,
                help="Expected stopped or near-stopped time at each aid station.",
            )
            rest_duration_sec = int(rest_duration_min * 60)

        with st.expander("Athlete Profile"):
            st.caption(
                "Optional runner baselines used to seed defaults and derive advanced trail effort behavior. Saved in plan JSON."
            )
            uploaded_profile_json = st.file_uploader(
                "Athlete Profile JSON",
                type=["json"],
                key="athlete_profile_json_uploader",
            )
            if uploaded_profile_json is not None:
                try:
                    athlete_profile = _normalized_athlete_profile(
                        json.loads(uploaded_profile_json.getvalue().decode("utf-8"))
                    )
                    st.session_state["general_athlete_profile"] = athlete_profile
                    st.success("Athlete profile loaded.")
                except json.JSONDecodeError:
                    st.error("Could not parse athlete profile JSON.")

            st.markdown("#### Road Baselines")
            lt1_hr_raw = st.text_input("LT1 HR", _profile_text(athlete_profile, "lt1_hr"))
            lt2_hr_raw = st.text_input("LT2 HR", _profile_text(athlete_profile, "lt2_hr"))
            lt1_hr_value = _parse_optional_number(lt1_hr_raw)
            lt2_hr_value = _parse_optional_number(lt2_hr_raw)
            athlete_profile["lt1_hr"] = int(lt1_hr_value) if lt1_hr_value is not None else None
            athlete_profile["lt2_hr"] = int(lt2_hr_value) if lt2_hr_value is not None else None
            athlete_profile["lt1_pace_min_km"] = _parse_optional_number(
                st.text_input(
                    "LT1 Road Pace (min/km)",
                    _profile_text(athlete_profile, "lt1_pace_min_km"),
                    help="Aerobic threshold pace on runnable road terrain.",
                )
            )
            athlete_profile["lt2_pace_min_km"] = _parse_optional_number(
                st.text_input(
                    "LT2 Road Pace (min/km)",
                    _profile_text(athlete_profile, "lt2_pace_min_km"),
                    help="Threshold pace on runnable road terrain.",
                )
            )

            st.markdown("#### Trail Adjustments")
            athlete_profile["flat_trail_slowdown_sec_km"] = _parse_optional_number(
                st.text_input(
                    "Flat Trail Slowdown vs Road (sec/km)",
                    _profile_text(athlete_profile, "flat_trail_slowdown_sec_km"),
                    help="How much slower flat trail is for you compared with road pace.",
                )
            )
            athlete_profile["technical_trail_slowdown_sec_km"] = _parse_optional_number(
                st.text_input(
                    "Technical Trail Slowdown vs Road (sec/km)",
                    _profile_text(athlete_profile, "technical_trail_slowdown_sec_km"),
                    help="Extra slowdown on technical trail compared with road pace.",
                )
            )

            st.markdown("#### Preferences")
            athlete_profile["default_road_split_bias"] = st.slider(
                "Default Road Split Bias",
                min_value=-10.0,
                max_value=10.0,
                value=float(athlete_profile.get("default_road_split_bias") or 0.0),
                step=0.5,
                help="Saved default split tendency for road events.",
            )
            athlete_profile["default_trail_fade_preset"] = st.selectbox(
                "Default Trail Fade Profile",
                options=list(FADE_PROFILE_PRESETS.keys()),
                index=list(FADE_PROFILE_PRESETS.keys()).index(
                    _derived_fade_preset(athlete_profile)
                ),
                format_func=lambda key: FADE_PROFILE_PRESETS[key][0],
            )
            athlete_profile["default_trail_effort_policy"] = st.selectbox(
                "Default Trail Effort Policy",
                options=list(EFFORT_POLICY_PRESETS.keys()),
                index=list(EFFORT_POLICY_PRESETS.keys()).index(
                    _derived_effort_policy(athlete_profile)
                ),
                format_func=lambda key: EFFORT_POLICY_PRESETS[key][0],
            )

            st.download_button(
                "Download Athlete Profile JSON",
                data=_athlete_profile_json(athlete_profile),
                file_name="athlete-profile.json",
                mime="application/json",
                use_container_width=True,
            )
            if st.button("Apply Profile Defaults To This Event", use_container_width=True):
                st.session_state["general_athlete_profile"] = athlete_profile
                st.session_state["general_config"] = _default_config_for_event(
                    selected_event,
                    athlete_profile,
                )
                st.session_state.pop("general_result", None)
                st.session_state.pop("general_selected_course", None)
                st.session_state.pop("general_loaded_course", None)
                st.rerun()

        if _is_trail_event(selected_event):
            with st.expander("Advanced Trail Controls"):
                effort_policy = st.selectbox(
                    "Effort Policy",
                    options=list(EFFORT_POLICY_PRESETS.keys()),
                    index=list(EFFORT_POLICY_PRESETS.keys()).index(
                        str(cfg.get("effort_policy") or _derived_effort_policy(athlete_profile))
                    ),
                    format_func=lambda key: EFFORT_POLICY_PRESETS[key][0],
                    help="High-level pacing attitude for the day. Conservative protects later durability, aggressive leans into earlier pace.",
                )
                derived_hr_cap = _derived_hr_guardrail_cap(
                    athlete_profile,
                    race_model,
                    effort_policy,
                )
                if derived_hr_cap is None:
                    use_hr_guardrail = False
                    st.info(
                        "Add LT1 and LT2 heart-rate values in the athlete profile to enable a derived HR guardrail."
                    )
                else:
                    use_hr_guardrail = st.checkbox(
                        "Use Derived HR Guardrail",
                        value=bool(cfg.get("use_hr_guardrail", True)),
                        help="Uses your LT1/LT2 profile to temper pacing on steep or late-race trail segments.",
                    )
                    st.caption(f"Derived guardrail cap for this event: {derived_hr_cap} bpm")

        calculate_plan_clicked = st.button(
            "Calculate Plan", type="primary", use_container_width=True
        )

    if selected_course is None:
        st.error("Could not resolve the selected event.")
        return

    overview_course = load_course_trackpoints(selected_course)
    total_gain_m, total_loss_m = elevation_changes(
        overview_course.trackpoints,
        0.0,
        overview_course.total_distance_km * 1000,
    )

    st.markdown("### Course Overview")
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
        fade_profile_preset=None if _is_road_event(selected_event) else fade_profile_preset,
        fade_early_bias=None if _is_road_event(selected_event) else fade_early_bias,
        fade_mid_bias=None if _is_road_event(selected_event) else fade_mid_bias,
        fade_late_bias=None if _is_road_event(selected_event) else fade_late_bias,
        effort_policy=None if _is_road_event(selected_event) else effort_policy,
        use_hr_guardrail=False if _is_road_event(selected_event) else use_hr_guardrail,
        rpe_target=(
            None if _is_road_event(selected_event) else EFFORT_POLICY_PRESETS[effort_policy][1]
        ),
        hr_cap=(None if _is_road_event(selected_event) or not use_hr_guardrail else derived_hr_cap),
    )
    st.session_state["general_config"] = asdict(new_config)
    st.session_state["general_athlete_profile"] = _normalized_athlete_profile(athlete_profile)

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
            athlete_profile=st.session_state["general_athlete_profile"],
        )
        st.download_button(
            "Download plan JSON",
            data=plan_json,
            file_name="race-plan.json",
            mime="application/json",
        )

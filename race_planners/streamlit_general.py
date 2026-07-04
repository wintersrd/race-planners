from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from race_planners.course_library import get_course_by_id
from race_planners.event_catalog import list_curated_events
from race_planners.formatting import format_clock_time, format_duration_minutes, format_pace_minutes
from race_planners.fueling import build_fueling_plan
from race_planners.grade import elevation_changes
from race_planners.models import CuratedEvent, PacingConfig
from race_planners.plots import plot_course_profile, plot_pace_profile
from race_planners.plan_io import export_plan_json, load_plan_into_state
from race_planners.planner import calculate_plan, load_course_trackpoints
from race_planners.profile import (
    EFFORT_POLICY_PRESETS,
    FADE_PROFILE_PRESETS,
    ROAD_INTENT_PRESETS,
    athlete_profile_json,
    default_athlete_profile,
    default_config,
    default_config_for_event,
    derived_effort_policy,
    derived_fade_preset,
    derived_hr_guardrail_cap,
    fade_profile_values,
    is_road_event,
    is_trail_event,
    normalized_athlete_profile,
    parse_optional_number,
    profile_text,
    road_capability_sources,
    selected_road_capability,
)
from race_planners.road_capability import (
    classify_road_effort_band,
    classify_road_feasibility,
    classify_road_recovery_cost,
    estimate_road_adjusted_best_likely,
    estimate_road_intent_target_time_min,
    road_race_distance_km,
)
from race_planners.splits import (
    aggregate_split_rows,
    course_overview_rows,
    default_split_block_size,
    split_block_options,
)


_AID_TIER_LABELS: dict[str, str] = {
    "none": "—",
    "water_only": "Water",
    "standard": "Water + snacks",
    "full_service": "Full food (race or drop bag)",
}


def _aid_tier_label(tier: str) -> str:
    return _AID_TIER_LABELS.get(tier, tier)


def render_general_planner(repo_root: Path) -> None:
    st.title("Unified Event Planner")
    st.caption("Choose a curated event and plan it through one event-first pacing flow.")

    st.session_state.setdefault("general_config", default_config())
    st.session_state.setdefault("general_athlete_profile", default_athlete_profile())
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
    athlete_profile = normalized_athlete_profile(st.session_state["general_athlete_profile"])
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
    fade_early_bias, fade_mid_bias, fade_late_bias = fade_profile_values(cfg)
    race_intent = str(cfg.get("race_intent") or "controlled")
    effort_policy = str(cfg.get("effort_policy") or derived_effort_policy(athlete_profile))
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
                    athlete_profile = normalized_athlete_profile(
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
            st.session_state["general_config"] = default_config_for_event(
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

        if is_road_event(selected_event):
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

        if is_road_event(selected_event):
            st.markdown("### Race Strategy")
            race_intent = st.selectbox(
                "Race Intent",
                options=list(ROAD_INTENT_PRESETS.keys()),
                index=list(ROAD_INTENT_PRESETS.keys()).index(
                    str(cfg.get("race_intent") or "controlled")
                ),
                format_func=lambda key: ROAD_INTENT_PRESETS[key],
                help="How hard you intend to race relative to your event-adjusted best-likely result.",
            )
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

        st.markdown("### Weather")
        peak_temperature_c = st.number_input(
            "Expected Peak Temperature (°C)",
            min_value=0.0,
            max_value=45.0,
            value=float(
                cfg.get(
                    "peak_temperature_c",
                    selected_event.baseline_peak_temp_c or 18.0,
                )
            ),
            step=1.0,
            help="Peak temperature for the event. The planner applies a diurnal temperature curve and slows pace when it's hot.",
        )

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
                    athlete_profile = normalized_athlete_profile(
                        json.loads(uploaded_profile_json.getvalue().decode("utf-8"))
                    )
                    st.session_state["general_athlete_profile"] = athlete_profile
                    st.success("Athlete profile loaded.")
                except json.JSONDecodeError:
                    st.error("Could not parse athlete profile JSON.")

            st.markdown("#### Road Baselines")
            lt1_hr_raw = st.text_input("LT1 HR", profile_text(athlete_profile, "lt1_hr"))
            lt2_hr_raw = st.text_input("LT2 HR", profile_text(athlete_profile, "lt2_hr"))
            lt1_hr_value = parse_optional_number(lt1_hr_raw)
            lt2_hr_value = parse_optional_number(lt2_hr_raw)
            athlete_profile["lt1_hr"] = int(lt1_hr_value) if lt1_hr_value is not None else None
            athlete_profile["lt2_hr"] = int(lt2_hr_value) if lt2_hr_value is not None else None
            athlete_profile["lt1_pace_min_km"] = parse_optional_number(
                st.text_input(
                    "LT1 Road Pace (min/km)",
                    profile_text(athlete_profile, "lt1_pace_min_km"),
                    help="Aerobic threshold pace on runnable road terrain.",
                )
            )
            athlete_profile["lt2_pace_min_km"] = parse_optional_number(
                st.text_input(
                    "LT2 Road Pace (min/km)",
                    profile_text(athlete_profile, "lt2_pace_min_km"),
                    help="Threshold pace on runnable road terrain.",
                )
            )

            st.markdown("#### Road Capability")
            st.caption(
                "Manual best-likely values outrank predictor values. Predictor values outrank LT-derived modeled estimates."
            )
            athlete_profile["best_likely_half_time_min"] = parse_optional_number(
                st.text_input(
                    "Best Likely Half Marathon (ideal min)",
                    profile_text(athlete_profile, "best_likely_half_time_min"),
                    help="Manual best-likely half-marathon result in neutral conditions. This will outrank derived estimates later.",
                )
            )
            athlete_profile["best_likely_marathon_time_min"] = parse_optional_number(
                st.text_input(
                    "Best Likely Marathon (ideal min)",
                    profile_text(athlete_profile, "best_likely_marathon_time_min"),
                    help="Manual best-likely marathon result in neutral conditions. This will outrank derived estimates later.",
                )
            )
            athlete_profile["predictor_half_time_min"] = parse_optional_number(
                st.text_input(
                    "Predictor Half Marathon (ideal min)",
                    profile_text(athlete_profile, "predictor_half_time_min"),
                    help="Optional external estimate such as COROS, Strava, or Intervals.icu.",
                )
            )
            athlete_profile["predictor_marathon_time_min"] = parse_optional_number(
                st.text_input(
                    "Predictor Marathon (ideal min)",
                    profile_text(athlete_profile, "predictor_marathon_time_min"),
                    help="Optional external marathon estimate from a trusted predictor source.",
                )
            )
            predictor_source = st.text_input(
                "Predictor Source",
                profile_text(athlete_profile, "predictor_source"),
                help="Short source label such as COROS, Strava, Intervals.icu, or coach estimate.",
            ).strip()
            athlete_profile["predictor_source"] = predictor_source or None

            st.markdown("#### Trail Adjustments")
            athlete_profile["flat_trail_slowdown_sec_km"] = parse_optional_number(
                st.text_input(
                    "Flat Trail Slowdown vs Road (sec/km)",
                    profile_text(athlete_profile, "flat_trail_slowdown_sec_km"),
                    help="How much slower flat trail is for you compared with road pace.",
                )
            )
            athlete_profile["technical_trail_slowdown_sec_km"] = parse_optional_number(
                st.text_input(
                    "Technical Trail Slowdown vs Road (sec/km)",
                    profile_text(athlete_profile, "technical_trail_slowdown_sec_km"),
                    help="Extra slowdown on technical trail compared with road pace.",
                )
            )

            st.markdown("#### Universal Factors")
            st.caption(
                "These factors influence both road and trail events. Positive values help you resist that cost; negative values amplify it."
            )
            athlete_profile["body_mass_kg"] = parse_optional_number(
                st.text_input(
                    "Body Mass (kg)",
                    profile_text(athlete_profile, "body_mass_kg"),
                    help="Used for calorie and fueling calculations.",
                )
            )
            athlete_profile["sweat_rate_l_hr"] = parse_optional_number(
                st.text_input(
                    "Sweat Rate (L/hr)",
                    profile_text(athlete_profile, "sweat_rate_l_hr"),
                    help="Optional. If blank, the planner estimates from temperature and intensity.",
                )
            )
            athlete_profile["gut_carb_tolerance_g_hr"] = parse_optional_number(
                st.text_input(
                    "Gut Carb Tolerance (g/hr)",
                    profile_text(athlete_profile, "gut_carb_tolerance_g_hr"),
                    help="Optional. Maximum carbs you can absorb per hour. If blank, planner uses event-duration defaults.",
                )
            )
            athlete_profile["durability_factor"] = st.slider(
                "Durability Factor",
                min_value=-1.0,
                max_value=1.0,
                value=float(athlete_profile.get("durability_factor") or 0.0),
                step=0.1,
                help="General late-race resilience. Higher means you hold pace better deep into long events.",
            )
            athlete_profile["heat_tolerance"] = st.slider(
                "Heat Tolerance",
                min_value=-1.0,
                max_value=1.0,
                value=float(athlete_profile.get("heat_tolerance") or 0.0),
                step=0.1,
                help="How well you cope with warmer conditions across road and trail events.",
            )
            athlete_profile["hill_tolerance"] = st.slider(
                "Hill Tolerance",
                min_value=-1.0,
                max_value=1.0,
                value=float(athlete_profile.get("hill_tolerance") or 0.0),
                step=0.1,
                help="How well you convert fitness into performance on rolling or hilly courses.",
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
                index=list(FADE_PROFILE_PRESETS.keys()).index(derived_fade_preset(athlete_profile)),
                format_func=lambda key: FADE_PROFILE_PRESETS[key][0],
            )
            athlete_profile["default_trail_effort_policy"] = st.selectbox(
                "Default Trail Effort Policy",
                options=list(EFFORT_POLICY_PRESETS.keys()),
                index=list(EFFORT_POLICY_PRESETS.keys()).index(
                    derived_effort_policy(athlete_profile)
                ),
                format_func=lambda key: EFFORT_POLICY_PRESETS[key][0],
            )

            st.download_button(
                "Download Athlete Profile JSON",
                data=athlete_profile_json(athlete_profile),
                file_name="athlete-profile.json",
                mime="application/json",
                use_container_width=True,
            )
            if st.button("Apply Profile Defaults To This Event", use_container_width=True):
                st.session_state["general_athlete_profile"] = athlete_profile
                st.session_state["general_config"] = default_config_for_event(
                    selected_event,
                    athlete_profile,
                )
                st.session_state.pop("general_result", None)
                st.session_state.pop("general_selected_course", None)
                st.session_state.pop("general_loaded_course", None)
                st.rerun()

        if is_trail_event(selected_event):
            with st.expander("Advanced Trail Controls"):
                effort_policy = st.selectbox(
                    "Effort Policy",
                    options=list(EFFORT_POLICY_PRESETS.keys()),
                    index=list(EFFORT_POLICY_PRESETS.keys()).index(
                        str(cfg.get("effort_policy") or derived_effort_policy(athlete_profile))
                    ),
                    format_func=lambda key: EFFORT_POLICY_PRESETS[key][0],
                    help="High-level pacing attitude for the day. Conservative protects later durability, aggressive leans into earlier pace.",
                )
                derived_hr_cap = derived_hr_guardrail_cap(
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
        for row in course_overview_rows(overview_course.total_distance_km, selected_event):
            st.markdown(f"**{row['label']}:** {row['value']}")
    with overview_b:
        st.markdown(f"**Elevation Gain / Loss:** +{total_gain_m:.0f}m / -{total_loss_m:.0f}m")
        if selected_course.aid_stations:
            st.caption(
                "Aid points: "
                + ", ".join(f"{aid.distance_km:.1f} km" for aid in selected_course.aid_stations)
            )

    if is_road_event(selected_event):
        capability_rows = road_capability_sources(athlete_profile, race_model)
        selected_capability_time_min, selected_capability_source = selected_road_capability(
            athlete_profile,
            race_model,
        )
        st.markdown("### Road Capability")
        st.caption(
            "Best-likely road capability uses manual profile values first, then predictor values, then LT-derived estimates."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Source": row["source"],
                        "Best Likely": format_duration_minutes(
                            float(row["time_min"]) if row["time_min"] is not None else None
                        ),
                        "Selected": "Yes" if row["selected"] else "",
                    }
                    for row in capability_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if selected_capability_time_min is not None and selected_capability_source is not None:
            st.caption(
                f"Selected capability source: {selected_capability_source} ({format_duration_minutes(selected_capability_time_min)})"
            )
            adjusted_capability = estimate_road_adjusted_best_likely(
                overview_course,
                selected_capability_time_min,
                peak_temperature_c=float(
                    cfg.get("peak_temperature_c", selected_event.baseline_peak_temp_c or 18.0)
                ),
                start_time_local=selected_event.start_time_local,
                hill_tolerance=athlete_profile.get("hill_tolerance"),
                heat_tolerance=athlete_profile.get("heat_tolerance"),
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Metric": "Selected Best Likely",
                            "Value": format_duration_minutes(adjusted_capability["base_time_min"]),
                        },
                        {
                            "Metric": "Course Impact",
                            "Value": f"x{adjusted_capability['course_multiplier']:.3f}",
                        },
                        {
                            "Metric": "Weather Impact",
                            "Value": f"x{adjusted_capability['weather_multiplier']:.3f}",
                        },
                        {
                            "Metric": "Adjusted Best Likely",
                            "Value": format_duration_minutes(
                                adjusted_capability["adjusted_time_min"]
                            ),
                        },
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            chosen_target_time_min: float | None = None
            race_distance_km = road_race_distance_km(race_model)
            if input_mode == "finish_time":
                chosen_target_time_min = target_finish_time_min
            elif marathon_pace_min_km is not None and race_distance_km is not None:
                chosen_target_time_min = marathon_pace_min_km * race_distance_km

            suggested_target_time_min = estimate_road_intent_target_time_min(
                adjusted_capability["adjusted_time_min"],
                race_intent,
            )
            previous_race_intent = str(cfg.get("race_intent") or "controlled")
            if input_mode == "finish_time" and race_intent != previous_race_intent:
                target_finish_time_min = suggested_target_time_min
            st.caption(
                "Adjusted best likely applies course and weather costs, scaled by hill and heat tolerance from the athlete profile."
            )
            if chosen_target_time_min is not None:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Metric": "Race Intent",
                                "Value": ROAD_INTENT_PRESETS[race_intent],
                            },
                            {
                                "Metric": "Intent Suggested Target",
                                "Value": format_duration_minutes(suggested_target_time_min),
                            },
                            {
                                "Metric": "Chosen Target",
                                "Value": format_duration_minutes(chosen_target_time_min),
                            },
                            {
                                "Metric": "Feasibility",
                                "Value": classify_road_feasibility(
                                    adjusted_capability["adjusted_time_min"],
                                    chosen_target_time_min,
                                ),
                            },
                            {
                                "Metric": "Expected Effort",
                                "Value": classify_road_effort_band(
                                    adjusted_capability["adjusted_time_min"],
                                    chosen_target_time_min,
                                ),
                            },
                            {
                                "Metric": "Recovery Cost",
                                "Value": classify_road_recovery_cost(
                                    adjusted_capability["adjusted_time_min"],
                                    chosen_target_time_min,
                                ),
                            },
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                with st.expander("How road planning works"):
                    st.markdown(
                        "- **Selected Best Likely** is the capability source that currently wins by precedence: manual profile, then predictor, then LT-derived model.\n"
                        "- **Adjusted Best Likely** applies course and weather costs to that capability for the selected event.\n"
                        "- **Race Intent** shifts the suggested target away from the adjusted best-likely result without changing the underlying capability estimate.\n"
                        "- **Chosen Target** is whatever target you currently entered. The planner compares it to adjusted best likely to estimate feasibility, effort, and recovery cost."
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
        fade_profile_preset=None if is_road_event(selected_event) else fade_profile_preset,
        fade_early_bias=None if is_road_event(selected_event) else fade_early_bias,
        fade_mid_bias=None if is_road_event(selected_event) else fade_mid_bias,
        fade_late_bias=None if is_road_event(selected_event) else fade_late_bias,
        race_intent=race_intent if is_road_event(selected_event) else None,
        effort_policy=None if is_road_event(selected_event) else effort_policy,
        use_hr_guardrail=False if is_road_event(selected_event) else use_hr_guardrail,
        athlete_lt1_hr=athlete_profile.get("lt1_hr"),
        athlete_lt2_hr=athlete_profile.get("lt2_hr"),
        athlete_lt1_pace_min_km=athlete_profile.get("lt1_pace_min_km"),
        athlete_lt2_pace_min_km=athlete_profile.get("lt2_pace_min_km"),
        athlete_flat_trail_slowdown_sec_km=athlete_profile.get("flat_trail_slowdown_sec_km"),
        athlete_technical_trail_slowdown_sec_km=athlete_profile.get(
            "technical_trail_slowdown_sec_km"
        ),
        athlete_durability_factor=athlete_profile.get("durability_factor"),
        athlete_heat_tolerance=athlete_profile.get("heat_tolerance"),
        athlete_hill_tolerance=athlete_profile.get("hill_tolerance"),
        rpe_target=(
            None if is_road_event(selected_event) else EFFORT_POLICY_PRESETS[effort_policy][1]
        ),
        hr_cap=(None if is_road_event(selected_event) or not use_hr_guardrail else derived_hr_cap),
        peak_temperature_c=peak_temperature_c,
        event_start_time_local=selected_event.start_time_local,
    )
    st.session_state["general_config"] = asdict(new_config)
    st.session_state["general_athlete_profile"] = normalized_athlete_profile(athlete_profile)

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
        average_pace_min_km = (
            result.moving_time_min / result.total_distance_km
            if result.total_distance_km > 0
            else None
        )
        summary_a, summary_b, summary_c, summary_d, summary_e = st.columns(5)
        summary_a.metric("Distance", f"{result.total_distance_km:.2f} km")
        summary_b.metric("Elapsed", format_duration_minutes(result.total_time_min))
        summary_c.metric("Moving", format_duration_minutes(result.moving_time_min))
        summary_d.metric("Rest", format_duration_minutes(result.total_rest_time_min))
        summary_e.metric("Avg Pace", format_pace_minutes(average_pace_min_km))
        if result.assumptions:
            st.caption("Assumptions: " + " | ".join(result.assumptions))
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

        tab_summary, tab_profile, tab_aid, tab_sections, tab_fueling, tab_splits = st.tabs(
            ["Summary", "Course Profile", "Aid Stations", "Sections", "Fueling", "Splits"]
        )

        with tab_summary:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": "Terrain", "Value": selected_event.terrain.title()},
                        {"Metric": "Elevation Gain", "Value": f"+{total_gain_m:.0f}m"},
                        {"Metric": "Elevation Loss", "Value": f"-{total_loss_m:.0f}m"},
                        {"Metric": "Aid Stations", "Value": str(len(result.aid_station_etas))},
                        {
                            "Metric": "Estimated Finish",
                            "Value": format_clock_time(selected_event, result.total_time_min),
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        with tab_profile:
            st.pyplot(plot_course_profile(loaded_course.trackpoints, chosen_course.aid_stops_km))
            st.pyplot(plot_pace_profile(result))

        with tab_aid:
            if result.aid_station_etas:
                st.markdown("#### Aid station timing")
                st.dataframe(
                    [
                        {
                            "label": aid_eta.label or f"Aid {idx + 1}",
                            "distance_km": round(aid_eta.distance_km, 2),
                            "arrival_elapsed": format_duration_minutes(
                                aid_eta.arrival_elapsed_time_min
                            ),
                            "arrival_clock": format_clock_time(
                                selected_event,
                                aid_eta.arrival_elapsed_time_min,
                            ),
                            "departure_elapsed": format_duration_minutes(
                                aid_eta.departure_elapsed_time_min
                            ),
                            "departure_clock": format_clock_time(
                                selected_event,
                                aid_eta.departure_elapsed_time_min,
                            ),
                            "split_time": format_duration_minutes(aid_eta.split_from_prev_min),
                            "split_pace": format_pace_minutes(aid_eta.actual_pace_min_km),
                            "rest_time": format_duration_minutes(aid_eta.suggested_rest_min),
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
                        "start_elapsed": format_duration_minutes(segment.start_time_min),
                        "end_elapsed": format_duration_minutes(segment.end_time_min),
                        "start_clock": format_clock_time(selected_event, segment.start_time_min),
                        "end_clock": format_clock_time(selected_event, segment.end_time_min),
                        "avg_grade": round(segment.avg_grade_percent, 2),
                        "avg_pace": format_pace_minutes(segment.avg_pace_min_km),
                        "segment_time": format_duration_minutes(segment.segment_time_min),
                    }
                    for segment in result.segments
                ],
                width="stretch",
                hide_index=True,
            )

        with tab_fueling:
            body_mass_kg = athlete_profile.get("body_mass_kg")
            if body_mass_kg is None:
                st.info("Add your body mass in the Athlete Profile to generate a fueling plan.")
            else:
                aid_tiers = [station.tier for station in selected_course.aid_stations]
                fueling_plan = build_fueling_plan(
                    result,
                    mass_kg=float(body_mass_kg),
                    peak_temperature_c=peak_temperature_c,
                    aid_station_tiers=aid_tiers,
                    athlete_sweat_rate=athlete_profile.get("sweat_rate_l_hr"),
                    gut_carb_tolerance_g_hr=athlete_profile.get("gut_carb_tolerance_g_hr"),
                )
                f_sum_a, f_sum_b, f_sum_c, f_sum_d = st.columns(4)
                f_sum_a.metric("Total kcal", f"{fueling_plan.total_kcal:.0f}")
                f_sum_b.metric("Avg kcal/hr", f"{fueling_plan.avg_kcal_hr:.0f}")
                f_sum_c.metric("Carb Target", f"{fueling_plan.total_carb_target_g:.0f}g")
                f_sum_d.metric("Fluid Target", f"{fueling_plan.total_fluid_target_l:.1f}L")
                if fueling_plan.warnings:
                    for warning in fueling_plan.warnings:
                        st.warning(warning)
                st.markdown("#### Per-block fueling")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "block": f"{block.start_km:.1f}-{block.end_km:.1f} km",
                                "tier": _aid_tier_label(block.aid_station_tier),
                                "duration": format_duration_minutes(block.duration_min),
                                "kcal": round(block.kcal_burned),
                                "carb_target_g": round(block.carb_target_g),
                                "carb_planned_g": round(block.carb_planned_g),
                                "on_site_kcal": round(block.on_site_kcal),
                                "on_site_carb_g": round(block.on_site_carb_g),
                                "fluid_l": block.fluid_target_l,
                                "deficit_g": round(block.cumulative_carb_deficit_g),
                                "carry": "; ".join(block.carry_items) if block.carry_items else "-",
                            }
                            for block in fueling_plan.blocks
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if fueling_plan.carb_deficit_g > 0:
                    st.caption(
                        f"Planned carb intake is {fueling_plan.carb_deficit_g:.0f}g below target. "
                        "Consider increasing fueling frequency, especially early in the event."
                    )

        with tab_splits:
            block_options = split_block_options(result.total_distance_km)
            default_split_block = default_split_block_size(result.total_distance_km)
            block_index = block_options.index(default_split_block)
            split_block_km = st.selectbox(
                "Split block size (km)",
                options=block_options,
                index=block_index,
                help="Use larger blocks for longer events so pacing is easier to reason about.",
            )
            st.markdown("#### Split pacing")
            st.dataframe(
                aggregate_split_rows(
                    result.splits,
                    loaded_course.trackpoints,
                    selected_event,
                    split_block_km,
                ),
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

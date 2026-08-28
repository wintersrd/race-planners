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
from race_planners.i18n import AVAILABLE_LOCALES, DEFAULT_LOCALE, t
from race_planners.models import CuratedEvent, PacingConfig
from race_planners.plots import (
    plot_course_profile,
    plot_cumulative_time,
    plot_half_comparison,
    plot_heat_impact,
    plot_pace_profile,
    plot_temperature_curve,
    plot_terrain_breakdown,
)
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
from race_planners.weather import average_heat_multiplier_for_duration
from race_planners.splits import (
    aggregate_split_rows,
    course_overview_rows,
    default_split_block_size,
    split_block_options,
)


def _aid_tier_label(tier: str, locale: str) -> str:
    return t(f"tier.{tier}", locale)


def _translate_message(message: str, locale: str) -> str:
    """Translate a domain-layer message key with optional inline kwargs.

    Domain messages use the format ``key|arg1=val1,arg2=val2``.
    Simple messages are just ``key``.
    """
    if "|" in message:
        key, args_str = message.split("|", 1)
        kwargs: dict[str, float] = {}
        for pair in args_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    kwargs[k.strip()] = float(v.strip())
                except ValueError:
                    kwargs[k.strip()] = 0.0
        return t(key, locale, **kwargs)
    return t(message, locale)


def _read_markdown_doc(repo_root: Path, relative_path: str, locale: str) -> str:
    doc_path = repo_root / relative_path
    if not doc_path.exists():
        return t("sidebar.doc_missing", locale, path=relative_path)
    return doc_path.read_text(encoding="utf-8")


def _translate_source_label(source: str, locale: str) -> str:
    source_key_map = {
        "Manual Profile": "source.manual_profile",
        "Predictor": "source.predictor",
        "LT-Derived Model": "source.lt_derived",
    }
    return t(source_key_map.get(source, source), locale)


def _pace_band_label(
    pace_min_km: float | None, reference_pace_min_km: float | None, locale: str
) -> str:
    if pace_min_km is None or reference_pace_min_km is None or reference_pace_min_km <= 0:
        return "-"
    ratio = pace_min_km / reference_pace_min_km
    if ratio < 0.94:
        return t("pace_band.fast", locale)
    if ratio <= 1.03:
        return t("pace_band.on_target", locale)
    if ratio <= 1.12:
        return t("pace_band.costly", locale)
    return t("pace_band.high_risk", locale)


def _road_split_story(pacing_bias: float, locale: str) -> str:
    if pacing_bias <= -1.5:
        return t("story.road_split.hold_back", locale)
    if pacing_bias >= 1.5:
        return t("story.road_split.front_load", locale)
    return t("story.road_split.even", locale)


def _section_cue(segment_type: str, locale: str) -> str:
    return t(f"section.cue.{segment_type}", locale)


def _aid_station_action(tier: str, locale: str) -> str:
    return t(f"aid.action.{tier}", locale)


def _fueling_block_action(block: Any, locale: str) -> tuple[str, str, str]:
    carry = "; ".join(block.carry_items) if block.carry_items else t("fueling.action.none", locale)
    station = _aid_tier_label(block.aid_station_tier, locale)
    fluid = f"{block.fluid_target_l:.1f}L"
    return carry, station, fluid


def _split_pace_display_to_minutes(pace_display: str) -> float | None:
    parts = pace_display.split(":")
    if len(parts) != 2:
        return None
    try:
        mins = int(parts[0])
        secs = int(parts[1])
    except ValueError:
        return None
    return mins + secs / 60.0


def _road_story_lines(
    locale: str,
    race_intent: str,
    pacing_bias: float,
    adjusted_best_likely_time_min: float | None,
    chosen_target_time_min: float | None,
) -> list[str]:
    lines = [
        t("story.road_intro", locale, intent=t(f"preset.intent.{race_intent}", locale).lower())
    ]
    if adjusted_best_likely_time_min is not None and chosen_target_time_min is not None:
        feasibility = classify_road_feasibility(
            adjusted_best_likely_time_min, chosen_target_time_min
        )
        lines.append(t("story.road_target", locale, feasibility=t(feasibility, locale).lower()))
    lines.append(_road_split_story(pacing_bias, locale))
    return lines


def _trail_story_lines(
    locale: str,
    fade_profile_preset: str | None,
    descent_caution: str,
    peak_temperature_c: float | None,
    sections: list[Any],
) -> list[str]:
    fade_label = t(f"preset.fade.{fade_profile_preset or 'stable'}", locale).lower()
    descent_label = t(f"desc.{descent_caution}", locale)
    lines = [t("story.trail_intro", locale, fade=fade_label, descent=descent_label)]
    if peak_temperature_c is not None and peak_temperature_c >= 24:
        lines.append(t("story.trail_heat", locale))

    costly_sections = sorted(sections, key=lambda segment: segment.segment_time_min, reverse=True)
    if len(costly_sections) >= 2:
        lines.append(
            t(
                "story.trail_sections",
                locale,
                section_one=costly_sections[0].block_label or costly_sections[0].section_name,
                section_two=costly_sections[1].block_label or costly_sections[1].section_name,
            )
        )
    elif len(costly_sections) == 1:
        lines.append(
            t(
                "story.trail_sections.single",
                locale,
                section_one=costly_sections[0].block_label or costly_sections[0].section_name,
            )
        )
    return lines


def render_general_planner(repo_root: Path) -> None:
    st.session_state.setdefault("general_locale", DEFAULT_LOCALE)
    locale = st.session_state["general_locale"]

    st.title(t("app.title", locale))
    st.caption(t("app.caption", locale))

    st.session_state.setdefault("general_config", default_config())
    st.session_state.setdefault("general_athlete_profile", default_athlete_profile())
    st.session_state.setdefault("general_event_id", "semi-marathon-finistere")
    st.session_state.setdefault("general_course_id", "semi-marathon-finistere")

    events = list_curated_events()
    if not events:
        st.error(t("app.no_events", locale))
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
    rest_duration_water_only_sec = int(cfg.get("rest_duration_water_only_sec", 240))
    rest_duration_standard_sec = int(cfg.get("rest_duration_standard_sec", 480))
    rest_duration_full_service_sec = int(cfg.get("rest_duration_full_service_sec", 720))
    carried_weight_kg = float(cfg.get("carried_weight_kg", 3.0))

    with st.sidebar:
        locale_options = list(AVAILABLE_LOCALES)
        locale_index = locale_options.index(locale) if locale in locale_options else 0
        locale = st.radio(
            t("sidebar.language", locale),
            options=locale_options,
            index=locale_index,
            format_func=lambda loc: t(f"sidebar.language.{loc}", loc),
            horizontal=True,
        )
        st.session_state["general_locale"] = locale

        st.header(t("sidebar.header", locale))
        with st.expander(t("sidebar.user_guide", locale)):
            guide_path = f"docs/unified-event-planner/user-guide/{locale}.md"
            st.markdown(_read_markdown_doc(repo_root, guide_path, locale))
        with st.expander(t("sidebar.technical_models", locale)):
            st.markdown(t("app.models_overview_body", locale))
        with st.expander(t("sidebar.technical_reference", locale)):
            technical_doc_path = (
                "docs/unified-event-planner/technical-models-fr.md"
                if locale == "fr"
                else "docs/unified-event-planner/technical-models.md"
            )
            st.markdown(_read_markdown_doc(repo_root, technical_doc_path, locale))

        with st.expander(t("expander.load_plan", locale)):
            loaded_plan = st.file_uploader(
                t("control.plan_json", locale), type=["json"], key="plan_json_uploader"
            )
            if loaded_plan is not None:
                updated_state, load_error = load_plan_into_state(
                    loaded_plan.getvalue().decode("utf-8"),
                    {str(key): st.session_state[key] for key in st.session_state.keys()},
                )
                if load_error is not None:
                    st.error(t("msg.plan_load_error", locale, error=load_error))
                    st.info(t("msg.plan_missing_gpx_hint", locale))
                else:
                    st.session_state["general_course_id"] = updated_state["general_course_id"]
                    st.session_state["general_config"] = updated_state["general_config"]
                    if "general_event_id" in updated_state:
                        st.session_state["general_event_id"] = updated_state["general_event_id"]
                    if "general_athlete_profile" in updated_state:
                        st.session_state["general_athlete_profile"] = updated_state[
                            "general_athlete_profile"
                        ]
                    st.success(t("msg.plan_loaded", locale))
                    cfg = st.session_state["general_config"]
                    athlete_profile = normalized_athlete_profile(
                        st.session_state["general_athlete_profile"]
                    )

        selected_event = st.selectbox(
            t("control.event", locale),
            options=events,
            index=event_index,
            format_func=lambda event: event.name,
            help=t("control.event.help", locale),
        )

        selected_course = get_course_by_id(selected_event.course_id)
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

        with st.expander(t("expander.athlete_profile", locale), expanded=True):
            st.caption(t("caption.athlete_profile", locale))
            uploaded_profile_json = st.file_uploader(
                t("control.athlete_profile_json", locale),
                type=["json"],
                key="athlete_profile_json_uploader",
            )
            if uploaded_profile_json is not None:
                try:
                    athlete_profile = normalized_athlete_profile(
                        json.loads(uploaded_profile_json.getvalue().decode("utf-8"))
                    )
                    st.session_state["general_athlete_profile"] = athlete_profile
                    st.success(t("msg.profile_loaded", locale))
                except json.JSONDecodeError:
                    st.error(t("msg.profile_parse_error", locale))

            st.markdown(t("section.road_baselines", locale))
            lt1_hr_raw = st.text_input(
                t("control.lt1_hr", locale), profile_text(athlete_profile, "lt1_hr")
            )
            lt2_hr_raw = st.text_input(
                t("control.lt2_hr", locale), profile_text(athlete_profile, "lt2_hr")
            )
            lt1_hr_value = parse_optional_number(lt1_hr_raw)
            lt2_hr_value = parse_optional_number(lt2_hr_raw)
            athlete_profile["lt1_hr"] = int(lt1_hr_value) if lt1_hr_value is not None else None
            athlete_profile["lt2_hr"] = int(lt2_hr_value) if lt2_hr_value is not None else None
            athlete_profile["lt1_pace_min_km"] = parse_optional_number(
                st.text_input(
                    t("control.lt1_pace", locale),
                    profile_text(athlete_profile, "lt1_pace_min_km"),
                    help=t("control.lt1_pace.help", locale),
                )
            )
            athlete_profile["lt2_pace_min_km"] = parse_optional_number(
                st.text_input(
                    t("control.lt2_pace", locale),
                    profile_text(athlete_profile, "lt2_pace_min_km"),
                    help=t("control.lt2_pace.help", locale),
                )
            )

            st.markdown(t("section.road_capability", locale))
            st.caption(t("caption.capability_precedence", locale))
            athlete_profile["best_likely_half_time_min"] = parse_optional_number(
                st.text_input(
                    t("control.best_likely_half", locale),
                    profile_text(athlete_profile, "best_likely_half_time_min"),
                    help=t("control.best_likely_half.help", locale),
                )
            )
            athlete_profile["best_likely_marathon_time_min"] = parse_optional_number(
                st.text_input(
                    t("control.best_likely_marathon", locale),
                    profile_text(athlete_profile, "best_likely_marathon_time_min"),
                    help=t("control.best_likely_marathon.help", locale),
                )
            )
            athlete_profile["predictor_half_time_min"] = parse_optional_number(
                st.text_input(
                    t("control.predictor_half", locale),
                    profile_text(athlete_profile, "predictor_half_time_min"),
                    help=t("control.predictor_half.help", locale),
                )
            )
            athlete_profile["predictor_marathon_time_min"] = parse_optional_number(
                st.text_input(
                    t("control.predictor_marathon", locale),
                    profile_text(athlete_profile, "predictor_marathon_time_min"),
                    help=t("control.predictor_marathon.help", locale),
                )
            )
            predictor_source = st.text_input(
                t("control.predictor_source", locale),
                profile_text(athlete_profile, "predictor_source"),
                help=t("control.predictor_source.help", locale),
            ).strip()
            athlete_profile["predictor_source"] = predictor_source or None

            st.markdown(t("section.trail_adjustments", locale))
            athlete_profile["flat_trail_slowdown_sec_km"] = parse_optional_number(
                st.text_input(
                    t("control.flat_trail_slowdown", locale),
                    profile_text(athlete_profile, "flat_trail_slowdown_sec_km"),
                    help=t("control.flat_trail_slowdown.help", locale),
                )
            )
            athlete_profile["technical_trail_slowdown_sec_km"] = parse_optional_number(
                st.text_input(
                    t("control.technical_trail_slowdown", locale),
                    profile_text(athlete_profile, "technical_trail_slowdown_sec_km"),
                    help=t("control.technical_trail_slowdown.help", locale),
                )
            )

            st.markdown(t("section.universal_factors", locale))
            st.caption(t("caption.universal_factors", locale))
            athlete_profile["body_mass_kg"] = parse_optional_number(
                st.text_input(
                    t("control.body_mass", locale),
                    profile_text(athlete_profile, "body_mass_kg"),
                    help=t("control.body_mass.help", locale),
                )
            )
            athlete_profile["sweat_rate_l_hr"] = parse_optional_number(
                st.text_input(
                    t("control.sweat_rate", locale),
                    profile_text(athlete_profile, "sweat_rate_l_hr"),
                    help=t("control.sweat_rate.help", locale),
                )
            )
            athlete_profile["gut_carb_tolerance_g_hr"] = parse_optional_number(
                st.text_input(
                    t("control.gut_carb_tolerance", locale),
                    profile_text(athlete_profile, "gut_carb_tolerance_g_hr"),
                    help=t("control.gut_carb_tolerance.help", locale),
                )
            )
            athlete_profile["durability_factor"] = st.slider(
                t("control.durability_factor", locale),
                min_value=-1.0,
                max_value=1.0,
                value=float(athlete_profile.get("durability_factor") or 0.0),
                step=0.1,
                help=t("control.durability_factor.help", locale),
            )
            athlete_profile["heat_tolerance"] = st.slider(
                t("control.heat_tolerance", locale),
                min_value=-1.0,
                max_value=1.0,
                value=float(athlete_profile.get("heat_tolerance") or 0.0),
                step=0.1,
                help=t("control.heat_tolerance.help", locale),
            )
            athlete_profile["hill_tolerance"] = st.slider(
                t("control.hill_tolerance", locale),
                min_value=-1.0,
                max_value=1.0,
                value=float(athlete_profile.get("hill_tolerance") or 0.0),
                step=0.1,
                help=t("control.hill_tolerance.help", locale),
            )

            st.markdown(t("section.preferences", locale))
            athlete_profile["default_road_split_bias"] = st.slider(
                t("control.default_road_split_bias", locale),
                min_value=-10.0,
                max_value=10.0,
                value=float(athlete_profile.get("default_road_split_bias") or 0.0),
                step=0.5,
                help=t("control.default_road_split_bias.help", locale),
            )
            athlete_profile["default_trail_fade_preset"] = st.selectbox(
                t("control.default_trail_fade", locale),
                options=list(FADE_PROFILE_PRESETS.keys()),
                index=list(FADE_PROFILE_PRESETS.keys()).index(derived_fade_preset(athlete_profile)),
                format_func=lambda key: t(f"preset.fade.{key}", locale),
            )
            athlete_profile["default_trail_effort_policy"] = st.selectbox(
                t("control.default_trail_effort", locale),
                options=list(EFFORT_POLICY_PRESETS.keys()),
                index=list(EFFORT_POLICY_PRESETS.keys()).index(
                    derived_effort_policy(athlete_profile)
                ),
                format_func=lambda key: t(f"preset.effort.{key}", locale),
            )

            st.download_button(
                t("button.download_profile", locale),
                data=athlete_profile_json(athlete_profile),
                file_name="athlete-profile.json",
                mime="application/json",
                use_container_width=True,
            )
            if st.button(t("button.apply_profile", locale), use_container_width=True):
                st.session_state["general_athlete_profile"] = athlete_profile
                st.session_state["general_config"] = default_config_for_event(
                    selected_event,
                    athlete_profile,
                )
                st.session_state.pop("general_result", None)
                st.session_state.pop("general_selected_course", None)
                st.session_state.pop("general_loaded_course", None)
                st.rerun()

        st.markdown(t("section.event_setup", locale))
        st.caption(t("caption.event_setup_road", locale))

        input_mode = st.radio(
            t("control.target_mode", locale),
            options=["finish_time", "effort_anchor"],
            index=0
            if cfg.get("input_mode", selected_event.default_input_mode) == "finish_time"
            else 1,
            format_func=lambda mode: t(f"target_mode.{mode}", locale),
            help=t("control.target_mode.help", locale),
        )

        if is_road_event(selected_event):
            if input_mode == "finish_time":
                target_finish_time_min = st.number_input(
                    t("control.target_finish_time", locale),
                    min_value=30.0,
                    max_value=2400.0,
                    value=float(cfg.get("target_finish_time_min") or 240.0),
                    step=5.0,
                )
            else:
                marathon_pace_min_km = st.number_input(
                    t("control.anchor_pace", locale),
                    min_value=3.0,
                    max_value=20.0,
                    value=float(cfg.get("marathon_pace_min_km") or 5.5),
                    step=0.1,
                    help=t("control.anchor_pace.help", locale),
                )
        elif race_model == "fire_road_ultra":
            if input_mode == "finish_time":
                target_finish_time_min = st.number_input(
                    t("control.target_finish_time", locale),
                    min_value=30.0,
                    max_value=4000.0,
                    value=float(cfg.get("target_finish_time_min") or 720.0),
                    step=10.0,
                )
            else:
                z1_pace_min_km = st.number_input(
                    t("control.z1_pace", locale),
                    min_value=4.0,
                    max_value=25.0,
                    value=float(cfg.get("z1_pace_min_km") or 8.0),
                    step=0.1,
                    help=t("control.z1_pace.help", locale),
                )
                z2_pace_min_km = st.number_input(
                    t("control.z2_pace", locale),
                    min_value=3.0,
                    max_value=20.0,
                    value=float(cfg.get("z2_pace_min_km") or 7.0),
                    step=0.1,
                    help=t("control.z2_pace.help", locale),
                )
                hike_pace_min_km = st.number_input(
                    t("control.hike_pace", locale),
                    min_value=5.0,
                    max_value=40.0,
                    value=float(cfg.get("hike_pace_min_km") or 12.0),
                    step=0.1,
                    help=t("control.hike_pace.fire_road.help", locale),
                )
        else:
            if input_mode == "finish_time":
                target_finish_time_min = st.number_input(
                    t("control.target_finish_time", locale),
                    min_value=30.0,
                    max_value=4000.0,
                    value=float(cfg.get("target_finish_time_min") or 720.0),
                    step=10.0,
                )
            else:
                flat_pace_min_km = st.number_input(
                    t("control.flat_trail_pace", locale),
                    min_value=4.0,
                    max_value=25.0,
                    value=float(cfg.get("flat_pace_min_km") or 8.5),
                    step=0.1,
                    help=t("control.flat_trail_pace.help", locale),
                )
                hike_pace_min_km = st.number_input(
                    t("control.hike_pace", locale),
                    min_value=5.0,
                    max_value=40.0,
                    value=float(cfg.get("hike_pace_min_km") or 13.0),
                    step=0.1,
                    help=t("control.hike_pace.technical.help", locale),
                )

        if is_road_event(selected_event):
            st.markdown(t("section.race_strategy", locale))
            race_intent = st.selectbox(
                t("control.race_intent", locale),
                options=list(ROAD_INTENT_PRESETS.keys()),
                index=list(ROAD_INTENT_PRESETS.keys()).index(
                    str(cfg.get("race_intent") or "controlled")
                ),
                format_func=lambda key: t(f"preset.intent.{key}", locale),
                help=t("control.race_intent.help", locale),
            )
            pacing_bias = st.slider(
                t("control.split_bias", locale),
                min_value=-10.0,
                max_value=10.0,
                value=float(cfg.get("pacing_bias", 0.0)),
                step=0.5,
                help=t("control.split_bias.help", locale),
            )
            rest_duration_sec = st.slider(
                t("control.aid_stop_time_sec", locale),
                min_value=0,
                max_value=90,
                value=int(cfg.get("rest_duration_sec", 10)),
                step=5,
                help=t("control.aid_stop_time_sec.help", locale),
            )
        else:
            st.markdown(t("section.terrain_fade", locale))
            climb_hike_threshold_percent = st.slider(
                t("control.climb_hike_threshold", locale),
                min_value=5.0,
                max_value=25.0,
                value=float(cfg.get("climb_hike_threshold_percent", 12.0)),
                step=0.5,
                help=t("control.climb_hike_threshold.help", locale),
            )
            if race_model == "technical_trail_ultra":
                descent_caution = st.selectbox(
                    t("control.descent_caution", locale),
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(cfg.get("descent_caution", "medium")),
                    format_func=lambda val: t(f"preset.descent.{val}", locale),
                    help=t("control.descent_caution.help", locale),
                )
            fade_profile_preset = st.selectbox(
                t("control.fade_profile", locale),
                options=list(FADE_PROFILE_PRESETS.keys()),
                index=list(FADE_PROFILE_PRESETS.keys()).index(
                    str(cfg.get("fade_profile_preset") or "progressive_fade")
                ),
                format_func=lambda key: t(f"preset.fade.{key}", locale),
                help=t("control.fade_profile.help", locale),
            )
            fade_early_bias, fade_mid_bias, fade_late_bias = FADE_PROFILE_PRESETS[
                fade_profile_preset
            ][1]
            st.caption(
                t(
                    "caption.fade_phases",
                    locale,
                    early=fade_early_bias,
                    mid=fade_mid_bias,
                    late=fade_late_bias,
                )
            )
            rest_duration_min = st.slider(
                t("control.aid_stop_time_min", locale),
                min_value=0.0,
                max_value=20.0,
                value=round(float(cfg.get("rest_duration_sec", 180)) / 60.0, 1),
                step=0.5,
                help=t("control.aid_stop_time_min.help", locale),
            )
            rest_duration_sec = int(rest_duration_min * 60)

            event_tiers = {station.tier for station in selected_course.aid_stations}
            if "water_only" in event_tiers:
                water_only_min = st.slider(
                    f"{t('tier.water_only', locale)} — {t('control.aid_stop_time_min', locale)}",
                    min_value=0.0,
                    max_value=15.0,
                    value=round(float(cfg.get("rest_duration_water_only_sec", 240)) / 60.0, 1),
                    step=0.5,
                )
                rest_duration_water_only_sec = int(water_only_min * 60)
            if "standard" in event_tiers:
                standard_min = st.slider(
                    f"{t('tier.standard', locale)} — {t('control.aid_stop_time_min', locale)}",
                    min_value=0.0,
                    max_value=20.0,
                    value=round(float(cfg.get("rest_duration_standard_sec", 480)) / 60.0, 1),
                    step=0.5,
                )
                rest_duration_standard_sec = int(standard_min * 60)
            if "full_service" in event_tiers:
                full_service_min = st.slider(
                    f"{t('tier.full_service', locale)} — {t('control.aid_stop_time_min', locale)}",
                    min_value=0.0,
                    max_value=30.0,
                    value=round(float(cfg.get("rest_duration_full_service_sec", 720)) / 60.0, 1),
                    step=1.0,
                )
                rest_duration_full_service_sec = int(full_service_min * 60)

        st.markdown(t("section.weather", locale))
        peak_temperature_c = st.number_input(
            t("control.peak_temp", locale),
            min_value=0.0,
            max_value=45.0,
            value=float(
                cfg.get(
                    "peak_temperature_c",
                    selected_event.baseline_peak_temp_c or 18.0,
                )
            ),
            step=1.0,
            help=t("control.peak_temp.help", locale),
        )

        if is_trail_event(selected_event):
            with st.expander(t("expander.advanced_trail", locale)):
                effort_policy = st.selectbox(
                    t("control.effort_policy", locale),
                    options=list(EFFORT_POLICY_PRESETS.keys()),
                    index=list(EFFORT_POLICY_PRESETS.keys()).index(
                        str(cfg.get("effort_policy") or derived_effort_policy(athlete_profile))
                    ),
                    format_func=lambda key: t(f"preset.effort.{key}", locale),
                    help=t("control.effort_policy.help", locale),
                )
                derived_hr_cap = derived_hr_guardrail_cap(
                    athlete_profile,
                    race_model,
                    effort_policy,
                )
                if derived_hr_cap is None:
                    use_hr_guardrail = False
                    st.info(t("msg.no_hr_data", locale))
                else:
                    use_hr_guardrail = st.checkbox(
                        t("control.use_hr_guardrail", locale),
                        value=bool(cfg.get("use_hr_guardrail", True)),
                        help=t("control.use_hr_guardrail.help", locale),
                    )
                    st.caption(t("caption.derived_hr_cap", locale, hr=derived_hr_cap))
                carried_weight_kg = st.slider(
                    t("control.carried_weight", locale),
                    min_value=0.0,
                    max_value=10.0,
                    value=float(cfg.get("carried_weight_kg", 3.0)),
                    step=0.5,
                    help=t("control.carried_weight.help", locale),
                )

        calculate_plan_clicked = st.button(
            t("button.calculate", locale), type="primary", use_container_width=True
        )

    if selected_course is None:
        st.error(t("msg.event_not_resolved", locale))
        return

    overview_course = load_course_trackpoints(selected_course)
    aid_tier_by_distance = {
        round(station.distance_km, 2): station.tier for station in selected_course.aid_stations
    }
    total_gain_m, total_loss_m = elevation_changes(
        overview_course.trackpoints,
        0.0,
        overview_course.total_distance_km * 1000,
    )

    st.markdown(t("section.course_overview", locale))
    overview_a, overview_b = st.columns(2)
    with overview_a:
        for row in course_overview_rows(overview_course.total_distance_km, selected_event, locale):
            st.markdown(f"**{row['label']}:** {row['value']}")
    with overview_b:
        st.markdown(t("overview.elevation", locale, gain=total_gain_m, loss=total_loss_m))
        if selected_course.aid_stations:
            aid_str = ", ".join(f"{aid.distance_km:.1f} km" for aid in selected_course.aid_stations)
            st.caption(t("caption.aid_points", locale, points=aid_str))

    if is_road_event(selected_event):
        capability_rows = road_capability_sources(athlete_profile, race_model)
        selected_capability_time_min, selected_capability_source = selected_road_capability(
            athlete_profile,
            race_model,
        )
        st.markdown(t("section.road_capability_panel", locale))
        st.caption(t("caption.road_capability", locale))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        t("col.source", locale): _translate_source_label(row["source"], locale),
                        t("col.best_likely", locale): format_duration_minutes(
                            float(row["time_min"]) if row["time_min"] is not None else None
                        ),
                        t("col.selected", locale): t("label.yes", locale)
                        if row["selected"]
                        else "",
                    }
                    for row in capability_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if selected_capability_time_min is not None and selected_capability_source is not None:
            st.caption(
                t(
                    "caption.selected_source",
                    locale,
                    source=_translate_source_label(selected_capability_source, locale),
                    time=format_duration_minutes(selected_capability_time_min),
                )
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
                            t("label.metric", locale): t("label.selected_best_likely", locale),
                            t("label.value", locale): format_duration_minutes(
                                adjusted_capability["base_time_min"]
                            ),
                        },
                        {
                            t("label.metric", locale): t("label.course_impact", locale),
                            t(
                                "label.value", locale
                            ): f"x{adjusted_capability['course_multiplier']:.3f}",
                        },
                        {
                            t("label.metric", locale): t("label.weather_impact", locale),
                            t(
                                "label.value", locale
                            ): f"x{adjusted_capability['weather_multiplier']:.3f}",
                        },
                        {
                            t("label.metric", locale): t("label.adjusted_best_likely", locale),
                            t("label.value", locale): format_duration_minutes(
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
            st.caption(t("caption.adjusted_explanation", locale))
            if chosen_target_time_min is not None:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                t("label.metric", locale): t("label.race_intent", locale),
                                t("label.value", locale): t(f"preset.intent.{race_intent}", locale),
                            },
                            {
                                t("label.metric", locale): t("label.intent_suggested", locale),
                                t("label.value", locale): format_duration_minutes(
                                    suggested_target_time_min
                                ),
                            },
                            {
                                t("label.metric", locale): t("label.chosen_target", locale),
                                t("label.value", locale): format_duration_minutes(
                                    chosen_target_time_min
                                ),
                            },
                            {
                                t("label.metric", locale): t("label.feasibility", locale),
                                t("label.value", locale): t(
                                    classify_road_feasibility(
                                        adjusted_capability["adjusted_time_min"],
                                        chosen_target_time_min,
                                    ),
                                    locale,
                                ),
                            },
                            {
                                t("label.metric", locale): t("label.expected_effort", locale),
                                t("label.value", locale): t(
                                    classify_road_effort_band(
                                        adjusted_capability["adjusted_time_min"],
                                        chosen_target_time_min,
                                    ),
                                    locale,
                                ),
                            },
                            {
                                t("label.metric", locale): t("label.recovery_cost", locale),
                                t("label.value", locale): t(
                                    classify_road_recovery_cost(
                                        adjusted_capability["adjusted_time_min"],
                                        chosen_target_time_min,
                                    ),
                                    locale,
                                ),
                            },
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                with st.expander(t("expander.how_road_works", locale)):
                    st.markdown(t("text.how_road_works", locale))

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
        rest_duration_water_only_sec=rest_duration_water_only_sec,
        rest_duration_standard_sec=rest_duration_standard_sec,
        rest_duration_full_service_sec=rest_duration_full_service_sec,
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
        event_month=selected_event.event_month,
        carried_weight_kg=0.0 if is_road_event(selected_event) else carried_weight_kg,
        athlete_body_mass_kg=athlete_profile.get("body_mass_kg"),
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
        is_road_result = is_road_event(selected_event)
        st.subheader(t("section.plan_output", locale))
        average_pace_min_km = (
            result.moving_time_min / result.total_distance_km
            if result.total_distance_km > 0
            else None
        )
        body_mass_kg = athlete_profile.get("body_mass_kg")
        fueling_plan = None
        if body_mass_kg is not None:
            fueling_plan = build_fueling_plan(
                result,
                mass_kg=float(body_mass_kg),
                peak_temperature_c=peak_temperature_c,
                aid_station_tiers=[station.tier for station in selected_course.aid_stations],
                athlete_sweat_rate=athlete_profile.get("sweat_rate_l_hr"),
                gut_carb_tolerance_g_hr=athlete_profile.get("gut_carb_tolerance_g_hr"),
            )

        adjusted_capability = None
        chosen_target_time_min: float | None = None
        if is_road_result:
            selected_capability_time_min, _ = selected_road_capability(athlete_profile, race_model)
            if selected_capability_time_min is not None:
                adjusted_capability = estimate_road_adjusted_best_likely(
                    overview_course,
                    selected_capability_time_min,
                    peak_temperature_c=peak_temperature_c,
                    start_time_local=selected_event.start_time_local,
                    hill_tolerance=athlete_profile.get("hill_tolerance"),
                    heat_tolerance=athlete_profile.get("heat_tolerance"),
                )
            race_distance_km = road_race_distance_km(race_model)
            if input_mode == "finish_time":
                chosen_target_time_min = target_finish_time_min
            elif marathon_pace_min_km is not None and race_distance_km is not None:
                chosen_target_time_min = marathon_pace_min_km * race_distance_km

        story_lines = (
            _road_story_lines(
                locale,
                race_intent,
                pacing_bias,
                adjusted_capability["adjusted_time_min"]
                if adjusted_capability is not None
                else None,
                chosen_target_time_min,
            )
            if is_road_result
            else _trail_story_lines(
                locale,
                fade_profile_preset,
                descent_caution,
                peak_temperature_c,
                result.segments,
            )
        )

        hero_a, hero_b, hero_c, hero_d, hero_e = st.columns(5)
        hero_a.metric(
            t("metric.finish_estimate", locale),
            format_duration_minutes(result.total_time_min),
        )
        hero_b.metric(t("metric.moving", locale), format_duration_minutes(result.moving_time_min))
        hero_c.metric(t("metric.avg_pace", locale), format_pace_minutes(average_pace_min_km))
        hero_d.metric(
            t("summary.elev_gain", locale),
            f"+{total_gain_m:.0f}m / -{total_loss_m:.0f}m",
        )
        hero_e.metric(
            t("metric.fueling_target", locale),
            (
                f"{fueling_plan.total_carb_target_g:.0f}g / {fueling_plan.total_fluid_target_l:.1f}L"
                if fueling_plan is not None
                else "-"
            ),
        )

        st.markdown(t("section.what_plan_means", locale))
        for story_line in story_lines:
            st.markdown(f"- {story_line}")

        st.markdown(t("section.race_story", locale))
        st.pyplot(
            plot_course_profile(loaded_course.trackpoints, chosen_course.aid_stops_km, locale)
        )
        st.pyplot(plot_pace_profile(result, locale))
        story_left, story_right = st.columns(2)
        with story_left:
            st.pyplot(plot_cumulative_time(result, selected_event, locale))
        with story_right:
            if is_road_result:
                st.pyplot(plot_half_comparison(result, locale))
            else:
                st.pyplot(plot_terrain_breakdown(result, locale))

        if result.assumptions:
            translated = [_translate_message(a, locale) for a in result.assumptions]
            st.caption(t("caption.assumptions", locale, assumptions=" | ".join(translated)))
        if result.warnings:
            for warning in result.warnings:
                st.warning(_translate_message(warning, locale))

        if is_road_result:
            (
                tab_aid,
                tab_splits,
                tab_fueling,
                tab_sections,
                tab_weather,
                tab_diagnostics,
                tab_snapshot,
            ) = st.tabs(
                [
                    t("tab.aid_stations", locale),
                    t("tab.splits", locale),
                    t("tab.fueling", locale),
                    t("tab.sections", locale),
                    t("tab.weather", locale),
                    t("tab.diagnostics", locale),
                    t("tab.snapshot", locale),
                ]
            )
        else:
            (
                tab_aid,
                tab_sections,
                tab_fueling,
                tab_splits,
                tab_weather,
                tab_diagnostics,
                tab_snapshot,
            ) = st.tabs(
                [
                    t("tab.aid_stations", locale),
                    t("tab.sections", locale),
                    t("tab.fueling", locale),
                    t("tab.splits", locale),
                    t("tab.weather", locale),
                    t("tab.diagnostics", locale),
                    t("tab.snapshot", locale),
                ]
            )

        with tab_diagnostics:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            t("label.metric", locale): t("summary.terrain", locale),
                            t("label.value", locale): selected_event.terrain.title(),
                        },
                        {
                            t("label.metric", locale): t("summary.elev_gain", locale),
                            t("label.value", locale): f"+{total_gain_m:.0f}m",
                        },
                        {
                            t("label.metric", locale): t("summary.elev_loss", locale),
                            t("label.value", locale): f"-{total_loss_m:.0f}m",
                        },
                        {
                            t("label.metric", locale): t("summary.aid_count", locale),
                            t("label.value", locale): str(len(result.aid_station_etas)),
                        },
                        {
                            t("label.metric", locale): t("summary.estimated_finish", locale),
                            t("label.value", locale): format_clock_time(
                                selected_event, result.total_time_min, locale
                            ),
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            if is_road_result:
                st.pyplot(plot_half_comparison(result, locale))
                st.pyplot(plot_terrain_breakdown(result, locale))
            else:
                st.pyplot(plot_terrain_breakdown(result, locale))
                st.pyplot(plot_half_comparison(result, locale))

        with tab_aid:
            if result.aid_station_etas:
                st.markdown(t("section.aid_actions", locale))
                station_cols = st.columns(2)
                for idx, aid_eta in enumerate(result.aid_station_etas):
                    tier = aid_tier_by_distance.get(round(aid_eta.distance_km, 2), "none")
                    with station_cols[idx % 2]:
                        st.markdown(
                            f"**{aid_eta.label or t('aid.label_fallback', locale, n=idx + 1)}**\n\n"
                            f"{t('col.distance_km', locale)}: {aid_eta.distance_km:.1f} km  \n"
                            f"{t('col.arrival_clock', locale)}: {format_clock_time(selected_event, aid_eta.arrival_elapsed_time_min, locale)}  \n"
                            f"{t('col.departure_clock', locale)}: {format_clock_time(selected_event, aid_eta.departure_elapsed_time_min, locale)}  \n"
                            f"{t('col.tier', locale)}: {_aid_tier_label(tier, locale)}  \n"
                            f"{_aid_station_action(tier, locale)}"
                        )
                st.markdown(t("section.aid_timing", locale))
                st.dataframe(
                    [
                        {
                            t("col.section", locale): aid_eta.label
                            or t("aid.label_fallback", locale, n=idx + 1),
                            t("col.distance_km", locale): round(aid_eta.distance_km, 2),
                            t("col.split_distance_km", locale): round(aid_eta.split_distance_km, 2),
                            t("col.arrival_elapsed", locale): format_duration_minutes(
                                aid_eta.arrival_elapsed_time_min
                            ),
                            t("col.arrival_clock", locale): format_clock_time(
                                selected_event,
                                aid_eta.arrival_elapsed_time_min,
                                locale,
                            ),
                            t("col.departure_elapsed", locale): format_duration_minutes(
                                aid_eta.departure_elapsed_time_min
                            ),
                            t("col.departure_clock", locale): format_clock_time(
                                selected_event,
                                aid_eta.departure_elapsed_time_min,
                                locale,
                            ),
                            t("col.split_time", locale): format_duration_minutes(
                                aid_eta.split_from_prev_min
                            ),
                            t("col.split_pace", locale): format_pace_minutes(
                                aid_eta.actual_pace_min_km
                            ),
                            t("col.elev_gain_m", locale): round(aid_eta.elevation_gain_m, 1),
                            t("col.elev_loss_m", locale): round(aid_eta.elevation_loss_m, 1),
                            t("col.rest_time", locale): format_duration_minutes(
                                aid_eta.suggested_rest_min
                            ),
                            t("col.tier", locale): _aid_tier_label(
                                aid_tier_by_distance.get(round(aid_eta.distance_km, 2), "none"),
                                locale,
                            ),
                            t("col.source", locale): aid_eta.source,
                        }
                        for idx, aid_eta in enumerate(result.aid_station_etas)
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(t("msg.no_aid_stations", locale))

        with tab_sections:
            st.markdown(t("section.segment_pacing", locale))
            st.dataframe(
                [
                    {
                        t("col.section", locale): segment.section_name or segment.segment_type,
                        t("col.block", locale): segment.block_label,
                        t("col.type", locale): segment.segment_type,
                        t("col.start_km", locale): round(segment.start_km, 2),
                        t("col.end_km", locale): round(segment.end_km, 2),
                        t("col.distance_km_short", locale): round(segment.distance_km, 2),
                        t("col.elev_gain_m", locale): round(segment.elevation_gain_m, 1),
                        t("col.elev_loss_m", locale): round(segment.elevation_loss_m, 1),
                        t("col.start_elapsed", locale): format_duration_minutes(
                            segment.start_time_min
                        ),
                        t("col.end_elapsed", locale): format_duration_minutes(segment.end_time_min),
                        t("col.start_clock", locale): format_clock_time(
                            selected_event, segment.start_time_min, locale
                        ),
                        t("col.end_clock", locale): format_clock_time(
                            selected_event, segment.end_time_min, locale
                        ),
                        t("col.avg_grade", locale): round(segment.avg_grade_percent, 2),
                        t("col.avg_pace", locale): format_pace_minutes(segment.avg_pace_min_km),
                        t("col.segment_time", locale): format_duration_minutes(
                            segment.segment_time_min
                        ),
                    }
                    for segment in result.segments
                ],
                width="stretch",
                hide_index=True,
            )

        with tab_fueling:
            if fueling_plan is None:
                st.info(t("msg.no_body_mass", locale))
            else:
                f_sum_a, f_sum_b, f_sum_c, f_sum_d = st.columns(4)
                f_sum_a.metric(t("metric.total_kcal", locale), f"{fueling_plan.total_kcal:.0f}")
                f_sum_b.metric(t("metric.avg_kcal_hr", locale), f"{fueling_plan.avg_kcal_hr:.0f}")
                f_sum_c.metric(
                    t("metric.carb_target", locale), f"{fueling_plan.total_carb_target_g:.0f}g"
                )
                f_sum_d.metric(
                    t("metric.fluid_target", locale), f"{fueling_plan.total_fluid_target_l:.1f}L"
                )
                if fueling_plan.warnings:
                    for warning in fueling_plan.warnings:
                        st.warning(_translate_message(warning, locale))
                st.markdown(t("section.fueling_actions", locale))
                fueling_cols = st.columns(2)
                for idx, block in enumerate(fueling_plan.blocks):
                    carry_text, station_text, fluid_text = _fueling_block_action(block, locale)
                    with fueling_cols[idx % 2]:
                        on_site_detail = ""
                        if block.aid_station_tier != "none":
                            on_site_detail = (
                                f"{t('col.on_site_kcal', locale)}: {block.on_site_kcal:.0f}  \n"
                                f"{t('col.on_site_carb_g', locale)}: {block.on_site_carb_g:.0f}g  \n"
                            )
                        st.markdown(
                            f"**{block.start_km:.1f}-{block.end_km:.1f} km**\n\n"
                            f"{t('fueling.action.carry', locale)}: {carry_text}  \n"
                            f"{t('fueling.action.station', locale)}: {station_text}  \n"
                            f"{on_site_detail}"
                            f"{t('fueling.action.fluid', locale)}: {fluid_text}"
                        )
                st.markdown(t("section.per_block_fueling", locale))
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                t(
                                    "col.block", locale
                                ): f"{block.start_km:.1f}-{block.end_km:.1f} km",
                                t("col.tier", locale): _aid_tier_label(
                                    block.aid_station_tier, locale
                                ),
                                t("col.duration", locale): format_duration_minutes(
                                    block.duration_min
                                ),
                                t("col.kcal", locale): round(block.kcal_burned),
                                t("col.carb_target_g", locale): round(block.carb_target_g),
                                t("col.carb_planned_g", locale): round(block.carb_planned_g),
                                t("col.on_site_kcal", locale): round(block.on_site_kcal),
                                t("col.on_site_carb_g", locale): round(block.on_site_carb_g),
                                t("col.fluid_l", locale): block.fluid_target_l,
                                t("col.deficit_g", locale): round(block.cumulative_carb_deficit_g),
                                t("col.carry", locale): "; ".join(block.carry_items)
                                if block.carry_items
                                else t("label.carry_none", locale),
                            }
                            for block in fueling_plan.blocks
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if fueling_plan.carb_deficit_g > 0:
                    st.caption(
                        t("caption.carb_deficit", locale, deficit=fueling_plan.carb_deficit_g)
                    )

        with tab_splits:
            block_options = split_block_options(result.total_distance_km)
            default_split_block = default_split_block_size(result.total_distance_km)
            block_index = block_options.index(default_split_block)
            split_view_mode = st.selectbox(
                t("control.split_view", locale),
                options=["compact", "expanded"],
                format_func=lambda value: t(f"control.split_view.{value}", locale),
            )
            split_block_km = st.selectbox(
                t("control.split_block_size", locale),
                options=block_options,
                index=block_index,
                help=t("control.split_block_size.help", locale),
            )
            st.markdown(t("section.split_pacing", locale))
            st.caption(t("section.compact_split_note", locale))
            split_rows = aggregate_split_rows(
                result.splits,
                loaded_course.trackpoints,
                selected_event,
                split_block_km,
                locale,
            )
            split_frame = pd.DataFrame(split_rows)
            split_frame[t("col.pace_band", locale)] = [
                _pace_band_label(
                    _split_pace_display_to_minutes(str(pace_display)),
                    average_pace_min_km,
                    locale,
                )
                for pace_display in split_frame[t("col.pace", locale)]
            ]
            if split_view_mode == "compact":
                split_frame = split_frame[
                    [
                        t("col.split_range", locale),
                        t("col.pace", locale),
                        t("col.pace_band", locale),
                        t("col.split_time", locale),
                        t("col.elapsed", locale),
                        t("col.clock", locale),
                    ]
                ]
            st.dataframe(split_frame, width="stretch", hide_index=True)

        with tab_weather:
            if peak_temperature_c is not None:
                st.markdown(t("weather.title", locale))
                st.metric(t("weather.peak_temp", locale), f"{peak_temperature_c:.0f}°C")
                st.markdown(f"**{t('weather.curve_title', locale)}**")
                st.caption(t("weather.curve_caption", locale))
                st.pyplot(
                    plot_temperature_curve(
                        peak_temperature_c,
                        selected_event.start_time_local,
                        result.total_time_min,
                        locale,
                        selected_event.event_month,
                    )
                )
                st.markdown(f"**{t('weather.impact_title', locale)}**")
                st.caption(t("weather.impact_caption", locale))
                st.pyplot(
                    plot_heat_impact(
                        peak_temperature_c,
                        selected_event.start_time_local,
                        result.total_time_min,
                        result.splits,
                        locale,
                        selected_event.event_month,
                    )
                )
                avg_penalty_pct = 0.0
                if peak_temperature_c > 10.0:
                    avg_multiplier = average_heat_multiplier_for_duration(
                        peak_temperature_c,
                        selected_event.start_time_local,
                        result.moving_time_min,
                        selected_event.event_month,
                    )
                    avg_penalty_pct = (avg_multiplier - 1.0) * 100.0
                st.caption(t("weather.avg_penalty", locale, penalty=avg_penalty_pct))
            else:
                st.info(t("weather.no_temp", locale))

        with tab_snapshot:
            key_stations = result.aid_station_etas[:3]
            key_sections = sorted(
                result.segments, key=lambda segment: segment.segment_time_min, reverse=True
            )[:3]
            st.markdown(t("snapshot.title", locale))
            snap_a, snap_b = st.columns(2)
            with snap_a:
                st.markdown(f"**{t('snapshot.key_stations', locale)}**")
                for aid_eta in key_stations:
                    st.markdown(
                        f"- {aid_eta.label or t('aid.label_fallback', locale, n=1)}: {format_clock_time(selected_event, aid_eta.arrival_elapsed_time_min, locale)}"
                    )
                st.markdown(f"**{t('snapshot.fueling', locale)}**")
                if fueling_plan is not None and fueling_plan.blocks:
                    for block in fueling_plan.blocks[:3]:
                        carry_text, _, _ = _fueling_block_action(block, locale)
                        st.markdown(f"- {block.start_km:.1f}-{block.end_km:.1f} km: {carry_text}")
                else:
                    st.markdown("- -")
            with snap_b:
                st.markdown(f"**{t('snapshot.key_sections', locale)}**")
                for segment in key_sections:
                    st.markdown(
                        f"- {segment.section_name or segment.segment_type}: {_section_cue(segment.segment_type, locale)}"
                    )

        plan_json = export_plan_json(
            course_id=chosen_course.course_id,
            gpx_filename=chosen_course.gpx_path.name,
            config=PacingConfig(**st.session_state["general_config"]),
            athlete_profile=st.session_state["general_athlete_profile"],
        )
        st.download_button(
            t("button.download_plan", locale),
            data=plan_json,
            file_name="race-plan.json",
            mime="application/json",
        )

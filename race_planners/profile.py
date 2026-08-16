from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from race_planners.models import AthleteProfile, CuratedEvent
from race_planners.road_capability import (
    estimate_road_best_likely_pace_min_km,
    estimate_road_best_likely_time_min,
    road_race_distance_km,
)


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

ROAD_INTENT_PRESETS: dict[str, str] = {
    "best_effort": "Best Effort",
    "strong": "Strong",
    "controlled": "Controlled",
    "easy_durable": "Easy / Durable",
}


def default_athlete_profile() -> dict[str, Any]:
    return asdict(AthleteProfile())


def normalized_athlete_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    normalized = default_athlete_profile()
    if profile is None:
        return normalized
    for key in normalized:
        if key in profile:
            normalized[key] = profile[key]
    return normalized


def is_road_event(event: CuratedEvent) -> bool:
    return event.race_model in {"half_marathon", "road_marathon"}


def is_trail_event(event: CuratedEvent) -> bool:
    return not is_road_event(event)


def profile_text(profile: dict[str, Any], key: str) -> str:
    value = profile.get(key)
    return "" if value is None else str(value)


def parse_optional_number(raw_value: str) -> float | None:
    stripped = raw_value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def fade_profile_values(config: dict[str, Any]) -> tuple[float, float, float]:
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


def trail_anchor_defaults_from_profile(
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


def road_anchor_default_from_profile(
    athlete_profile: dict[str, Any], race_model: str
) -> float | None:
    lt1_pace = athlete_profile.get("lt1_pace_min_km")
    if lt1_pace is None:
        return None

    return estimate_road_best_likely_pace_min_km(
        race_model,
        float(lt1_pace),
        athlete_profile.get("lt2_pace_min_km"),
    )


def modeled_road_best_likely_time_min(
    athlete_profile: dict[str, Any], race_model: str
) -> float | None:
    anchor_pace_min_km = road_anchor_default_from_profile(athlete_profile, race_model)
    lt1_pace_min_km = athlete_profile.get("lt1_pace_min_km")
    if anchor_pace_min_km is None or lt1_pace_min_km is None:
        return None
    return estimate_road_best_likely_time_min(
        race_model,
        float(lt1_pace_min_km),
        athlete_profile.get("lt2_pace_min_km"),
    )


def road_capability_sources(
    athlete_profile: dict[str, Any], race_model: str
) -> list[dict[str, Any]]:
    profile_key = (
        "best_likely_half_time_min"
        if race_model == "half_marathon"
        else "best_likely_marathon_time_min"
    )
    predictor_key = (
        "predictor_half_time_min"
        if race_model == "half_marathon"
        else "predictor_marathon_time_min"
    )
    predictor_source = athlete_profile.get("predictor_source") or "Predictor"
    modeled_time_min = modeled_road_best_likely_time_min(athlete_profile, race_model)

    return [
        {
            "source": "Manual Profile",
            "time_min": athlete_profile.get(profile_key),
            "selected": athlete_profile.get(profile_key) is not None,
        },
        {
            "source": str(predictor_source),
            "time_min": athlete_profile.get(predictor_key),
            "selected": athlete_profile.get(profile_key) is None
            and athlete_profile.get(predictor_key) is not None,
        },
        {
            "source": "LT-Derived Model",
            "time_min": modeled_time_min,
            "selected": athlete_profile.get(profile_key) is None
            and athlete_profile.get(predictor_key) is None
            and modeled_time_min is not None,
        },
    ]


def selected_road_capability(
    athlete_profile: dict[str, Any], race_model: str
) -> tuple[float | None, str | None]:
    for source in road_capability_sources(athlete_profile, race_model):
        if source["selected"] and source["time_min"] is not None:
            return float(source["time_min"]), str(source["source"])
    return None, None


def selected_road_capability_pace_min_km(
    athlete_profile: dict[str, Any], race_model: str
) -> float | None:
    selected_time_min, selected_source = selected_road_capability(athlete_profile, race_model)
    race_distance_km = road_race_distance_km(race_model)
    if selected_time_min is None or race_distance_km is None:
        return None
    if selected_source == "LT-Derived Model":
        lt1_pace_min_km = athlete_profile.get("lt1_pace_min_km")
        if lt1_pace_min_km is None:
            return round(selected_time_min / race_distance_km, 2)
        return estimate_road_best_likely_pace_min_km(
            race_model,
            float(lt1_pace_min_km),
            athlete_profile.get("lt2_pace_min_km"),
        )
    return round(selected_time_min / race_distance_km, 2)


def derived_effort_policy(profile: dict[str, Any]) -> str:
    return str(profile.get("default_trail_effort_policy") or "steady")


def derived_fade_preset(profile: dict[str, Any]) -> str:
    return str(profile.get("default_trail_fade_preset") or "progressive_fade")


def derived_split_bias(profile: dict[str, Any]) -> float:
    return float(profile.get("default_road_split_bias") or 0.0)


def derived_hr_guardrail_cap(
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


def athlete_profile_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, sort_keys=True)


def default_config() -> dict[str, Any]:
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
        "rest_duration_water_only_sec": 240,
        "rest_duration_standard_sec": 480,
        "rest_duration_full_service_sec": 720,
        "pacing_bias": 0.0,
        "fade_profile_preset": "stable",
        "fade_early_bias": None,
        "fade_mid_bias": None,
        "fade_late_bias": None,
        "race_intent": None,
        "effort_policy": None,
        "use_hr_guardrail": False,
        "rpe_target": None,
        "hr_cap": None,
    }


def default_config_for_event(
    event: CuratedEvent, athlete_profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    athlete_profile = athlete_profile or {}
    config = default_config()
    config["race_model"] = event.race_model
    config["input_mode"] = event.default_input_mode
    config["peak_temperature_c"] = event.baseline_peak_temp_c or 18.0

    if event.race_model == "half_marathon":
        selected_capability_time_min, _ = selected_road_capability(
            athlete_profile, event.race_model
        )
        config["target_finish_time_min"] = selected_capability_time_min or 105.0
        config["rest_duration_sec"] = 10
        config["race_intent"] = "controlled"
        config["marathon_pace_min_km"] = road_anchor_default_from_profile(
            athlete_profile,
            event.race_model,
        )
        config["pacing_bias"] = derived_split_bias(athlete_profile)
    elif event.race_model == "road_marathon":
        selected_capability_time_min, _ = selected_road_capability(
            athlete_profile, event.race_model
        )
        config["target_finish_time_min"] = selected_capability_time_min or 240.0
        config["rest_duration_sec"] = 10
        config["rest_duration_water_only_sec"] = 240
        config["rest_duration_standard_sec"] = 480
        config["rest_duration_full_service_sec"] = 720
        config["race_intent"] = "controlled"
        config["marathon_pace_min_km"] = road_anchor_default_from_profile(
            athlete_profile,
            event.race_model,
        )
        config["pacing_bias"] = derived_split_bias(athlete_profile)
    else:
        config["input_mode"] = "effort_anchor"
        config["target_finish_time_min"] = None
        config["rest_duration_sec"] = 180
        config["rest_duration_water_only_sec"] = 240
        config["rest_duration_standard_sec"] = 480
        config["rest_duration_full_service_sec"] = 720
        config["fade_profile_preset"] = derived_fade_preset(athlete_profile)
        config["effort_policy"] = derived_effort_policy(athlete_profile)
        config["use_hr_guardrail"] = (
            derived_hr_guardrail_cap(
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
            flat_pace_min_km, hike_pace_min_km = trail_anchor_defaults_from_profile(
                athlete_profile, event.race_model
            )
            config["flat_pace_min_km"] = flat_pace_min_km or 8.5
            config["hike_pace_min_km"] = hike_pace_min_km or 13.0

    return config

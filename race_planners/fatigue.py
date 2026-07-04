from __future__ import annotations

from race_planners.models import PacingConfig
from race_planners.pacing import PacingContext


def is_road_race_model(race_model: str) -> bool:
    return race_model in {"road_marathon", "half_marathon"}


def fatigue_multiplier(race_model: str, progress_ratio: float) -> float:
    ratio = min(max(progress_ratio, 0.0), 1.0)
    if race_model in {"road_marathon", "half_marathon"}:
        if ratio <= 0.65:
            return 1.0
        return 1.0 + ((ratio - 0.65) / 0.35) * 0.04
    if race_model == "fire_road_ultra":
        if ratio <= 0.55:
            return 1.0
        return 1.0 + ((ratio - 0.55) / 0.45) * 0.06
    if race_model == "technical_trail_ultra":
        if ratio <= 0.5:
            return 1.0
        return 1.0 + ((ratio - 0.5) / 0.5) * 0.08
    return 1.0


def tolerance_penalty_scale(tolerance: float | None) -> float:
    if tolerance is None:
        return 1.0
    return min(max(1.0 - (float(tolerance) * 0.25), 0.6), 1.4)


def pacing_bias_multiplier(pacing_bias: float, progress_ratio: float) -> float:
    ratio = min(max(progress_ratio, 0.0), 1.0)
    return max(0.85, 1.0 + (pacing_bias * 0.005 * ratio))


def fade_profile_values(config: PacingConfig) -> tuple[float, float, float]:
    if None not in (config.fade_early_bias, config.fade_mid_bias, config.fade_late_bias):
        return (
            float(config.fade_early_bias or 0.0),
            float(config.fade_mid_bias or 0.0),
            float(config.fade_late_bias or 0.0),
        )

    preset_map = {
        "stable": (0.0, 0.75, 1.5),
        "late_fade": (0.0, 1.25, 3.5),
        "progressive_fade": (0.5, 2.0, 4.5),
        "blow_up_risk": (1.5, 4.0, 7.0),
    }
    return preset_map.get(config.fade_profile_preset or "stable", preset_map["stable"])


def interpolated_fade_bias(config: PacingConfig, progress_ratio: float) -> float:
    early_bias, mid_bias, late_bias = fade_profile_values(config)
    ratio = min(max(progress_ratio, 0.0), 1.0)
    if ratio <= 0.33:
        return early_bias * (ratio / 0.33)
    if ratio <= 0.66:
        blend = (ratio - 0.33) / 0.33
        return early_bias + ((mid_bias - early_bias) * blend)
    blend = (ratio - 0.66) / 0.34
    return mid_bias + ((late_bias - mid_bias) * blend)


def pacing_shape_multiplier(config: PacingConfig, progress_ratio: float) -> float:
    if is_road_race_model(config.race_model):
        return pacing_bias_multiplier(config.pacing_bias, progress_ratio)

    fade_bias = interpolated_fade_bias(config, progress_ratio)
    return max(0.85, 1.0 + (fade_bias * 0.006))


def trail_hill_tolerance_multiplier(config: PacingConfig, context: PacingContext) -> float:
    if is_road_race_model(config.race_model):
        return 1.0

    terrain_load = max(
        max(context.steepest_climb_percent, 0.0) / 20.0,
        abs(min(context.steepest_descent_percent, 0.0)) / 20.0,
        context.climb_m_per_km / 35.0,
    )
    terrain_load = min(max(terrain_load, 0.0), 1.5)
    tolerance = float(config.athlete_hill_tolerance or 0.0)
    return min(max(1.0 - (tolerance * 0.12 * terrain_load), 0.85), 1.15)


def durability_multiplier(config: PacingConfig, context: PacingContext) -> float:
    durability = float(config.athlete_durability_factor or 0.0)
    if durability == 0.0:
        return 1.0

    progress_load = min(max(context.progress_ratio, 0.0), 1.0) ** 2.4
    duration_load = min(max(context.elapsed_hours / 8.0, 0.0), 2.0)
    breakdown_load = progress_load * duration_load
    multiplier = min(max(1.0 - (durability * 0.08 * breakdown_load), 0.8), 1.2)
    return float(multiplier)

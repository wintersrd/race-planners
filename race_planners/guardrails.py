from __future__ import annotations

from race_planners.models import PacingConfig
from race_planners.pacing import PacingContext


def effort_policy_rpe_target(config: PacingConfig) -> float | None:
    if config.rpe_target is not None:
        return config.rpe_target
    policy_map = {
        "conservative": 4.5,
        "steady": 6.0,
        "aggressive": 7.5,
    }
    return policy_map.get(config.effort_policy or "", None)


def hr_guardrail_phase_offsets(config: PacingConfig) -> tuple[float, float, float]:
    offsets = {
        "conservative": (-3.0, 0.0, -4.0),
        "steady": (-1.5, 0.0, -2.5),
        "aggressive": (0.0, 1.0, -1.0),
    }
    return offsets.get(config.effort_policy or "steady", offsets["steady"])


def interpolate_three_phase_values(
    progress_ratio: float,
    early_value: float,
    mid_value: float,
    late_value: float,
) -> float:
    ratio = min(max(progress_ratio, 0.0), 1.0)
    if ratio <= 0.33:
        return early_value + ((mid_value - early_value) * (ratio / 0.33))
    if ratio <= 0.66:
        return mid_value
    return mid_value + ((late_value - mid_value) * ((ratio - 0.66) / 0.34))


def terrain_slowdown_minutes(config: PacingConfig, context: PacingContext) -> float:
    flat_slowdown_min = (config.athlete_flat_trail_slowdown_sec_km or 0.0) / 60.0
    technical_extra_min = (config.athlete_technical_trail_slowdown_sec_km or 0.0) / 60.0
    if config.race_model == "technical_trail_ultra":
        technicality = min(
            1.0,
            max(
                context.steepest_climb_percent / 14.0, abs(context.steepest_descent_percent) / 14.0
            ),
        )
        return flat_slowdown_min + (technical_extra_min * max(0.35, technicality))
    if config.race_model == "fire_road_ultra":
        return flat_slowdown_min * 0.65
    return 0.0


def road_equivalent_pace_min_km(
    config: PacingConfig,
    context: PacingContext,
    pace_min_km: float,
) -> float:
    return max(2.5, pace_min_km - terrain_slowdown_minutes(config, context))


def estimate_segment_hr(
    config: PacingConfig,
    context: PacingContext,
    pace_min_km: float,
) -> float | None:
    lt1_hr = config.athlete_lt1_hr
    lt2_hr = config.athlete_lt2_hr
    lt1_pace = config.athlete_lt1_pace_min_km
    lt2_pace = config.athlete_lt2_pace_min_km
    if None in (lt1_hr, lt2_hr, lt1_pace, lt2_pace):
        return None

    road_equivalent_pace = road_equivalent_pace_min_km(config, context, pace_min_km)
    assert lt1_hr is not None
    assert lt2_hr is not None
    assert lt1_pace is not None
    assert lt2_pace is not None
    lt1_hr_f = float(lt1_hr)
    lt2_hr_f = float(lt2_hr)
    lt1_pace_f = float(lt1_pace)
    lt2_pace_f = float(lt2_pace)

    if road_equivalent_pace >= lt1_pace_f:
        return max(lt1_hr_f - 10.0, lt1_hr_f - ((road_equivalent_pace - lt1_pace_f) * 6.0))

    if road_equivalent_pace >= lt2_pace_f:
        fraction = (lt1_pace_f - road_equivalent_pace) / max(lt1_pace_f - lt2_pace_f, 0.01)
        return lt1_hr_f + ((lt2_hr_f - lt1_hr_f) * fraction)

    return min(lt2_hr_f + 8.0, lt2_hr_f + ((lt2_pace_f - road_equivalent_pace) * 10.0))


def dynamic_hr_guardrail_cap(config: PacingConfig, context: PacingContext) -> float | None:
    if not config.use_hr_guardrail or config.hr_cap is None:
        return None

    early_offset, mid_offset, late_offset = hr_guardrail_phase_offsets(config)
    phase_cap = interpolate_three_phase_values(
        context.progress_ratio,
        float(config.hr_cap) + early_offset,
        float(config.hr_cap) + mid_offset,
        float(config.hr_cap) + late_offset,
    )
    terrain_penalty = min(
        4.0,
        max(context.steepest_climb_percent - 8.0, 0.0) * 0.12
        + max(abs(context.steepest_descent_percent) - 10.0, 0.0) * 0.05,
    )
    return phase_cap - terrain_penalty


def hr_guardrail_strategy_summary(config: PacingConfig) -> str | None:
    if not config.use_hr_guardrail or config.hr_cap is None:
        return None

    base_cap = float(config.hr_cap)
    early_offset, mid_offset, late_offset = hr_guardrail_phase_offsets(config)
    early_cap = int(round(base_cap + early_offset))
    mid_cap = int(round(base_cap + mid_offset))
    late_cap = int(round(base_cap + late_offset))
    return f"Derived HR strategy targets roughly {early_cap}/{mid_cap}/{late_cap} bpm across early, mid, and late race phases."


def effort_guardrail_multiplier(
    config: PacingConfig,
    context: PacingContext,
    pace_min_km: float,
) -> float:
    multiplier = 1.0
    rpe_target = effort_policy_rpe_target(config)

    if rpe_target is not None:
        multiplier *= max(0.88, min(1.12, 1.0 - ((rpe_target - 6.0) * 0.02)))

    dynamic_cap = dynamic_hr_guardrail_cap(config, context)
    estimated_hr = estimate_segment_hr(config, context, pace_min_km)
    if dynamic_cap is not None and estimated_hr is not None:
        overage_bpm = max(estimated_hr - dynamic_cap, 0.0)
        if overage_bpm > 0:
            severity = min(1.0, overage_bpm / 10.0)
            climb_weight = min(1.0, max(context.steepest_climb_percent, 0.0) / 15.0)
            late_weight = context.progress_ratio
            multiplier *= 1.0 + (
                severity * 0.12 * (0.4 + (climb_weight * 0.35) + (late_weight * 0.25))
            )
    elif config.hr_cap is not None:
        climb_load = max(context.grade_percent, 0.0) + (context.steepest_climb_percent * 0.6)
        late_load = context.progress_ratio * 4.0
        guardrail_load = min(1.0, (climb_load / 18.0) + (late_load / 10.0))
        multiplier *= 1.0 + (((155 - config.hr_cap) / 25.0) * 0.08 * guardrail_load)

    return max(0.82, min(1.18, multiplier))

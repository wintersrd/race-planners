from __future__ import annotations

import math

from race_planners.models import (
    FuelingBlock,
    FuelingPlan,
    PaceSplit,
    PlanResult,
)


_BASE_RUNNING_COST_KCAL_PER_KG_KM = 1.0


def _grade_cost_multiplier(grade_percent: float) -> float:
    if grade_percent > 0:
        return 1.0 + (grade_percent / 100.0) * 4.5
    if grade_percent < 0:
        adjusted_grade = max(grade_percent, -20.0)
        return 1.0 + (abs(adjusted_grade) / 100.0) * 0.6
    return 1.0


def segment_kcal(
    mass_kg: float,
    distance_km: float,
    grade_percent: float,
) -> float:
    return (
        mass_kg
        * distance_km
        * _BASE_RUNNING_COST_KCAL_PER_KG_KM
        * _grade_cost_multiplier(grade_percent)
    )


def estimate_event_kcal(
    mass_kg: float,
    splits: list[PaceSplit],
) -> float:
    total = 0.0
    prev_km = 0.0
    for split in splits:
        distance_km = max(split.km - prev_km, 0.0)
        total += segment_kcal(mass_kg, distance_km, split.grade_percent)
        prev_km = split.km
    return round(total, 1)


def _carb_target_band_g_per_hr(duration_hr: float) -> tuple[float, float]:
    if duration_hr < 1.0:
        return 0.0, 20.0
    if duration_hr < 1.5:
        return 20.0, 40.0
    if duration_hr < 3.0:
        return 40.0, 60.0
    if duration_hr < 6.0:
        return 60.0, 90.0
    if duration_hr < 12.0:
        return 40.0, 75.0
    return 30.0, 60.0


def _sweat_rate_l_per_hr(temperature_c: float, athlete_sweat_rate: float | None) -> float:
    if athlete_sweat_rate is not None:
        return athlete_sweat_rate
    if temperature_c <= 10.0:
        return 0.5
    if temperature_c <= 20.0:
        return 0.65
    if temperature_c <= 28.0:
        return 0.9
    return 1.2


def _aid_station_tier_on_site(tier: str) -> tuple[float, float]:
    tier_defaults = {
        "water_only": (0.0, 0.0),
        "standard": (150.0, 30.0),
        "full_service": (500.0, 80.0),
    }
    return tier_defaults.get(tier, tier_defaults["standard"])


def _carry_items_for_block(carb_target_g: float, gut_tolerance_g_hr: float | None) -> list[str]:
    if carb_target_g <= 0:
        return []
    gels_needed = max(1, int(math.ceil(carb_target_g / 23.0)))
    items: list[str] = []
    for i in range(min(gels_needed, 8)):
        items.append(f"Gel {i + 1} (~23g carbs)")
    return items


def build_fueling_plan(
    plan_result: PlanResult,
    mass_kg: float,
    peak_temperature_c: float | None,
    aid_station_tiers: list[str] | None,
    athlete_sweat_rate: float | None = None,
    gut_carb_tolerance_g_hr: float | None = None,
) -> FuelingPlan:
    aid_station_tiers = aid_station_tiers or []
    total_kcal = estimate_event_kcal(mass_kg, plan_result.splits)
    duration_hr = plan_result.moving_time_min / 60.0
    avg_kcal_hr = total_kcal / max(duration_hr, 0.01)
    carb_low, carb_high = _carb_target_band_g_per_hr(duration_hr)
    effective_carb_target = gut_carb_tolerance_g_hr or carb_high
    total_carb_target_g = effective_carb_target * duration_hr
    sweat_rate = _sweat_rate_l_per_hr(peak_temperature_c or 18.0, athlete_sweat_rate)
    total_fluid_target_l = sweat_rate * duration_hr

    aid_distances = [eta.distance_km for eta in plan_result.aid_station_etas]
    boundaries = [0.0, *aid_distances, plan_result.total_distance_km]
    aid_etas = list(plan_result.aid_station_etas)

    blocks: list[FuelingBlock] = []
    cumulative_carb_planned = 0.0
    cumulative_carb_target = 0.0

    for block_index in range(len(boundaries) - 1):
        start_km = boundaries[block_index]
        end_km = boundaries[block_index + 1]
        distance_km = end_km - start_km
        if distance_km <= 0:
            continue

        if block_index < len(aid_etas):
            block_duration = (
                aid_etas[block_index].split_from_prev_min
                if block_index > 0
                else aid_etas[0].arrival_moving_time_min
            )
        else:
            block_duration = plan_result.moving_time_min - (
                aid_etas[-1].arrival_moving_time_min if aid_etas else 0.0
            )
        block_duration = max(block_duration, 0.0)
        block_duration_hr = block_duration / 60.0

        block_splits = [split for split in plan_result.splits if start_km < split.km <= end_km]
        block_kcal = (
            estimate_event_kcal(mass_kg, block_splits)
            if block_splits
            else mass_kg * distance_km * _BASE_RUNNING_COST_KCAL_PER_KG_KM
        )
        block_carb_target = effective_carb_target * block_duration_hr
        block_fluid_target = sweat_rate * block_duration_hr

        tier = "standard"
        on_site_kcal = 0.0
        on_site_carb = 0.0
        if block_index < len(aid_etas):
            aid_tier_index = block_index
            if aid_tier_index < len(aid_station_tiers):
                tier = aid_station_tiers[aid_tier_index]
            on_site_kcal, on_site_carb = _aid_station_tier_on_site(tier)

        moving_carb = min(block_carb_target, effective_carb_target * block_duration_hr)
        carb_planned = moving_carb + on_site_carb

        cumulative_carb_target += block_carb_target
        cumulative_carb_planned += carb_planned

        carry_items = _carry_items_for_block(
            max(block_carb_target - on_site_carb, 0.0), gut_carb_tolerance_g_hr
        )

        blocks.append(
            FuelingBlock(
                start_km=round(start_km, 2),
                end_km=round(end_km, 2),
                distance_km=round(distance_km, 2),
                duration_min=round(block_duration, 1),
                kcal_burned=round(block_kcal, 1),
                carb_target_g=round(block_carb_target, 1),
                carb_planned_g=round(carb_planned, 1),
                fluid_target_l=round(block_fluid_target, 2),
                carry_items=carry_items,
                aid_station_tier=tier,
                on_site_kcal=round(on_site_kcal, 1),
                on_site_carb_g=round(on_site_carb, 1),
                cumulative_carb_deficit_g=round(
                    max(cumulative_carb_target - cumulative_carb_planned, 0.0), 1
                ),
            )
        )

    carb_deficit_g = max(total_carb_target_g - cumulative_carb_planned, 0.0)
    warnings: list[str] = []
    if duration_hr > 3.0 and carb_deficit_g > total_carb_target_g * 0.2:
        warnings.append(
            f"Planned carb intake falls {carb_deficit_g:.0f}g short of target. Consider increasing fueling frequency."
        )
    if duration_hr > 6.0:
        warnings.append(
            "Late-race gut stress may reduce absorption by 30-50%. Consider front-loading carb intake."
        )

    return FuelingPlan(
        total_kcal=round(total_kcal, 1),
        avg_kcal_hr=round(avg_kcal_hr, 1),
        total_carb_target_g=round(total_carb_target_g, 1),
        total_carb_planned_g=round(cumulative_carb_planned, 1),
        total_fluid_target_l=round(total_fluid_target_l, 2),
        blocks=blocks,
        carb_deficit_g=round(carb_deficit_g, 1),
        warnings=warnings,
    )

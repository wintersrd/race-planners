from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from race_planners.grade import gap_factor


def progressive_fire_road_bias(
    progress_ratio: float, elapsed_hours: float, climb_m_per_km: float
) -> float:
    """Return Z2->Z1 drift ratio for fire-road ultras.

    Returns value in [0, 1], where:
    - 0.0 means strongly Z2-biased
    - 1.0 means strongly Z1-biased
    """
    progress_component = min(max(progress_ratio, 0.0), 1.0)
    time_component = min(max(elapsed_hours / 12.0, 0.0), 1.0)
    climb_component = min(max(climb_m_per_km / 35.0, 0.0), 1.0)
    return min(
        1.0, (0.45 * progress_component) + (0.35 * time_component) + (0.20 * climb_component)
    )


def blend_z1_z2(z1_pace_min_km: float, z2_pace_min_km: float, z1_bias: float) -> float:
    clamped_bias = min(max(z1_bias, 0.0), 1.0)
    return (z2_pace_min_km * (1.0 - clamped_bias)) + (z1_pace_min_km * clamped_bias)


def apply_hike_switch(
    running_pace_min_km: float,
    hike_pace_min_km: float,
    grade_percent: float,
    threshold_percent: float,
) -> float:
    if grade_percent >= threshold_percent:
        return max(running_pace_min_km, hike_pace_min_km)
    return running_pace_min_km


def technical_descent_multiplier(grade_percent: float, caution: str) -> float:
    """Return multiplier against GAP pace for technical descents.

    Multiplier >1 slows the pace relative to GAP.
    """
    caution_map = {
        "low": 0.7,
        "medium": 1.0,
        "high": 1.3,
    }
    caution_scale = caution_map.get(caution, 1.0)

    if grade_percent >= 0:
        return 1.0

    steepness = abs(grade_percent)
    if steepness <= 5:
        return 1.0 + (0.01 * caution_scale)
    if steepness <= 10:
        return 1.0 + (0.03 * caution_scale)
    if steepness <= 15:
        return 1.0 + (0.08 * caution_scale)
    return 1.0 + (0.18 * caution_scale)


def technical_trail_pace(
    flat_pace_min_km: float,
    hike_pace_min_km: float,
    grade_percent: float,
    climb_hike_threshold_percent: float,
    descent_caution: str,
) -> float:
    uphill_or_flat_gap = flat_pace_min_km * gap_factor(grade_percent)
    pace_after_hike = apply_hike_switch(
        running_pace_min_km=uphill_or_flat_gap,
        hike_pace_min_km=hike_pace_min_km,
        grade_percent=grade_percent,
        threshold_percent=climb_hike_threshold_percent,
    )

    if grade_percent < 0:
        pace_after_hike *= technical_descent_multiplier(grade_percent, descent_caution)
        if grade_percent <= -18:
            pace_after_hike = max(pace_after_hike, flat_pace_min_km * 1.05)

    return pace_after_hike


@dataclass
class PacingContext:
    grade_percent: float
    progress_ratio: float
    elapsed_hours: float
    climb_m_per_km: float
    steepest_climb_percent: float = 0.0
    steepest_descent_percent: float = 0.0


class PacingModel(Protocol):
    def pace_for_context(self, context: PacingContext) -> float: ...


@dataclass
class FireRoadUltraModel:
    z1_pace_min_km: float
    z2_pace_min_km: float
    hike_pace_min_km: float
    hike_threshold_percent: float = 12.0

    def pace_for_context(self, context: PacingContext) -> float:
        z1_bias = progressive_fire_road_bias(
            progress_ratio=context.progress_ratio,
            elapsed_hours=context.elapsed_hours,
            climb_m_per_km=context.climb_m_per_km,
        )
        flat_base = blend_z1_z2(self.z1_pace_min_km, self.z2_pace_min_km, z1_bias)
        run_pace = flat_base * gap_factor(context.grade_percent)
        return apply_hike_switch(
            running_pace_min_km=run_pace,
            hike_pace_min_km=self.hike_pace_min_km,
            grade_percent=max(context.grade_percent, context.steepest_climb_percent),
            threshold_percent=self.hike_threshold_percent,
        )


@dataclass
class TechnicalTrailUltraModel:
    flat_pace_min_km: float
    hike_pace_min_km: float
    hike_threshold_percent: float = 12.0
    descent_caution: str = "medium"

    def pace_for_context(self, context: PacingContext) -> float:
        return technical_trail_pace(
            flat_pace_min_km=self.flat_pace_min_km,
            hike_pace_min_km=self.hike_pace_min_km,
            grade_percent=max(context.grade_percent, context.steepest_climb_percent),
            climb_hike_threshold_percent=self.hike_threshold_percent,
            descent_caution=self.descent_caution,
        )


@dataclass
class GapEffortModel:
    """Simple GAP-based effort model for half and marathon pacing."""

    base_pace_min_km: float

    def pace_for_context(self, context: PacingContext) -> float:
        return self.base_pace_min_km * gap_factor(context.grade_percent)

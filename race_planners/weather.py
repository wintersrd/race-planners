from __future__ import annotations

import math

from race_planners.fatigue import tolerance_penalty_scale
from race_planners.models import PacingConfig


TEMP_PENALTY_TABLE: tuple[tuple[float, float], ...] = (
    (10.0, 0.0),
    (15.0, 0.015),
    (20.0, 0.035),
    (25.0, 0.065),
    (28.0, 0.085),
    (30.0, 0.10),
    (35.0, 0.15),
)


def heat_multiplier(temperature_c: float) -> float:
    if temperature_c <= 10.0:
        return 1.0
    if temperature_c >= 35.0:
        return 1.15 + (temperature_c - 35.0) * 0.005
    for i in range(len(TEMP_PENALTY_TABLE) - 1):
        low_temp, low_penalty = TEMP_PENALTY_TABLE[i]
        high_temp, high_penalty = TEMP_PENALTY_TABLE[i + 1]
        if low_temp <= temperature_c <= high_temp:
            fraction = (temperature_c - low_temp) / (high_temp - low_temp)
            return 1.0 + low_penalty + (high_penalty - low_penalty) * fraction
    return 1.0


def _seasonal_params(event_month: int | None) -> tuple[float, float]:
    """Return (diurnal_swing_c, coldest_hour) for a given month.

    Heuristic model for mid-latitude Northern Hemisphere (~48°N):
    - Swing peaks in summer (~15°C) and bottoms in winter (~6°C)
    - Coldest point shifts slightly later in winter (dawn is later)
    """
    if event_month is None:
        return 12.0, 4.0
    # Summer solstice ~month 6 (June), winter ~month 12 (December)
    seasonal_angle = 2.0 * math.pi * (event_month - 6) / 12.0
    swing = 10.5 + 4.5 * math.cos(seasonal_angle)
    coldest_hour = 4.5 - 0.5 * math.cos(seasonal_angle)
    return swing, coldest_hour


def temperature_at_elapsed(
    peak_temp_c: float,
    elapsed_hours: float,
    start_time_local: str | None,
    event_month: int | None = None,
) -> float:
    swing, coldest_hour = _seasonal_params(event_month)
    temp_min = peak_temp_c - swing
    if start_time_local:
        try:
            parts = start_time_local.split(":")
            start_hour = float(parts[0]) + (float(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
        except (ValueError, IndexError):
            start_hour = 9.0
    else:
        start_hour = 9.0
    wall_hour = (start_hour + elapsed_hours) % 24.0
    phase = 2.0 * math.pi * ((wall_hour - coldest_hour) / 24.0)
    return temp_min + swing * 0.5 * (1.0 - math.cos(phase))


def segment_heat_multiplier(config: PacingConfig, cumulative_time_min: float) -> float:
    if config.peak_temperature_c is None:
        return 1.0
    temp = temperature_at_elapsed(
        config.peak_temperature_c,
        cumulative_time_min / 60.0,
        config.event_start_time_local,
        getattr(config, "event_month", None),
    )
    raw_heat_multiplier = heat_multiplier(temp)
    return 1.0 + (
        (raw_heat_multiplier - 1.0) * tolerance_penalty_scale(config.athlete_heat_tolerance)
    )


def average_heat_multiplier_for_duration(
    peak_temperature_c: float | None,
    start_time_local: str | None,
    duration_min: float,
    event_month: int | None = None,
) -> float:
    if peak_temperature_c is None or duration_min <= 0:
        return 1.0

    sample_count = max(3, min(24, int(math.ceil(duration_min / 30.0))))
    total_multiplier = 0.0
    for index in range(sample_count):
        elapsed_hours = (duration_min * ((index + 0.5) / sample_count)) / 60.0
        total_multiplier += heat_multiplier(
            temperature_at_elapsed(peak_temperature_c, elapsed_hours, start_time_local, event_month)
        )
    return total_multiplier / sample_count

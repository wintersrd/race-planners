from __future__ import annotations

from race_planners.fatigue import tolerance_penalty_scale
from race_planners.grade import estimate_course_gap_multiplier
from race_planners.models import LoadedCourse
from race_planners.weather import average_heat_multiplier_for_duration


ROAD_EFFORT_FRACTION_TABLE: dict[str, list[tuple[float, float]]] = {
    "half_marathon": [
        (80.0, 0.94),
        (90.0, 0.90),
        (100.0, 0.85),
        (110.0, 0.79),
        (125.0, 0.72),
        (140.0, 0.62),
        (160.0, 0.50),
    ],
    "road_marathon": [
        (165.0, 0.74),
        (180.0, 0.70),
        (195.0, 0.65),
        (210.0, 0.60),
        (225.0, 0.55),
        (240.0, 0.50),
        (270.0, 0.45),
        (300.0, 0.35),
    ],
}


def road_race_distance_km(race_model: str) -> float | None:
    distance_map = {
        "half_marathon": 21.0975,
        "road_marathon": 42.195,
    }
    return distance_map.get(race_model)


def interpolate_road_effort_fraction(duration_min: float, race_model: str) -> float:
    points = ROAD_EFFORT_FRACTION_TABLE.get(race_model)
    if not points:
        return 0.5
    if duration_min <= points[0][0]:
        return points[0][1]
    for (start_duration, start_fraction), (end_duration, end_fraction) in zip(
        points, points[1:], strict=False
    ):
        if duration_min <= end_duration:
            blend = (duration_min - start_duration) / max(end_duration - start_duration, 0.01)
            return start_fraction + ((end_fraction - start_fraction) * blend)
    return points[-1][1]


def estimate_road_best_likely_pace_min_km(
    race_model: str,
    lt1_pace_min_km: float,
    lt2_pace_min_km: float | None,
) -> float:
    race_distance_km = road_race_distance_km(race_model)
    if race_distance_km is None or lt2_pace_min_km is None:
        return round(float(lt1_pace_min_km), 2)

    lt1_pace = float(lt1_pace_min_km)
    lt2_pace = float(lt2_pace_min_km)
    duration_guess = race_distance_km * ((lt1_pace + lt2_pace) / 2.0)
    pace_min_km = lt1_pace

    for _ in range(24):
        effort_fraction = interpolate_road_effort_fraction(duration_guess, race_model)
        pace_min_km = lt1_pace - ((lt1_pace - lt2_pace) * effort_fraction)
        updated_duration = pace_min_km * race_distance_km
        if abs(updated_duration - duration_guess) < 0.01:
            duration_guess = updated_duration
            break
        duration_guess = updated_duration

    return round(pace_min_km, 2)


def estimate_road_best_likely_time_min(
    race_model: str,
    lt1_pace_min_km: float,
    lt2_pace_min_km: float | None,
) -> float:
    race_distance_km = road_race_distance_km(race_model)
    if race_distance_km is None:
        return 0.0
    pace_min_km = estimate_road_best_likely_pace_min_km(
        race_model, lt1_pace_min_km, lt2_pace_min_km
    )
    return round(pace_min_km * race_distance_km, 2)


def estimate_road_adjusted_best_likely(
    loaded_course: LoadedCourse,
    base_time_min: float,
    peak_temperature_c: float | None,
    start_time_local: str | None,
    hill_tolerance: float | None,
    heat_tolerance: float | None,
) -> dict[str, float]:
    course_gap_multiplier = estimate_course_gap_multiplier(
        loaded_course.trackpoints,
        loaded_course.total_distance_km,
    )
    course_multiplier = 1.0 + (
        (course_gap_multiplier - 1.0) * tolerance_penalty_scale(hill_tolerance)
    )

    adjusted_time_min = base_time_min * course_multiplier
    average_heat_multiplier = 1.0
    weather_multiplier = 1.0
    for _ in range(4):
        average_heat_multiplier = average_heat_multiplier_for_duration(
            peak_temperature_c,
            start_time_local,
            adjusted_time_min,
        )
        weather_multiplier = 1.0 + (
            (average_heat_multiplier - 1.0) * tolerance_penalty_scale(heat_tolerance)
        )
        adjusted_time_min = base_time_min * course_multiplier * weather_multiplier

    return {
        "base_time_min": round(base_time_min, 2),
        "course_multiplier": round(course_multiplier, 4),
        "weather_multiplier": round(weather_multiplier, 4),
        "average_heat_multiplier": round(average_heat_multiplier, 4),
        "adjusted_time_min": round(adjusted_time_min, 2),
    }


def estimate_road_intent_target_time_min(
    adjusted_best_likely_time_min: float, race_intent: str
) -> float:
    intent_multiplier = {
        "best_effort": 1.00,
        "strong": 1.02,
        "controlled": 1.05,
        "easy_durable": 1.09,
    }.get(race_intent, 1.05)
    return round(adjusted_best_likely_time_min * intent_multiplier, 2)


def road_target_delta_percent(
    adjusted_best_likely_time_min: float, chosen_target_time_min: float
) -> float:
    if adjusted_best_likely_time_min <= 0:
        return 0.0
    return (
        (chosen_target_time_min - adjusted_best_likely_time_min) / adjusted_best_likely_time_min
    ) * 100.0


def classify_road_feasibility(
    adjusted_best_likely_time_min: float, chosen_target_time_min: float
) -> str:
    delta_percent = road_target_delta_percent(adjusted_best_likely_time_min, chosen_target_time_min)
    if delta_percent >= 6.0:
        return "Very High"
    if delta_percent >= 2.0:
        return "Reasonable"
    if delta_percent >= -1.5:
        return "Stretch"
    return "Aggressive"


def classify_road_effort_band(
    adjusted_best_likely_time_min: float, chosen_target_time_min: float
) -> str:
    delta_percent = road_target_delta_percent(adjusted_best_likely_time_min, chosen_target_time_min)
    if delta_percent >= 8.0:
        return "Controlled"
    if delta_percent >= 3.0:
        return "Strong"
    if delta_percent >= -1.0:
        return "Near Limit"
    return "Maximal"


def classify_road_recovery_cost(
    adjusted_best_likely_time_min: float, chosen_target_time_min: float
) -> str:
    delta_percent = road_target_delta_percent(adjusted_best_likely_time_min, chosen_target_time_min)
    if delta_percent >= 8.0:
        return "Low"
    if delta_percent >= 3.0:
        return "Moderate"
    if delta_percent >= -1.0:
        return "High"
    return "Very High"

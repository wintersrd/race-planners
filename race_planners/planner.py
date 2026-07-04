from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from race_planners.fatigue import (
    fade_profile_values,
    fatigue_multiplier,
    is_road_race_model,
    pacing_shape_multiplier,
    durability_multiplier,
    tolerance_penalty_scale,
    trail_hill_tolerance_multiplier,
)
from race_planners.grade import (
    calculate_segment_grades,
    extreme_grade_in_range,
    gap_factor,
    parse_gpx,
    smooth_elevation,
    weighted_average_grade,
)
from race_planners.models import (
    AidStationEta,
    AidStation,
    Course,
    PaceSplit,
    PacingConfig,
    PlanResult,
    TrackPoint,
)
from race_planners.pacing import (
    FireRoadUltraModel,
    GapEffortModel,
    PacingContext,
    PacingModel,
    TechnicalTrailUltraModel,
)
from race_planners.segments import build_segment_summaries
from race_planners.weather import (
    average_heat_multiplier_for_duration,
    segment_heat_multiplier,
)


@dataclass
class LoadedCourse:
    course: Course
    trackpoints: list[TrackPoint]
    total_distance_km: float


_ROAD_EFFORT_FRACTION_TABLE: dict[str, list[tuple[float, float]]] = {
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


def _interpolate_road_effort_fraction(duration_min: float, race_model: str) -> float:
    points = _ROAD_EFFORT_FRACTION_TABLE.get(race_model)
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
        effort_fraction = _interpolate_road_effort_fraction(duration_guess, race_model)
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
        return round(float(lt1_pace_min_km), 2)
    pace_min_km = estimate_road_best_likely_pace_min_km(
        race_model,
        lt1_pace_min_km,
        lt2_pace_min_km,
    )
    return round(pace_min_km * race_distance_km, 2)


def load_course_trackpoints(course: Course, smoothing_window: int = 5) -> LoadedCourse:
    raw_points = parse_gpx(str(course.gpx_path))
    smooth_points = smooth_elevation(raw_points, window_size=smoothing_window)
    graded_points = calculate_segment_grades(smooth_points)
    total_distance_km = graded_points[-1].distance_from_start / 1000 if graded_points else 0.0
    return LoadedCourse(
        course=course,
        trackpoints=graded_points,
        total_distance_km=total_distance_km,
    )


def _estimate_course_gap_multiplier(
    trackpoints: list[TrackPoint], total_distance_km: float
) -> float:
    if total_distance_km <= 0:
        return 1.0
    grade_sum = 0.0
    total_segments = max(1, int(total_distance_km))
    for km in range(1, total_segments + 1):
        start_m = (km - 1) * 1000
        end_m = min(km * 1000, total_distance_km * 1000)
        grade_sum += gap_factor(weighted_average_grade(trackpoints, start_m, end_m))
    return grade_sum / total_segments


def _build_model(
    config: PacingConfig, trackpoints: list[TrackPoint], total_distance_km: float
) -> PacingModel:
    if config.race_model == "fire_road_ultra":
        if (
            config.z1_pace_min_km is None
            or config.z2_pace_min_km is None
            or config.hike_pace_min_km is None
        ):
            raise ValueError("fire_road_ultra requires Z1 pace, Z2 pace, and hike pace")
        return FireRoadUltraModel(
            z1_pace_min_km=config.z1_pace_min_km,
            z2_pace_min_km=config.z2_pace_min_km,
            hike_pace_min_km=config.hike_pace_min_km,
            hike_threshold_percent=config.climb_hike_threshold_percent,
        )

    if config.race_model == "technical_trail_ultra":
        if config.flat_pace_min_km is None or config.hike_pace_min_km is None:
            raise ValueError("technical_trail_ultra requires flat pace and hike pace")
        return TechnicalTrailUltraModel(
            flat_pace_min_km=config.flat_pace_min_km,
            hike_pace_min_km=config.hike_pace_min_km,
            hike_threshold_percent=config.climb_hike_threshold_percent,
            descent_caution=config.descent_caution,
        )

    if config.marathon_pace_min_km is not None:
        return GapEffortModel(base_pace_min_km=config.marathon_pace_min_km)

    if config.target_finish_time_min is not None and total_distance_km > 0:
        avg_course_gap = _estimate_course_gap_multiplier(trackpoints, total_distance_km)
        base_pace = (config.target_finish_time_min / total_distance_km) / max(avg_course_gap, 0.8)
        return GapEffortModel(base_pace_min_km=base_pace)

    raise ValueError("road/half model requires marathon pace or target finish time")


def _target_running_time_min(config: PacingConfig, aid_stop_count: int) -> float | None:
    if config.target_finish_time_min is None:
        return None

    total_rest_time_min = max(aid_stop_count, 0) * (config.rest_duration_sec / 60.0)
    running_time_min = config.target_finish_time_min - total_rest_time_min
    if running_time_min <= 0:
        raise ValueError("Rest time exceeds target finish time")
    return running_time_min


def _valid_aid_stations(course: Course, total_distance_km: float) -> list[AidStation]:
    return [
        aid_station
        for aid_station in course.aid_stations
        if 0 < aid_station.distance_km <= total_distance_km
    ]


def _normalize_config(
    config: PacingConfig,
    trackpoints: list[TrackPoint],
    total_distance_km: float,
    aid_stop_count: int,
) -> PacingConfig:
    target_running_time_min = _target_running_time_min(config, aid_stop_count)
    if target_running_time_min is None or total_distance_km <= 0:
        return config

    config = replace(config, target_finish_time_min=target_running_time_min)
    normalized_target_finish_time_min = target_running_time_min

    if config.race_model == "technical_trail_ultra" and (
        config.flat_pace_min_km is None or config.hike_pace_min_km is None
    ):
        flat_pace_min_km = _solve_base_pace_for_target(
            race_model=config.race_model,
            target_finish_time_min=normalized_target_finish_time_min,
            build_model=lambda base_pace: TechnicalTrailUltraModel(
                flat_pace_min_km=base_pace,
                hike_pace_min_km=max(base_pace * 1.55, base_pace + 4.0),
                hike_threshold_percent=config.climb_hike_threshold_percent,
                descent_caution=config.descent_caution,
            ),
            trackpoints=trackpoints,
            total_distance_km=total_distance_km,
            pacing_bias=config.pacing_bias,
            fade_profile_preset=config.fade_profile_preset,
            fade_early_bias=config.fade_early_bias,
            fade_mid_bias=config.fade_mid_bias,
            fade_late_bias=config.fade_late_bias,
            rpe_target=config.rpe_target,
            hr_cap=config.hr_cap,
            peak_temperature_c=config.peak_temperature_c,
            event_start_time_local=config.event_start_time_local,
        )
        return replace(
            config,
            flat_pace_min_km=round(flat_pace_min_km, 2),
            hike_pace_min_km=round(max(flat_pace_min_km * 1.55, flat_pace_min_km + 4.0), 2),
        )

    if config.race_model == "fire_road_ultra" and (
        config.z1_pace_min_km is None
        or config.z2_pace_min_km is None
        or config.hike_pace_min_km is None
    ):
        z2_pace_min_km = _solve_base_pace_for_target(
            race_model=config.race_model,
            target_finish_time_min=normalized_target_finish_time_min,
            build_model=lambda base_pace: FireRoadUltraModel(
                z1_pace_min_km=max(base_pace * 1.12, base_pace + 0.6),
                z2_pace_min_km=base_pace,
                hike_pace_min_km=max(base_pace * 1.75, base_pace + 4.0),
                hike_threshold_percent=config.climb_hike_threshold_percent,
            ),
            trackpoints=trackpoints,
            total_distance_km=total_distance_km,
            pacing_bias=config.pacing_bias,
            fade_profile_preset=config.fade_profile_preset,
            fade_early_bias=config.fade_early_bias,
            fade_mid_bias=config.fade_mid_bias,
            fade_late_bias=config.fade_late_bias,
            rpe_target=config.rpe_target,
            hr_cap=config.hr_cap,
            peak_temperature_c=config.peak_temperature_c,
            event_start_time_local=config.event_start_time_local,
        )
        return replace(
            config,
            z1_pace_min_km=round(max(z2_pace_min_km * 1.12, z2_pace_min_km + 0.6), 2),
            z2_pace_min_km=round(z2_pace_min_km, 2),
            hike_pace_min_km=round(max(z2_pace_min_km * 1.75, z2_pace_min_km + 4.0), 2),
        )

    return config


def _solve_base_pace_for_target(
    race_model: str,
    target_finish_time_min: float,
    build_model: Callable[[float], PacingModel],
    trackpoints: list[TrackPoint],
    total_distance_km: float,
    pacing_bias: float = 0.0,
    fade_profile_preset: str | None = None,
    fade_early_bias: float | None = None,
    fade_mid_bias: float | None = None,
    fade_late_bias: float | None = None,
    rpe_target: float | None = None,
    hr_cap: int | None = None,
    peak_temperature_c: float | None = None,
    event_start_time_local: str | None = None,
) -> float:
    low = 1.0
    high = 60.0
    for _ in range(32):
        mid = (low + high) / 2
        total_time_min = _simulate_total_time(
            race_model=race_model,
            model=build_model(mid),
            trackpoints=trackpoints,
            total_distance_km=total_distance_km,
            pacing_bias=pacing_bias,
            fade_profile_preset=fade_profile_preset,
            fade_early_bias=fade_early_bias,
            fade_mid_bias=fade_mid_bias,
            fade_late_bias=fade_late_bias,
            rpe_target=rpe_target,
            hr_cap=hr_cap,
            peak_temperature_c=peak_temperature_c,
            event_start_time_local=event_start_time_local,
        )
        if total_time_min < target_finish_time_min:
            low = mid
        else:
            high = mid
    return high


def _simulate_total_time(
    race_model: str,
    model: PacingModel,
    trackpoints: list[TrackPoint],
    total_distance_km: float,
    pacing_bias: float = 0.0,
    fade_profile_preset: str | None = None,
    fade_early_bias: float | None = None,
    fade_mid_bias: float | None = None,
    fade_late_bias: float | None = None,
    rpe_target: float | None = None,
    hr_cap: int | None = None,
    peak_temperature_c: float | None = None,
    event_start_time_local: str | None = None,
) -> float:
    config = PacingConfig(
        race_model=race_model,
        input_mode="effort_anchor",
        pacing_bias=pacing_bias,
        fade_profile_preset=fade_profile_preset,
        fade_early_bias=fade_early_bias,
        fade_mid_bias=fade_mid_bias,
        fade_late_bias=fade_late_bias,
        rpe_target=rpe_target,
        hr_cap=hr_cap,
        peak_temperature_c=peak_temperature_c,
        event_start_time_local=event_start_time_local,
    )
    cumulative_time = 0.0

    full_km_count = int(total_distance_km)
    for km in range(1, full_km_count + 1):
        start_m: float = (km - 1) * 1000
        end_m: float = km * 1000
        context = _pacing_context_for_range(
            trackpoints, start_m, end_m, km / max(total_distance_km, 1.0), cumulative_time
        )
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= pacing_shape_multiplier(config, context.progress_ratio)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        cumulative_time += pace_min_km * fatigue_multiplier(race_model, context.progress_ratio)

    remaining = total_distance_km - full_km_count
    if remaining > 0.01:
        start_m = float(full_km_count * 1000)
        end_m = total_distance_km * 1000
        context = _pacing_context_for_range(trackpoints, start_m, end_m, 1.0, cumulative_time)
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= pacing_shape_multiplier(config, 1.0)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        cumulative_time += pace_min_km * remaining * fatigue_multiplier(race_model, 1.0)

    return cumulative_time


def calculate_plan(loaded_course: LoadedCourse, config: PacingConfig) -> PlanResult:
    trackpoints = loaded_course.trackpoints
    total_distance_km = loaded_course.total_distance_km
    valid_aid_stations = _valid_aid_stations(loaded_course.course, total_distance_km)
    config = _normalize_config(
        config,
        trackpoints,
        total_distance_km,
        len(valid_aid_stations),
    )
    model = _build_model(config, trackpoints, total_distance_km)

    splits: list[PaceSplit] = []
    cumulative_time = 0.0

    full_km_count = int(total_distance_km)
    for km in range(1, full_km_count + 1):
        start_m: float = (km - 1) * 1000
        end_m: float = km * 1000
        progress_ratio = km / max(total_distance_km, 1.0)
        context = _pacing_context_for_range(
            trackpoints, start_m, end_m, progress_ratio, cumulative_time
        )
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= pacing_shape_multiplier(config, progress_ratio)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= fatigue_multiplier(config.race_model, progress_ratio)
        cumulative_time += pace_min_km
        splits.append(
            PaceSplit(
                km=float(km),
                actual_pace_min_km=pace_min_km,
                grade_percent=context.grade_percent,
                segment_time_min=pace_min_km,
                cumulative_time_min=cumulative_time,
            )
        )

    remaining = total_distance_km - full_km_count
    if remaining > 0.01:
        start_m = float(full_km_count * 1000)
        end_m = total_distance_km * 1000
        context = _pacing_context_for_range(trackpoints, start_m, end_m, 1.0, cumulative_time)
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= pacing_shape_multiplier(config, 1.0)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= fatigue_multiplier(config.race_model, 1.0)
        segment_time = pace_min_km * remaining
        cumulative_time += segment_time
        splits.append(
            PaceSplit(
                km=round(total_distance_km, 2),
                actual_pace_min_km=pace_min_km,
                grade_percent=context.grade_percent,
                segment_time_min=segment_time,
                cumulative_time_min=cumulative_time,
            )
        )

    aid_arrival_times: list[float] = []
    aid_station_etas: list[AidStationEta] = []
    total_rest_time_min = len(valid_aid_stations) * (config.rest_duration_sec / 60.0)
    for stop_index, aid_station in enumerate(valid_aid_stations, start=1):
        aid_km = aid_station.distance_km
        elapsed = 0.0
        prev_km = 0.0
        for split in splits:
            split_end = split.km
            if aid_km <= split_end:
                split_length = split_end - prev_km
                if split_length > 0:
                    fraction = (aid_km - prev_km) / split_length
                else:
                    fraction = 0.0
                elapsed += split.segment_time_min * fraction
                break
            elapsed += split.segment_time_min
            prev_km = split_end
        aid_arrival_times.append(elapsed)
        prior_rest_time_min = (stop_index - 1) * (config.rest_duration_sec / 60.0)
        split_distance_km = aid_km - (aid_station_etas[-1].distance_km if aid_station_etas else 0.0)
        split_from_prev_min = elapsed - (
            aid_station_etas[-1].arrival_moving_time_min if aid_station_etas else 0.0
        )
        actual_pace_min_km = (
            split_from_prev_min / split_distance_km if split_distance_km > 0 else 0.0
        )
        aid_station_etas.append(
            AidStationEta(
                distance_km=aid_km,
                label=aid_station.label,
                source=aid_station.source,
                waypoint_type=aid_station.waypoint_type,
                arrival_moving_time_min=elapsed,
                arrival_elapsed_time_min=elapsed + prior_rest_time_min,
                departure_elapsed_time_min=elapsed
                + prior_rest_time_min
                + (config.rest_duration_sec / 60.0),
                split_from_prev_min=split_from_prev_min,
                split_distance_km=split_distance_km,
                actual_pace_min_km=actual_pace_min_km,
                suggested_rest_min=config.rest_duration_sec / 60.0,
            )
        )

    assumptions: list[str] = []
    warnings: list[str] = []
    if aid_station_etas and config.rest_duration_sec > 0:
        assumptions.append("Rest stops are modeled as fixed additive pauses.")
    if is_road_race_model(config.race_model) and config.pacing_bias != 0:
        assumptions.append("Pacing bias progressively shifts pace across the course.")
    if not is_road_race_model(config.race_model) and any(fade_profile_values(config)):
        assumptions.append("Fade profile progressively slows pace across the event.")
    if config.effort_policy is not None and config.effort_policy != "steady":
        assumptions.append("Effort policy nudges pacing more conservatively or aggressively.")
    elif config.rpe_target is not None and config.rpe_target != 6.0:
        assumptions.append("Effort policy nudges pacing more conservatively or aggressively.")
    hr_strategy_summary = _hr_guardrail_strategy_summary(config)
    if hr_strategy_summary is not None:
        assumptions.append(hr_strategy_summary)
    if config.use_hr_guardrail and config.hr_cap is not None:
        assumptions.append(
            "Derived HR guardrail tempers pacing on steeper or later-course segments."
        )
    elif config.hr_cap is not None and config.hr_cap != 155:
        assumptions.append(
            "Derived HR guardrail tempers pacing on steeper or later-course segments."
        )
    if config.peak_temperature_c is not None and config.peak_temperature_c > 10.0:
        assumptions.append(
            f"Weather model applies heat penalty using peak {config.peak_temperature_c:.0f}°C "
            "with diurnal temperature variation across the event."
        )

    return PlanResult(
        splits=splits,
        segments=build_segment_summaries(
            splits,
            aid_stations=valid_aid_stations,
            total_distance_km=total_distance_km,
            trackpoints=trackpoints,
        ),
        aid_arrival_times_min=aid_arrival_times,
        aid_station_etas=aid_station_etas,
        moving_time_min=cumulative_time,
        total_rest_time_min=total_rest_time_min,
        total_time_min=cumulative_time + total_rest_time_min,
        total_distance_km=total_distance_km,
        assumptions=assumptions,
        warnings=warnings,
    )


def estimate_road_adjusted_best_likely(
    loaded_course: LoadedCourse,
    base_time_min: float,
    peak_temperature_c: float | None,
    start_time_local: str | None,
    hill_tolerance: float | None,
    heat_tolerance: float | None,
) -> dict[str, float]:
    course_gap_multiplier = _estimate_course_gap_multiplier(
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


def _road_target_delta_percent(
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
    delta_percent = _road_target_delta_percent(
        adjusted_best_likely_time_min, chosen_target_time_min
    )
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
    delta_percent = _road_target_delta_percent(
        adjusted_best_likely_time_min, chosen_target_time_min
    )
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
    delta_percent = _road_target_delta_percent(
        adjusted_best_likely_time_min, chosen_target_time_min
    )
    if delta_percent >= 8.0:
        return "Low"
    if delta_percent >= 3.0:
        return "Moderate"
    if delta_percent >= -1.0:
        return "High"
    return "Very High"


def _effort_policy_rpe_target(config: PacingConfig) -> float | None:
    if config.rpe_target is not None:
        return config.rpe_target
    policy_map = {
        "conservative": 4.5,
        "steady": 6.0,
        "aggressive": 7.5,
    }
    return policy_map.get(config.effort_policy or "", None)


def _hr_guardrail_phase_offsets(config: PacingConfig) -> tuple[float, float, float]:
    offsets = {
        "conservative": (-3.0, 0.0, -4.0),
        "steady": (-1.5, 0.0, -2.5),
        "aggressive": (0.0, 1.0, -1.0),
    }
    return offsets.get(config.effort_policy or "steady", offsets["steady"])


def _interpolate_three_phase_values(
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


def _terrain_slowdown_minutes(config: PacingConfig, context: PacingContext) -> float:
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


def _road_equivalent_pace_min_km(
    config: PacingConfig,
    context: PacingContext,
    pace_min_km: float,
) -> float:
    return max(2.5, pace_min_km - _terrain_slowdown_minutes(config, context))


def _estimate_segment_hr(
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

    road_equivalent_pace = _road_equivalent_pace_min_km(config, context, pace_min_km)
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


def _dynamic_hr_guardrail_cap(config: PacingConfig, context: PacingContext) -> float | None:
    if not config.use_hr_guardrail or config.hr_cap is None:
        return None

    early_offset, mid_offset, late_offset = _hr_guardrail_phase_offsets(config)
    phase_cap = _interpolate_three_phase_values(
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


def _hr_guardrail_strategy_summary(config: PacingConfig) -> str | None:
    if not config.use_hr_guardrail or config.hr_cap is None:
        return None

    base_cap = float(config.hr_cap)
    early_offset, mid_offset, late_offset = _hr_guardrail_phase_offsets(config)
    early_cap = int(round(base_cap + early_offset))
    mid_cap = int(round(base_cap + mid_offset))
    late_cap = int(round(base_cap + late_offset))
    return f"Derived HR strategy targets roughly {early_cap}/{mid_cap}/{late_cap} bpm across early, mid, and late race phases."


def _effort_guardrail_multiplier(
    config: PacingConfig,
    context: PacingContext,
    pace_min_km: float,
) -> float:
    multiplier = 1.0
    rpe_target = _effort_policy_rpe_target(config)

    if rpe_target is not None:
        multiplier *= max(0.88, min(1.12, 1.0 - ((rpe_target - 6.0) * 0.02)))

    dynamic_hr_cap = _dynamic_hr_guardrail_cap(config, context)
    estimated_hr = _estimate_segment_hr(config, context, pace_min_km)
    if dynamic_hr_cap is not None and estimated_hr is not None:
        overage_bpm = max(estimated_hr - dynamic_hr_cap, 0.0)
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


def _pacing_context_for_range(
    trackpoints: list[TrackPoint],
    start_m: float,
    end_m: float,
    progress_ratio: float,
    cumulative_time_min: float,
) -> PacingContext:
    avg_grade = weighted_average_grade(trackpoints, start_m, end_m)
    steepest_climb = extreme_grade_in_range(trackpoints, start_m, end_m, uphill=True)
    steepest_descent = extreme_grade_in_range(trackpoints, start_m, end_m, uphill=False)
    return PacingContext(
        grade_percent=avg_grade,
        progress_ratio=progress_ratio,
        elapsed_hours=cumulative_time_min / 60,
        climb_m_per_km=max(avg_grade, 0.0) * 10,
        steepest_climb_percent=steepest_climb,
        steepest_descent_percent=steepest_descent,
    )

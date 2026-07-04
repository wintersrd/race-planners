from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

from race_planners.grade import (
    calculate_segment_grades,
    elevation_changes,
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
    SegmentSummary,
    TrackPoint,
)
from race_planners.pacing import (
    FireRoadUltraModel,
    GapEffortModel,
    PacingContext,
    PacingModel,
    TechnicalTrailUltraModel,
)


@dataclass
class LoadedCourse:
    course: Course
    trackpoints: list[TrackPoint]
    total_distance_km: float


_ROAD_EFFORT_FRACTION_TABLE: dict[str, list[tuple[float, float]]] = {
    "half_marathon": [
        (80.0, 0.97),
        (90.0, 0.93),
        (100.0, 0.88),
        (110.0, 0.82),
        (125.0, 0.72),
        (140.0, 0.62),
        (160.0, 0.50),
    ],
    "road_marathon": [
        (165.0, 0.88),
        (180.0, 0.83),
        (195.0, 0.77),
        (210.0, 0.70),
        (225.0, 0.63),
        (240.0, 0.56),
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
        pace_min_km *= _pacing_shape_multiplier(config, context.progress_ratio)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= _segment_heat_multiplier(config, cumulative_time)
        cumulative_time += pace_min_km * _fatigue_multiplier(race_model, context.progress_ratio)

    remaining = total_distance_km - full_km_count
    if remaining > 0.01:
        start_m = float(full_km_count * 1000)
        end_m = total_distance_km * 1000
        context = _pacing_context_for_range(trackpoints, start_m, end_m, 1.0, cumulative_time)
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= _pacing_shape_multiplier(config, 1.0)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= _segment_heat_multiplier(config, cumulative_time)
        cumulative_time += pace_min_km * remaining * _fatigue_multiplier(race_model, 1.0)

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
        pace_min_km *= _pacing_shape_multiplier(config, progress_ratio)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= _segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= _fatigue_multiplier(config.race_model, progress_ratio)
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
        pace_min_km *= _pacing_shape_multiplier(config, 1.0)
        pace_min_km *= _effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= _segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= _fatigue_multiplier(config.race_model, 1.0)
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
    if _is_road_race_model(config.race_model) and config.pacing_bias != 0:
        assumptions.append("Pacing bias progressively shifts pace across the course.")
    if not _is_road_race_model(config.race_model) and any(_fade_profile_values(config)):
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
        segments=_build_segment_summaries(
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


def _fatigue_multiplier(race_model: str, progress_ratio: float) -> float:
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


_TEMP_PENALTY_TABLE: tuple[tuple[float, float], ...] = (
    (10.0, 0.0),
    (15.0, 0.015),
    (20.0, 0.035),
    (25.0, 0.065),
    (28.0, 0.085),
    (30.0, 0.10),
    (35.0, 0.15),
)


def _heat_multiplier(temperature_c: float) -> float:
    if temperature_c <= 10.0:
        return 1.0
    if temperature_c >= 35.0:
        return 1.15 + (temperature_c - 35.0) * 0.005
    for i in range(len(_TEMP_PENALTY_TABLE) - 1):
        low_temp, low_penalty = _TEMP_PENALTY_TABLE[i]
        high_temp, high_penalty = _TEMP_PENALTY_TABLE[i + 1]
        if low_temp <= temperature_c <= high_temp:
            fraction = (temperature_c - low_temp) / (high_temp - low_temp)
            return 1.0 + low_penalty + (high_penalty - low_penalty) * fraction
    return 1.0


def _temperature_at_elapsed(
    peak_temp_c: float, elapsed_hours: float, start_time_local: str | None
) -> float:
    temp_min = peak_temp_c - 12.0
    if start_time_local:
        try:
            parts = start_time_local.split(":")
            start_hour = float(parts[0]) + (float(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
        except (ValueError, IndexError):
            start_hour = 9.0
    else:
        start_hour = 9.0
    wall_hour = (start_hour + elapsed_hours) % 24.0
    phase = 2.0 * math.pi * ((wall_hour - 4.0) / 24.0)
    return temp_min + (peak_temp_c - temp_min) * 0.5 * (1.0 - math.cos(phase))


def _segment_heat_multiplier(config: PacingConfig, cumulative_time_min: float) -> float:
    if config.peak_temperature_c is None:
        return 1.0
    temp = _temperature_at_elapsed(
        config.peak_temperature_c,
        cumulative_time_min / 60.0,
        config.event_start_time_local,
    )
    return _heat_multiplier(temp)


def _tolerance_penalty_scale(tolerance: float | None) -> float:
    if tolerance is None:
        return 1.0
    return min(max(1.0 - (float(tolerance) * 0.25), 0.6), 1.4)


def _average_heat_multiplier_for_duration(
    peak_temperature_c: float | None,
    start_time_local: str | None,
    duration_min: float,
) -> float:
    if peak_temperature_c is None or duration_min <= 0:
        return 1.0

    sample_count = max(3, min(24, int(math.ceil(duration_min / 30.0))))
    total_multiplier = 0.0
    for index in range(sample_count):
        elapsed_hours = (duration_min * ((index + 0.5) / sample_count)) / 60.0
        total_multiplier += _heat_multiplier(
            _temperature_at_elapsed(peak_temperature_c, elapsed_hours, start_time_local)
        )
    return total_multiplier / sample_count


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
        (course_gap_multiplier - 1.0) * _tolerance_penalty_scale(hill_tolerance)
    )

    adjusted_time_min = base_time_min * course_multiplier
    average_heat_multiplier = 1.0
    weather_multiplier = 1.0
    for _ in range(4):
        average_heat_multiplier = _average_heat_multiplier_for_duration(
            peak_temperature_c,
            start_time_local,
            adjusted_time_min,
        )
        weather_multiplier = 1.0 + (
            (average_heat_multiplier - 1.0) * _tolerance_penalty_scale(heat_tolerance)
        )
        adjusted_time_min = base_time_min * course_multiplier * weather_multiplier

    return {
        "base_time_min": round(base_time_min, 2),
        "course_multiplier": round(course_multiplier, 4),
        "weather_multiplier": round(weather_multiplier, 4),
        "average_heat_multiplier": round(average_heat_multiplier, 4),
        "adjusted_time_min": round(adjusted_time_min, 2),
    }


def _is_road_race_model(race_model: str) -> bool:
    return race_model in {"road_marathon", "half_marathon"}


def _pacing_bias_multiplier(pacing_bias: float, progress_ratio: float) -> float:
    ratio = min(max(progress_ratio, 0.0), 1.0)
    return max(0.85, 1.0 + (pacing_bias * 0.005 * ratio))


def _fade_profile_values(config: PacingConfig) -> tuple[float, float, float]:
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


def _interpolated_fade_bias(config: PacingConfig, progress_ratio: float) -> float:
    early_bias, mid_bias, late_bias = _fade_profile_values(config)
    ratio = min(max(progress_ratio, 0.0), 1.0)
    if ratio <= 0.33:
        return early_bias * (ratio / 0.33)
    if ratio <= 0.66:
        blend = (ratio - 0.33) / 0.33
        return early_bias + ((mid_bias - early_bias) * blend)
    blend = (ratio - 0.66) / 0.34
    return mid_bias + ((late_bias - mid_bias) * blend)


def _pacing_shape_multiplier(config: PacingConfig, progress_ratio: float) -> float:
    if _is_road_race_model(config.race_model):
        return _pacing_bias_multiplier(config.pacing_bias, progress_ratio)

    fade_bias = _interpolated_fade_bias(config, progress_ratio)
    return max(0.85, 1.0 + (fade_bias * 0.006))


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


def _segment_type(grade_percent: float) -> str:
    if grade_percent >= 2.0:
        return "climb"
    if grade_percent <= -2.0:
        return "descent"
    return "flat"


def _station_label(aid_station: AidStation, index: int) -> str:
    return aid_station.label or f"Aid {index + 1}"


@dataclass(frozen=True)
class _SegmentPiece:
    start_km: float
    end_km: float
    start_time_min: float
    end_time_min: float
    grade_percent: float
    pace_min_km: float


def _clip_split_pieces_to_block(
    splits: list[PaceSplit], block_start_km: float, block_end_km: float
) -> list[_SegmentPiece]:
    pieces: list[_SegmentPiece] = []
    prev_end_km = 0.0

    for split in splits:
        split_start_km = prev_end_km
        split_end_km = split.km
        split_start_time_min = split.cumulative_time_min - split.segment_time_min
        prev_end_km = split_end_km

        overlap_start_km = max(split_start_km, block_start_km)
        overlap_end_km = min(split_end_km, block_end_km)
        overlap_distance_km = overlap_end_km - overlap_start_km
        split_distance_km = split_end_km - split_start_km
        if overlap_distance_km <= 0 or split_distance_km <= 0:
            continue

        start_fraction = (overlap_start_km - split_start_km) / split_distance_km
        end_fraction = (overlap_end_km - split_start_km) / split_distance_km
        piece_start_time_min = split_start_time_min + (split.segment_time_min * start_fraction)
        piece_end_time_min = split_start_time_min + (split.segment_time_min * end_fraction)
        pieces.append(
            _SegmentPiece(
                start_km=overlap_start_km,
                end_km=overlap_end_km,
                start_time_min=piece_start_time_min,
                end_time_min=piece_end_time_min,
                grade_percent=split.grade_percent,
                pace_min_km=split.actual_pace_min_km,
            )
        )

    return pieces


def _build_segment_summaries(
    splits: list[PaceSplit],
    aid_stations: list[AidStation] | None = None,
    total_distance_km: float | None = None,
    trackpoints: list[TrackPoint] | None = None,
) -> list[SegmentSummary]:
    if not splits:
        return []

    course_distance_km = total_distance_km or splits[-1].km
    valid_aid_stations = sorted(
        [
            aid_station
            for aid_station in (aid_stations or [])
            if 0 < aid_station.distance_km < course_distance_km
        ],
        key=lambda aid_station: aid_station.distance_km,
    )
    boundaries_km = [
        0.0,
        *[aid_station.distance_km for aid_station in valid_aid_stations],
        course_distance_km,
    ]
    boundary_labels = [
        "Start",
        *[
            _station_label(aid_station, index)
            for index, aid_station in enumerate(valid_aid_stations)
        ],
        "Finish",
    ]

    segments: list[SegmentSummary] = []
    for block_index, (block_start_km, block_end_km) in enumerate(
        zip(boundaries_km, boundaries_km[1:], strict=False)
    ):
        block_pieces = _clip_split_pieces_to_block(splits, block_start_km, block_end_km)
        if not block_pieces:
            continue

        block_label = f"{boundary_labels[block_index]} to {boundary_labels[block_index + 1]}"
        current_type = _segment_type(block_pieces[0].grade_percent)
        start_km = block_pieces[0].start_km
        start_time_min = block_pieces[0].start_time_min
        distance_km = 0.0
        weighted_grade = 0.0
        weighted_pace = 0.0
        segment_time = 0.0
        prev_end_km = block_start_km
        end_time_min = start_time_min

        for piece in block_pieces:
            piece_distance_km = max(0.0, piece.end_km - piece.start_km)
            piece_type = _segment_type(piece.grade_percent)

            if piece_type != current_type and distance_km > 0:
                segments.append(
                    SegmentSummary(
                        segment_type=current_type,
                        block_label=block_label,
                        section_name=f"{block_label}: {current_type}",
                        start_km=start_km,
                        end_km=prev_end_km,
                        distance_km=distance_km,
                        start_time_min=start_time_min,
                        end_time_min=end_time_min,
                        avg_grade_percent=weighted_grade / distance_km,
                        avg_pace_min_km=weighted_pace / distance_km,
                        segment_time_min=segment_time,
                        elevation_gain_m=(
                            elevation_changes(trackpoints, start_km * 1000, prev_end_km * 1000)[0]
                            if trackpoints is not None
                            else 0.0
                        ),
                        elevation_loss_m=(
                            elevation_changes(trackpoints, start_km * 1000, prev_end_km * 1000)[1]
                            if trackpoints is not None
                            else 0.0
                        ),
                    )
                )
                current_type = piece_type
                start_km = piece.start_km
                start_time_min = piece.start_time_min
                distance_km = 0.0
                weighted_grade = 0.0
                weighted_pace = 0.0
                segment_time = 0.0

            distance_km += piece_distance_km
            weighted_grade += piece.grade_percent * piece_distance_km
            weighted_pace += piece.pace_min_km * piece_distance_km
            segment_time += piece.end_time_min - piece.start_time_min
            prev_end_km = piece.end_km
            end_time_min = piece.end_time_min

        if distance_km > 0:
            segments.append(
                SegmentSummary(
                    segment_type=current_type,
                    block_label=block_label,
                    section_name=f"{block_label}: {current_type}",
                    start_km=start_km,
                    end_km=prev_end_km,
                    distance_km=distance_km,
                    start_time_min=start_time_min,
                    end_time_min=end_time_min,
                    avg_grade_percent=weighted_grade / distance_km,
                    avg_pace_min_km=weighted_pace / distance_km,
                    segment_time_min=segment_time,
                    elevation_gain_m=(
                        elevation_changes(trackpoints, start_km * 1000, prev_end_km * 1000)[0]
                        if trackpoints is not None
                        else 0.0
                    ),
                    elevation_loss_m=(
                        elevation_changes(trackpoints, start_km * 1000, prev_end_km * 1000)[1]
                        if trackpoints is not None
                        else 0.0
                    ),
                )
            )

    return segments

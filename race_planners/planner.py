from __future__ import annotations

from dataclasses import replace
from typing import Callable

from race_planners.fatigue import (
    fade_profile_values,
    fatigue_multiplier,
    is_road_race_model,
    pacing_shape_multiplier,
    durability_multiplier,
    trail_hill_tolerance_multiplier,
)
from race_planners.grade import (
    calculate_segment_grades,
    elevation_changes,
    estimate_course_gap_multiplier,
    extreme_grade_in_range,
    parse_gpx,
    smooth_elevation,
    weighted_average_grade,
)
from race_planners.guardrails import (
    effort_guardrail_multiplier,
    hr_guardrail_strategy_summary,
)
from race_planners.models import (
    AidStationEta,
    AidStation,
    Course,
    LoadedCourse,
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
from race_planners.weather import segment_heat_multiplier


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
        avg_course_gap = estimate_course_gap_multiplier(trackpoints, total_distance_km)
        base_pace = (config.target_finish_time_min / total_distance_km) / max(avg_course_gap, 0.8)
        return GapEffortModel(base_pace_min_km=base_pace)

    raise ValueError("road/half model requires marathon pace or target finish time")


def _carried_weight_multiplier(config: PacingConfig, body_mass_kg: float | None) -> float:
    if config.carried_weight_kg <= 0 or body_mass_kg is None or body_mass_kg <= 0:
        return 1.0
    return 1.0 + (config.carried_weight_kg / body_mass_kg) * 0.8


def _rest_duration_sec_for_station(config: PacingConfig, aid_station: AidStation) -> float:
    if is_road_race_model(config.race_model):
        return float(config.rest_duration_sec)
    tier = getattr(aid_station, "tier", "standard") or "standard"
    if tier == "water_only":
        return float(config.rest_duration_water_only_sec)
    if tier == "full_service":
        return float(config.rest_duration_full_service_sec)
    if tier == "standard":
        return float(config.rest_duration_standard_sec)
    return float(config.rest_duration_sec)


def _total_rest_time_min(config: PacingConfig, valid_aid_stations: list[AidStation]) -> float:
    total = 0.0
    for station in valid_aid_stations:
        total += _rest_duration_sec_for_station(config, station) / 60.0
    return total


def _target_running_time_min(
    config: PacingConfig, valid_aid_stations: list[AidStation]
) -> float | None:
    if config.target_finish_time_min is None:
        return None

    total_rest_time_min = _total_rest_time_min(config, valid_aid_stations)
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
    valid_aid_stations: list[AidStation],
) -> PacingConfig:
    target_running_time_min = _target_running_time_min(config, valid_aid_stations)
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
        pace_min_km *= effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= _carried_weight_multiplier(config, config.athlete_body_mass_kg)
        cumulative_time += pace_min_km * fatigue_multiplier(race_model, context.progress_ratio)

    remaining = total_distance_km - full_km_count
    if remaining > 0.01:
        start_m = float(full_km_count * 1000)
        end_m = total_distance_km * 1000
        context = _pacing_context_for_range(trackpoints, start_m, end_m, 1.0, cumulative_time)
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= pacing_shape_multiplier(config, 1.0)
        pace_min_km *= effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= _carried_weight_multiplier(config, config.athlete_body_mass_kg)
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
        valid_aid_stations,
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
        pace_min_km *= effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= _carried_weight_multiplier(config, config.athlete_body_mass_kg)
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
        pace_min_km *= effort_guardrail_multiplier(config, context, pace_min_km)
        pace_min_km *= durability_multiplier(config, context)
        pace_min_km *= trail_hill_tolerance_multiplier(config, context)
        pace_min_km *= segment_heat_multiplier(config, cumulative_time)
        pace_min_km *= _carried_weight_multiplier(config, config.athlete_body_mass_kg)
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
    cumulative_rest_min = 0.0
    for aid_station in valid_aid_stations:
        aid_km = aid_station.distance_km
        prev_station_km = aid_station_etas[-1].distance_km if aid_station_etas else 0.0
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
        station_rest_min = _rest_duration_sec_for_station(config, aid_station) / 60.0
        prior_rest_time_min = cumulative_rest_min
        split_distance_km = aid_km - prev_station_km
        split_from_prev_min = elapsed - (
            aid_station_etas[-1].arrival_moving_time_min if aid_station_etas else 0.0
        )
        actual_pace_min_km = (
            split_from_prev_min / split_distance_km if split_distance_km > 0 else 0.0
        )
        block_gain_m, block_loss_m = elevation_changes(
            trackpoints, prev_station_km * 1000, aid_km * 1000
        )
        cumulative_rest_min += station_rest_min
        aid_station_etas.append(
            AidStationEta(
                distance_km=aid_km,
                label=aid_station.label,
                source=aid_station.source,
                waypoint_type=aid_station.waypoint_type,
                arrival_moving_time_min=elapsed,
                arrival_elapsed_time_min=elapsed + prior_rest_time_min,
                departure_elapsed_time_min=elapsed + prior_rest_time_min + station_rest_min,
                split_from_prev_min=split_from_prev_min,
                split_distance_km=split_distance_km,
                actual_pace_min_km=actual_pace_min_km,
                suggested_rest_min=station_rest_min,
                elevation_gain_m=round(block_gain_m, 1),
                elevation_loss_m=round(block_loss_m, 1),
            )
        )

    total_rest_time_min = cumulative_rest_min

    assumptions: list[str] = []
    warnings: list[str] = []
    if aid_station_etas and total_rest_time_min > 0:
        assumptions.append("assumption.rest_stops")
    if is_road_race_model(config.race_model) and config.pacing_bias != 0:
        assumptions.append("assumption.pacing_bias")
    if not is_road_race_model(config.race_model) and any(fade_profile_values(config)):
        assumptions.append("assumption.fade_profile")
    if config.effort_policy is not None and config.effort_policy != "steady":
        assumptions.append("assumption.effort_policy")
    elif config.rpe_target is not None and config.rpe_target != 6.0:
        assumptions.append("assumption.effort_policy")
    hr_strategy_summary = hr_guardrail_strategy_summary(config)
    if hr_strategy_summary is not None:
        assumptions.append(hr_strategy_summary)
    if config.use_hr_guardrail and config.hr_cap is not None:
        assumptions.append("assumption.hr_guardrail")
    elif config.hr_cap is not None and config.hr_cap != 155:
        assumptions.append("assumption.hr_guardrail")
    if config.peak_temperature_c is not None and config.peak_temperature_c > 10.0:
        assumptions.append(f"assumption.weather_heat|temp={config.peak_temperature_c:.0f}")

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

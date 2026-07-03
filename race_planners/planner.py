from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from race_planners.grade import (
    calculate_segment_grades,
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
) -> float:
    cumulative_time = 0.0

    full_km_count = int(total_distance_km)
    for km in range(1, full_km_count + 1):
        start_m: float = (km - 1) * 1000
        end_m: float = km * 1000
        avg_grade = weighted_average_grade(trackpoints, start_m, end_m)
        progress_ratio = km / max(total_distance_km, 1.0)
        context = PacingContext(
            grade_percent=avg_grade,
            progress_ratio=progress_ratio,
            elapsed_hours=cumulative_time / 60,
            climb_m_per_km=max(avg_grade, 0.0) * 10,
        )
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= _pacing_bias_multiplier(pacing_bias, progress_ratio)
        cumulative_time += pace_min_km * _fatigue_multiplier(race_model, progress_ratio)

    remaining = total_distance_km - full_km_count
    if remaining > 0.01:
        start_m = float(full_km_count * 1000)
        end_m = total_distance_km * 1000
        avg_grade = weighted_average_grade(trackpoints, start_m, end_m)
        context = PacingContext(
            grade_percent=avg_grade,
            progress_ratio=1.0,
            elapsed_hours=cumulative_time / 60,
            climb_m_per_km=max(avg_grade, 0.0) * 10,
        )
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= _pacing_bias_multiplier(pacing_bias, 1.0)
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
        avg_grade = weighted_average_grade(trackpoints, start_m, end_m)
        progress_ratio = km / max(total_distance_km, 1.0)
        context = PacingContext(
            grade_percent=avg_grade,
            progress_ratio=progress_ratio,
            elapsed_hours=cumulative_time / 60,
            climb_m_per_km=max(avg_grade, 0.0) * 10,
        )
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= _pacing_bias_multiplier(config.pacing_bias, progress_ratio)
        pace_min_km *= _fatigue_multiplier(config.race_model, progress_ratio)
        cumulative_time += pace_min_km
        splits.append(
            PaceSplit(
                km=float(km),
                actual_pace_min_km=pace_min_km,
                grade_percent=avg_grade,
                segment_time_min=pace_min_km,
                cumulative_time_min=cumulative_time,
            )
        )

    remaining = total_distance_km - full_km_count
    if remaining > 0.01:
        start_m = float(full_km_count * 1000)
        end_m = total_distance_km * 1000
        avg_grade = weighted_average_grade(trackpoints, start_m, end_m)
        context = PacingContext(
            grade_percent=avg_grade,
            progress_ratio=1.0,
            elapsed_hours=cumulative_time / 60,
            climb_m_per_km=max(avg_grade, 0.0) * 10,
        )
        pace_min_km = model.pace_for_context(context)
        pace_min_km *= _pacing_bias_multiplier(config.pacing_bias, 1.0)
        pace_min_km *= _fatigue_multiplier(config.race_model, 1.0)
        segment_time = pace_min_km * remaining
        cumulative_time += segment_time
        splits.append(
            PaceSplit(
                km=round(total_distance_km, 2),
                actual_pace_min_km=pace_min_km,
                grade_percent=avg_grade,
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
    if config.pacing_bias != 0:
        assumptions.append("Pacing bias progressively shifts pace across the course.")

    return PlanResult(
        splits=splits,
        segments=_build_segment_summaries(
            splits,
            aid_stations=valid_aid_stations,
            total_distance_km=total_distance_km,
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


def _pacing_bias_multiplier(pacing_bias: float, progress_ratio: float) -> float:
    ratio = min(max(progress_ratio, 0.0), 1.0)
    return max(0.85, 1.0 + (pacing_bias * 0.005 * ratio))


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
                )
            )

    return segments

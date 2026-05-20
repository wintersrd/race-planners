from __future__ import annotations

from dataclasses import dataclass

from race_planners.grade import (
    calculate_segment_grades,
    gap_factor,
    parse_gpx,
    smooth_elevation,
    weighted_average_grade,
)
from race_planners.models import (
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


def calculate_plan(loaded_course: LoadedCourse, config: PacingConfig) -> PlanResult:
    trackpoints = loaded_course.trackpoints
    total_distance_km = loaded_course.total_distance_km
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
    for aid_km in loaded_course.course.aid_stops_km:
        if aid_km <= 0 or aid_km > total_distance_km:
            continue
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

    return PlanResult(
        splits=splits,
        segments=_build_segment_summaries(splits),
        aid_arrival_times_min=aid_arrival_times,
        total_time_min=cumulative_time,
        total_distance_km=total_distance_km,
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


def _segment_type(grade_percent: float) -> str:
    if grade_percent >= 2.0:
        return "climb"
    if grade_percent <= -2.0:
        return "descent"
    return "flat"


def _build_segment_summaries(splits: list[PaceSplit]) -> list[SegmentSummary]:
    if not splits:
        return []

    segments: list[SegmentSummary] = []
    current_type = _segment_type(splits[0].grade_percent)
    start_km = 0.0
    distance_km = 0.0
    weighted_grade = 0.0
    weighted_pace = 0.0
    segment_time = 0.0
    prev_end_km = 0.0

    for split in splits:
        end_km = split.km
        split_distance = max(0.0, end_km - prev_end_km)
        split_type = _segment_type(split.grade_percent)

        if split_type != current_type and distance_km > 0:
            segments.append(
                SegmentSummary(
                    segment_type=current_type,
                    start_km=start_km,
                    end_km=prev_end_km,
                    distance_km=distance_km,
                    avg_grade_percent=weighted_grade / distance_km,
                    avg_pace_min_km=weighted_pace / distance_km,
                    segment_time_min=segment_time,
                )
            )
            current_type = split_type
            start_km = prev_end_km
            distance_km = 0.0
            weighted_grade = 0.0
            weighted_pace = 0.0
            segment_time = 0.0

        distance_km += split_distance
        weighted_grade += split.grade_percent * split_distance
        weighted_pace += split.actual_pace_min_km * split_distance
        segment_time += split.segment_time_min
        prev_end_km = end_km

    if distance_km > 0:
        segments.append(
            SegmentSummary(
                segment_type=current_type,
                start_km=start_km,
                end_km=prev_end_km,
                distance_km=distance_km,
                avg_grade_percent=weighted_grade / distance_km,
                avg_pace_min_km=weighted_pace / distance_km,
                segment_time_min=segment_time,
            )
        )

    return segments

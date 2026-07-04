from __future__ import annotations

from dataclasses import dataclass

from race_planners.grade import elevation_changes
from race_planners.models import AidStation, PaceSplit, SegmentSummary, TrackPoint


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


def build_segment_summaries(
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

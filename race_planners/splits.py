from __future__ import annotations

from typing import Any

from race_planners.formatting import format_clock_time, format_duration_minutes, format_pace_minutes
from race_planners.grade import elevation_changes
from race_planners.i18n import t
from race_planners.models import CuratedEvent, PaceSplit, TrackPoint


def course_overview_rows(
    total_distance_km: float, event: CuratedEvent, locale: str = "en"
) -> list[dict[str, str]]:
    aid_mode = (
        t("overview.aid_configured", locale, count=len(event.aid_stops_km))
        if event.aid_stops_km
        else t("overview.aid_derived", locale)
    )
    return [
        {"label": t("overview.distance", locale), "value": f"{total_distance_km:.2f} km"},
        {"label": t("overview.terrain", locale), "value": event.terrain.title()},
        {"label": t("overview.aid_stations", locale), "value": aid_mode},
        {
            "label": t("overview.start_time", locale),
            "value": format_clock_time(event, 0.0)
            if event.start_time_local is not None
            else t("overview.unknown", locale),
        },
    ]


def split_block_options(total_distance_km: float) -> list[int]:
    options = [1, 2, 5, 10]
    return [option for option in options if option < total_distance_km or option == 1]


def default_split_block_size(total_distance_km: float) -> int:
    if total_distance_km > 120:
        return 10
    if total_distance_km > 60:
        return 5
    if total_distance_km > 25:
        return 2
    return 1


def split_piece_rows(splits: list[PaceSplit]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    prev_end_km = 0.0
    for split in splits:
        start_km = prev_end_km
        end_km = split.km
        if end_km <= start_km:
            continue
        rows.append(
            {
                "start_km": start_km,
                "end_km": end_km,
                "start_elapsed": split.cumulative_time_min - split.segment_time_min,
                "end_elapsed": split.cumulative_time_min,
                "segment_time": split.segment_time_min,
                "grade": split.grade_percent,
            }
        )
        prev_end_km = end_km
    return rows


def elapsed_at_distance(split_pieces: list[dict[str, float]], distance_km: float) -> float:
    if distance_km <= 0:
        return 0.0
    for piece in split_pieces:
        if distance_km <= piece["end_km"]:
            piece_distance = piece["end_km"] - piece["start_km"]
            if piece_distance <= 0:
                return piece["end_elapsed"]
            fraction = (distance_km - piece["start_km"]) / piece_distance
            return piece["start_elapsed"] + (piece["segment_time"] * fraction)
    return split_pieces[-1]["end_elapsed"] if split_pieces else 0.0


def aggregate_split_rows(
    splits: list[PaceSplit],
    trackpoints: list[TrackPoint],
    event: CuratedEvent,
    block_size_km: int,
    locale: str = "en",
) -> list[dict[str, Any]]:
    if not splits:
        return []

    pieces = split_piece_rows(splits)
    total_distance_km = splits[-1].km
    rows: list[dict[str, Any]] = []
    block_start_km = 0.0

    while block_start_km < total_distance_km - 0.001:
        block_end_km = min(block_start_km + block_size_km, total_distance_km)
        block_distance_km = block_end_km - block_start_km
        block_time_min = 0.0
        weighted_grade = 0.0

        for piece in pieces:
            overlap_start_km = max(piece["start_km"], block_start_km)
            overlap_end_km = min(piece["end_km"], block_end_km)
            overlap_distance_km = overlap_end_km - overlap_start_km
            piece_distance_km = piece["end_km"] - piece["start_km"]
            if overlap_distance_km <= 0 or piece_distance_km <= 0:
                continue

            overlap_time_min = piece["segment_time"] * (overlap_distance_km / piece_distance_km)
            block_time_min += overlap_time_min
            weighted_grade += piece["grade"] * overlap_distance_km

        start_elapsed_min = elapsed_at_distance(pieces, block_start_km)
        end_elapsed_min = elapsed_at_distance(pieces, block_end_km)
        elev_gain_m, elev_loss_m = elevation_changes(
            trackpoints,
            block_start_km * 1000,
            block_end_km * 1000,
        )
        rows.append(
            {
                t("col.split_range", locale): f"{block_start_km:.1f}-{block_end_km:.1f} km",
                t("col.distance_km_short", locale): round(block_distance_km, 2),
                t("col.pace", locale): format_pace_minutes(block_time_min / block_distance_km),
                t("col.grade", locale): round(weighted_grade / block_distance_km, 2),
                t("col.elev_gain_m", locale): round(elev_gain_m, 1),
                t("col.elev_loss_m", locale): round(elev_loss_m, 1),
                t("col.split_time", locale): format_duration_minutes(block_time_min),
                t("col.elapsed", locale): format_duration_minutes(end_elapsed_min),
                t("col.clock", locale): format_clock_time(event, end_elapsed_min),
                t("col.start_elapsed_short", locale): format_duration_minutes(start_elapsed_min),
            }
        )
        block_start_km = block_end_km

    return rows

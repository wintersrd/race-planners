from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from race_planners.models import TrackPoint


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters."""
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


def parse_gpx(filepath: str) -> list[TrackPoint]:
    """Parse GPX file and return list of TrackPoints."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    trackpoints: list[TrackPoint] = []
    cumulative_distance = 0.0
    prev_point: TrackPoint | None = None

    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat_attr = trkpt.get("lat")
        lon_attr = trkpt.get("lon")
        if lat_attr is None or lon_attr is None:
            continue

        lat = float(lat_attr)
        lon = float(lon_attr)
        ele_elem = trkpt.find("gpx:ele", ns)
        elevation = (
            float(ele_elem.text) if (ele_elem is not None and ele_elem.text is not None) else 0.0
        )
        time_elem = trkpt.find("gpx:time", ns)
        time = time_elem.text if (time_elem is not None and time_elem.text is not None) else ""
        if prev_point is not None:
            distance = haversine(prev_point.lat, prev_point.lon, lat, lon)
            cumulative_distance += distance

        point = TrackPoint(
            lat=lat,
            lon=lon,
            elevation=elevation,
            time=time,
            distance_from_start=cumulative_distance,
        )
        trackpoints.append(point)
        prev_point = point

    return trackpoints


def smooth_elevation(trackpoints: list[TrackPoint], window_size: int = 3) -> list[TrackPoint]:
    if len(trackpoints) < window_size:
        return trackpoints

    smoothed: list[TrackPoint] = []
    half_window = window_size // 2
    for i, point in enumerate(trackpoints):
        start_idx = max(0, i - half_window)
        end_idx = min(len(trackpoints), i + half_window + 1)
        avg_elevation = sum(trackpoints[j].elevation for j in range(start_idx, end_idx)) / (
            end_idx - start_idx
        )
        smoothed.append(
            TrackPoint(
                lat=point.lat,
                lon=point.lon,
                elevation=avg_elevation,
                time=point.time,
                distance_from_start=point.distance_from_start,
            )
        )
    return smoothed


def calculate_segment_grades(
    trackpoints: list[TrackPoint], smoothing_window: int = 5
) -> list[TrackPoint]:
    if len(trackpoints) < 2:
        return trackpoints

    for i in range(1, len(trackpoints)):
        distance_diff = trackpoints[i].distance_from_start - trackpoints[i - 1].distance_from_start
        elevation_diff = trackpoints[i].elevation - trackpoints[i - 1].elevation
        trackpoints[i].grade_percent = (
            (elevation_diff / distance_diff) * 100 if distance_diff > 0 else 0.0
        )

    grades = [p.grade_percent for p in trackpoints]
    for i in range(len(trackpoints)):
        start = max(0, i - smoothing_window // 2)
        end = min(len(grades), i + smoothing_window // 2 + 1)
        trackpoints[i].grade_percent = sum(grades[start:end]) / (end - start)

    return trackpoints


def gap_factor(grade_percent: float) -> float:
    """Current GAP polynomial used by legacy planner."""
    return 0.0021 * grade_percent**2 + 0.034 * grade_percent + 1


def weighted_average_grade(trackpoints: list[TrackPoint], start_m: float, end_m: float) -> float:
    """Return weighted average grade percent between distances."""
    total_distance = 0.0
    weighted_grade = 0.0

    for i in range(1, len(trackpoints)):
        seg_start = trackpoints[i - 1].distance_from_start
        seg_end = trackpoints[i].distance_from_start
        if seg_end <= start_m or seg_start >= end_m:
            continue

        overlap_start = max(seg_start, start_m)
        overlap_end = min(seg_end, end_m)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue

        grade = (trackpoints[i - 1].grade_percent + trackpoints[i].grade_percent) / 2
        weighted_grade += grade * overlap
        total_distance += overlap

    if total_distance == 0:
        return 0.0
    return weighted_grade / total_distance

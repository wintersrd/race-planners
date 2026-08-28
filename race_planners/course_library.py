from __future__ import annotations

from pathlib import Path

from race_planners.event_catalog import CURATED_COURSES_DIR, list_curated_events
from race_planners.grade import extract_aid_stations
from race_planners.models import AidStation, Course


def get_builtin_courses() -> list[Course]:
    """Return curated course definitions shipped with the application."""
    courses: list[Course] = []
    for event in list_curated_events():
        gpx_path = CURATED_COURSES_DIR / event.gpx_relative_path
        aid_stations = _aid_stations_for_event(
            gpx_path, event.aid_stops_km, event.aid_station_tiers
        )
        courses.append(
            Course(
                course_id=event.course_id,
                name=event.name,
                gpx_path=gpx_path,
                aid_stations=aid_stations,
                terrain=event.terrain,
                event_id=event.event_id,
                template_id=event.template_id,
            )
        )
    return courses


def get_course_by_id(course_id: str) -> Course:
    for course in get_builtin_courses():
        if course.course_id == course_id:
            return course
    raise ValueError(f"Unknown course id: {course_id}")


def _aid_stations_for_event(
    gpx_path: Path,
    aid_stop_overrides_km: list[float],
    aid_station_tiers: dict[float, str] | None = None,
) -> list[AidStation]:
    aid_station_tiers = aid_station_tiers or {}
    if aid_stop_overrides_km:
        return [
            AidStation(
                distance_km=distance_km,
                source="config_override",
                tier=_tier_for_distance(distance_km, aid_station_tiers),
            )
            for distance_km in aid_stop_overrides_km
            if distance_km > 0
        ]
    stations = extract_aid_stations(str(gpx_path))
    if aid_station_tiers:
        return [
            AidStation(
                distance_km=station.distance_km,
                label=station.label,
                source=station.source,
                waypoint_type=station.waypoint_type,
                tier=_tier_for_distance(station.distance_km, aid_station_tiers),
            )
            for station in stations
        ]
    return stations


def _tier_for_distance(distance_km: float, tiers: dict[float, str]) -> str:
    for configured_km, tier in tiers.items():
        if abs(distance_km - configured_km) < 0.1:
            return tier
    return "standard"

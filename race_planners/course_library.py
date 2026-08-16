from __future__ import annotations

import re
from pathlib import Path

from race_planners.event_catalog import list_curated_events
from race_planners.grade import extract_aid_stations
from race_planners.models import AidStation, Course


def get_builtin_courses(repo_root: Path) -> list[Course]:
    """Return built-in course definitions shipped in repository."""
    courses: list[Course] = []
    for event in list_curated_events(repo_root):
        gpx_path = repo_root / event.gpx_relative_path
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


def get_course_by_id(repo_root: Path, course_id: str) -> Course:
    courses = list_courses(repo_root)
    for course in courses:
        if course.course_id == course_id:
            return course
    raise ValueError(f"Unknown course id: {course_id}")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "course"


def _course_from_gpx(gpx_path: Path, namespace: str = "local") -> Course:
    stem = gpx_path.stem
    return Course(
        course_id=f"{namespace}:{_slugify(stem)}",
        name=stem.replace("-", " ").replace("_", " ").title(),
        gpx_path=gpx_path,
        aid_stations=extract_aid_stations(str(gpx_path)),
        terrain="mixed",
    )


def list_courses(repo_root: Path) -> list[Course]:
    """List built-in and local library GPX courses from repository."""
    courses = get_builtin_courses(repo_root)
    seen_paths = {course.gpx_path.resolve() for course in courses if course.gpx_path.exists()}

    library_root = repo_root / "courses"
    if library_root.exists():
        for gpx_path in sorted(library_root.glob("**/*.gpx")):
            resolved = gpx_path.resolve()
            if resolved in seen_paths:
                continue
            courses.append(_course_from_gpx(gpx_path, namespace="library"))
            seen_paths.add(resolved)

    return courses


def save_uploaded_gpx(
    upload_name: str,
    upload_bytes: bytes,
    upload_dir: Path,
) -> Course:
    """Persist uploaded GPX to local repository directory and return Course."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    stem = _slugify(Path(upload_name).stem)
    filename = f"{stem}.gpx"
    target = upload_dir / filename
    suffix = 2
    while target.exists():
        target = upload_dir / f"{stem}-{suffix}.gpx"
        suffix += 1

    target.write_bytes(upload_bytes)
    return _course_from_gpx(target, namespace="upload")


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

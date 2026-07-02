from __future__ import annotations

import re
from pathlib import Path

from race_planners.event_catalog import list_curated_events
from race_planners.grade import extract_aid_stops_km
from race_planners.models import Course


def get_builtin_courses(repo_root: Path) -> list[Course]:
    """Return built-in course definitions shipped in repository."""
    courses: list[Course] = []
    for event in list_curated_events(repo_root):
        gpx_path = repo_root / event.gpx_relative_path
        aid_stops_km = list(event.aid_stops_km) or extract_aid_stops_km(str(gpx_path))
        courses.append(
            Course(
                course_id=event.course_id,
                name=event.name,
                gpx_path=gpx_path,
                aid_stops_km=aid_stops_km,
                terrain=event.terrain,
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
        aid_stops_km=extract_aid_stops_km(str(gpx_path)),
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

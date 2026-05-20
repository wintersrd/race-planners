from __future__ import annotations

from pathlib import Path

from race_planners.models import Course


def get_builtin_courses(repo_root: Path) -> list[Course]:
    """Return built-in course definitions shipped in repository."""
    course_dir = repo_root / "semi-marathon-finistere"
    return [
        Course(
            course_id="semi-marathon-finistere",
            name="Semi-Marathon du Finistere",
            gpx_path=course_dir / "semi-marathon-du-finistere.gpx",
            aid_stops_km=[5.3, 9.1, 14.5],
            terrain="road",
        )
    ]


def get_course_by_id(repo_root: Path, course_id: str) -> Course:
    courses = get_builtin_courses(repo_root)
    for course in courses:
        if course.course_id == course_id:
            return course
    raise ValueError(f"Unknown course id: {course_id}")

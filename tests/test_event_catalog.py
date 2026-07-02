from pathlib import Path

from race_planners.course_library import get_builtin_courses
from race_planners.event_catalog import get_curated_event, get_event_template, list_curated_events


def test_list_curated_events_returns_existing_repo_events(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    for filename in [
        "semi-marathon-du-finistere.gpx",
        "2026-grf56.gpx",
        "2026-grf92.gpx",
        "2026-grf166.gpx",
    ]:
        (course_dir / filename).write_text("<gpx></gpx>", encoding="utf-8")

    events = list_curated_events(tmp_path)

    assert [event.event_id for event in events] == [
        "semi-marathon-finistere",
        "grf56",
        "grf92",
        "grf166",
    ]
    assert events[0].template_id == "road_half"
    assert all(event.template_id == "trail_ultra" for event in events[1:])
    assert all(event.race_model == "technical_trail_ultra" for event in events[1:])


def test_get_curated_event_returns_event_metadata(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    event = get_curated_event(tmp_path, "semi-marathon-finistere")
    template = get_event_template(event.template_id)

    assert event.course_id == "semi-marathon-finistere"
    assert event.aid_stops_km == [5.3, 9.1, 14.5]
    assert template.race_model == "half_marathon"


def test_builtin_course_preserves_event_and_template_metadata(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    courses = get_builtin_courses(tmp_path)

    assert len(courses) == 1
    assert courses[0].event_id == "semi-marathon-finistere"
    assert courses[0].template_id == "road_half"

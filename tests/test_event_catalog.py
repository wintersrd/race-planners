from race_planners.course_library import get_builtin_courses
from race_planners.event_catalog import (
    CURATED_COURSES_DIR,
    get_curated_event,
    get_event_template,
    list_curated_events,
)


def test_list_curated_events_resolves_packaged_courses() -> None:
    events = list_curated_events()

    assert [event.event_id for event in events] == [
        "semi-marathon-finistere",
        "grf56",
        "grf92",
        "grf166",
        "marathon-etoiles-baie",
        "trail-odet-ultra",
    ]
    assert all((CURATED_COURSES_DIR / event.gpx_relative_path).is_file() for event in events)
    assert len({event.gpx_relative_path.name for event in events}) == len(events)
    assert events[0].template_id == "road_half"
    assert events[4].template_id == "road_marathon"
    assert events[4].race_model == "road_marathon"
    assert all(event.template_id == "trail_ultra" for event in events[1:4])
    assert all(event.race_model == "technical_trail_ultra" for event in events[1:4])


def test_get_curated_event_returns_event_metadata() -> None:
    event = get_curated_event("semi-marathon-finistere")
    template = get_event_template(event.template_id)

    assert event.course_id == "semi-marathon-finistere"
    assert event.aid_stops_km == [5.3, 9.1, 14.5]
    assert template.race_model == "half_marathon"


def test_get_curated_event_returns_new_marathon_and_ultra_metadata() -> None:
    marathon_event = get_curated_event("marathon-etoiles-baie")
    ultra_event = get_curated_event("trail-odet-ultra")

    assert marathon_event.template_id == "road_marathon"
    assert marathon_event.race_model == "road_marathon"
    assert marathon_event.start_time_local == "09:00"
    assert ultra_event.template_id == "trail_ultra"
    assert ultra_event.race_model == "technical_trail_ultra"
    assert ultra_event.start_time_local == "11:00"


def test_builtin_course_preserves_event_and_template_metadata() -> None:
    course = next(
        course for course in get_builtin_courses() if course.course_id == "semi-marathon-finistere"
    )

    assert course.event_id == "semi-marathon-finistere"
    assert course.template_id == "road_half"

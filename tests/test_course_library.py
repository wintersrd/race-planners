from race_planners.course_library import get_builtin_courses, get_course_by_id
from race_planners.event_catalog import CURATED_COURSES_DIR


def test_get_builtin_courses_uses_curated_event_catalog() -> None:
    courses = get_builtin_courses()

    assert [course.course_id for course in courses] == [
        "semi-marathon-finistere",
        "grf56",
        "grf92",
        "grf166",
        "marathon-etoiles-baie",
        "trail-odet-ultra",
    ]
    assert all(course.gpx_path.parent == CURATED_COURSES_DIR for course in courses)
    assert all(course.gpx_path.exists() for course in courses)

    finistere = get_course_by_id("semi-marathon-finistere")
    grf56 = get_course_by_id("grf56")
    assert finistere.event_id == "semi-marathon-finistere"
    assert finistere.template_id == "road_half"
    assert finistere.aid_stations[0].source == "config_override"
    assert finistere.aid_stations[0].tier == "water_only"
    assert finistere.aid_stations[1].tier == "water_only"
    assert finistere.aid_stations[2].tier == "standard"
    assert len(grf56.aid_stops_km) == 3
    assert all(station.source == "config_override" for station in grf56.aid_stations)
    assert all(station.tier == "water_only" for station in grf56.aid_stations)

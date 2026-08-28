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
        "marathon-des-etoiles-de-la-baie.gpx",
        "trail-de-l-odet-ultra.gpx",
    ]:
        (course_dir / filename).write_text("<gpx></gpx>", encoding="utf-8")

    events = list_curated_events(tmp_path)

    assert [event.event_id for event in events] == [
        "semi-marathon-finistere",
        "grf56",
        "grf92",
        "grf166",
        "marathon-etoiles-baie",
        "trail-odet-ultra",
    ]
    assert events[0].template_id == "road_half"
    assert events[4].template_id == "road_marathon"
    assert events[4].race_model == "road_marathon"
    assert all(event.template_id == "trail_ultra" for event in events[1:4])
    assert all(event.race_model == "technical_trail_ultra" for event in events[1:4])
    assert events[5].template_id == "trail_ultra"
    assert events[5].race_model == "technical_trail_ultra"


def test_get_curated_event_returns_event_metadata(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    event = get_curated_event(tmp_path, "semi-marathon-finistere")
    template = get_event_template(event.template_id)

    assert event.course_id == "semi-marathon-finistere"
    assert event.aid_stops_km == [5.3, 9.1, 14.5]
    assert template.race_model == "half_marathon"


def test_get_curated_event_returns_new_marathon_and_ultra_metadata(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    (course_dir / "marathon-des-etoiles-de-la-baie.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "trail-de-l-odet-ultra.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    marathon_event = get_curated_event(tmp_path, "marathon-etoiles-baie")
    ultra_event = get_curated_event(tmp_path, "trail-odet-ultra")

    assert marathon_event.template_id == "road_marathon"
    assert marathon_event.race_model == "road_marathon"
    assert marathon_event.start_time_local == "09:00"
    assert ultra_event.template_id == "trail_ultra"
    assert ultra_event.race_model == "technical_trail_ultra"
    assert ultra_event.start_time_local == "11:00"


def test_builtin_course_preserves_event_and_template_metadata(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    courses = get_builtin_courses(tmp_path)

    assert len(courses) == 1
    assert courses[0].event_id == "semi-marathon-finistere"
    assert courses[0].template_id == "road_half"


def test_curated_grf_events_declare_shared_time_gates(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    for filename in [
        "2026-grf56.gpx",
        "2026-grf92.gpx",
        "2026-grf166.gpx",
    ]:
        (course_dir / filename).write_text("<gpx></gpx>", encoding="utf-8")

    grf56 = get_curated_event(tmp_path, "grf56")
    grf92 = get_curated_event(tmp_path, "grf92")
    grf166 = get_curated_event(tmp_path, "grf166")

    assert [gate.label for gate in grf56.time_gates] == ["L'aber", "Arrivée Telgruc"]
    assert [gate.label for gate in grf92.time_gates] == [
        "Camaret",
        "Saint-Hernot",
        "L'aber",
        "Arrivée Telgruc",
    ]
    assert [gate.label for gate in grf166.time_gates] == [
        "Trégarvan",
        "Lanvéoc",
        "Camaret",
        "Saint-Hernot",
        "L'aber",
        "Arrivée Telgruc",
    ]
    assert all(gate.distance_km is not None for gate in grf56.time_gates[:-1])
    for event in (grf56, grf92, grf166):
        finish_gates = [gate for gate in event.time_gates if gate.distance_km is None]
        station_gates = [gate for gate in event.time_gates if gate.distance_km is not None]
        assert len(finish_gates) == 1
        assert finish_gates[0].barrier_time_local == "23:59"
        assert all(gate.distance_km is not None and 0 < gate.distance_km for gate in station_gates)


def test_curated_non_grf_events_have_no_time_gates(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "marathon-des-etoiles-de-la-baie.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "trail-de-l-odet-ultra.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    for event_id in ("semi-marathon-finistere", "marathon-etoiles-baie", "trail-odet-ultra"):
        event = get_curated_event(tmp_path, event_id)
        assert event.time_gates == []

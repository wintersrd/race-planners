from pathlib import Path

from race_planners.course_library import get_builtin_courses, list_courses, save_uploaded_gpx


def test_list_courses_includes_gpx_from_local_library(tmp_path: Path) -> None:
    (tmp_path / "semi-marathon-finistere").mkdir()
    (tmp_path / "semi-marathon-finistere" / "semi-marathon-du-finistere.gpx").write_text(
        "<gpx></gpx>", encoding="utf-8"
    )
    library_dir = tmp_path / "courses" / "ultras"
    library_dir.mkdir(parents=True)
    (library_dir / "gr34-test-course.gpx").write_text("<gpx></gpx>", encoding="utf-8")

    courses = list_courses(tmp_path)

    assert any(c.course_id == "semi-marathon-finistere" for c in courses)
    assert any(c.course_id == "library:gr34-test-course" for c in courses)


def test_get_builtin_courses_uses_curated_event_catalog(tmp_path: Path) -> None:
    course_dir = tmp_path / "semi-marathon-finistere"
    course_dir.mkdir()
    (course_dir / "semi-marathon-du-finistere.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "2026-grf92.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "2026-grf166.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "marathon-des-etoiles-de-la-baie.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "trail-de-l-odet-ultra.gpx").write_text("<gpx></gpx>", encoding="utf-8")
    (course_dir / "2026-grf56.gpx").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="48.0" lon="-3.99"><type>ravitoliquide</type></wpt>
  <trk>
    <trkseg>
      <trkpt lat="48.0" lon="-4.0"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.99"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.98"><ele>5</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
""",
        encoding="utf-8",
    )

    courses = get_builtin_courses(tmp_path)

    assert [course.course_id for course in courses] == [
        "semi-marathon-finistere",
        "grf56",
        "grf92",
        "grf166",
        "marathon-etoiles-baie",
        "trail-odet-ultra",
    ]
    finistere = next(course for course in courses if course.course_id == "semi-marathon-finistere")
    grf56 = next(course for course in courses if course.course_id == "grf56")
    assert finistere.event_id == "semi-marathon-finistere"
    assert finistere.template_id == "road_half"
    assert finistere.aid_stations[0].source == "config_override"
    assert len(grf56.aid_stops_km) == 1
    assert grf56.aid_stations[0].source == "gpx_waypoint"


def test_save_uploaded_gpx_avoids_collisions(tmp_path: Path) -> None:
    upload_dir = tmp_path / "courses" / "uploads"

    first = save_uploaded_gpx("my route.gpx", b"<gpx></gpx>", upload_dir)
    second = save_uploaded_gpx("my route.gpx", b"<gpx></gpx>", upload_dir)

    assert first.gpx_path.exists()
    assert second.gpx_path.exists()
    assert first.gpx_path != second.gpx_path
    assert first.course_id.startswith("upload:")

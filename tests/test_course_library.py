from pathlib import Path

from race_planners.course_library import list_courses, save_uploaded_gpx


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


def test_save_uploaded_gpx_avoids_collisions(tmp_path: Path) -> None:
    upload_dir = tmp_path / "courses" / "uploads"

    first = save_uploaded_gpx("my route.gpx", b"<gpx></gpx>", upload_dir)
    second = save_uploaded_gpx("my route.gpx", b"<gpx></gpx>", upload_dir)

    assert first.gpx_path.exists()
    assert second.gpx_path.exists()
    assert first.gpx_path != second.gpx_path
    assert first.course_id.startswith("upload:")

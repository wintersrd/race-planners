from pathlib import Path

import pytest

from race_planners.course_library import get_course_by_id
from race_planners.formatting import format_signed_duration_minutes
from race_planners.i18n import t
from race_planners.models import (
    AidStation,
    Course,
    LoadedCourse,
    PacingConfig,
    TimeGate,
    TrackPoint,
)
from race_planners.planner import calculate_plan, load_course_trackpoints
from race_planners.time_gates import (
    TIGHT_GATE_BUFFER_MIN,
    attach_gates_to_stations,
    evaluate_time_gates,
    gate_deadline_elapsed_min,
    gate_warning_codes,
)


def test_gate_deadline_same_day() -> None:
    assert gate_deadline_elapsed_min("09:00", "13:00") == 240.0


def test_gate_deadline_rolls_over_midnight() -> None:
    assert gate_deadline_elapsed_min("17:00", "07:00") == 840.0


def test_gate_deadline_at_start_time_is_next_day() -> None:
    assert gate_deadline_elapsed_min("17:00", "17:00") == 1440.0


def test_gate_deadline_rejects_invalid_clock_time() -> None:
    with pytest.raises(ValueError, match="Invalid clock time"):
        gate_deadline_elapsed_min("17:00", "7am")
    with pytest.raises(ValueError, match="out of range"):
        gate_deadline_elapsed_min("17:00", "24:30")


def _stations(*distances_km: float) -> list[AidStation]:
    return [AidStation(distance_km=km, label=f"km{km}") for km in distances_km]


def test_attach_gates_matches_nearest_station_within_tolerance() -> None:
    stations = _stations(10.2, 40.0)
    gate = TimeGate(label="Gate A", barrier_time_local="23:30", distance_km=10.5)

    attached, unmatched = attach_gates_to_stations(stations, [gate])

    assert not unmatched
    assert attached[0].time_gate is gate
    assert attached[1].time_gate is None


def test_attach_gates_allows_one_gate_per_station() -> None:
    stations = _stations(10.2)
    first = TimeGate(label="First", barrier_time_local="23:30", distance_km=10.0)
    second = TimeGate(label="Second", barrier_time_local="07:00", distance_km=10.5)

    attached, unmatched = attach_gates_to_stations(stations, [first, second])

    assert attached[0].time_gate is first
    assert unmatched == [second]


def test_attach_gates_reports_unmatched_beyond_tolerance() -> None:
    stations = _stations(10.0)
    gate = TimeGate(label="Far", barrier_time_local="13:00", distance_km=15.0)

    attached, unmatched = attach_gates_to_stations(stations, [gate])

    assert attached[0].time_gate is None
    assert unmatched == [gate]


def test_attach_gates_ignores_finish_gates() -> None:
    stations = _stations(10.0)
    finish = TimeGate(label="Finish", barrier_time_local="23:59")

    attached, unmatched = attach_gates_to_stations(stations, [finish])

    assert attached[0].time_gate is None
    assert not unmatched


def test_evaluate_time_gates_forces_monotonic_multi_day_deadlines() -> None:
    gates = [
        TimeGate(label="Trégarvan", barrier_time_local="23:30", distance_km=38),
        TimeGate(label="Lanvéoc", barrier_time_local="07:00", distance_km=74),
        TimeGate(label="Camaret", barrier_time_local="13:00", distance_km=112),
        TimeGate(label="Saint-Hernot", barrier_time_local="20:00", distance_km=146.9),
        TimeGate(label="L'aber", barrier_time_local="22:15", distance_km=159.5),
        TimeGate(label="Arrivée Telgruc", barrier_time_local="23:59"),
    ]

    checks = evaluate_time_gates([(gate, 100.0) for gate in gates], "17:00")

    assert [check.deadline_elapsed_min for check in checks] == [
        390.0,
        840.0,
        1200.0,
        1620.0,
        1755.0,
        1859.0,
    ]
    assert all(not check.missed for check in checks)
    assert checks[-1].distance_km is None


def test_evaluate_time_gates_flags_missed_and_buffer_sign() -> None:
    early = TimeGate(label="Early", barrier_time_local="09:00", distance_km=10)
    late = TimeGate(label="Late", barrier_time_local="10:00", distance_km=20)

    checks = evaluate_time_gates([(early, 40.0), (late, 150.0)], "08:00")

    assert checks[0].buffer_min == 20.0
    assert not checks[0].missed
    assert checks[1].buffer_min == -30.0
    assert checks[1].missed


def test_gate_warning_codes_distinguish_missed_and_tight() -> None:
    checks = evaluate_time_gates(
        [
            (TimeGate(label="Missed", barrier_time_local="09:00", distance_km=10), 90.0),
            (TimeGate(label="Tight", barrier_time_local="10:00", distance_km=20), 105.0),
            (TimeGate(label="Fine", barrier_time_local="11:00", distance_km=30), 150.0),
        ],
        "08:00",
    )

    codes = gate_warning_codes(checks)

    assert codes == [
        "warning.gate_missed|label=Missed,buffer=-30",
        "warning.gate_tight|label=Tight,buffer=15",
    ]


def test_tight_buffer_boundary_is_exclusive() -> None:
    checks = evaluate_time_gates(
        [(TimeGate(label="Edge", barrier_time_local="09:00", distance_km=10), 30.0)],
        "08:00",
    )

    assert checks[0].buffer_min == TIGHT_GATE_BUFFER_MIN
    assert gate_warning_codes(checks) == []


def test_format_signed_duration_minutes() -> None:
    assert format_signed_duration_minutes(45.0) == "+45:00"
    assert format_signed_duration_minutes(-72.0) == "−1:12:00"
    assert format_signed_duration_minutes(None) == "-"


def test_time_gate_i18n_keys_render_in_both_locales() -> None:
    for locale in ("en", "fr"):
        assert t("section.time_gates", locale)
        assert t("col.barrier", locale)
        assert t("col.buffer", locale)
        rendered = t("warning.gate_missed", locale, label="Trégarvan", buffer=-45)
        assert "Trégarvan" in rendered
        assert "-45" in rendered


def _grf166_loaded() -> LoadedCourse:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf166")
    return load_course_trackpoints(course)


def _grf166_config(target_finish_time_min: float, start: str | None) -> PacingConfig:
    return PacingConfig(
        race_model="technical_trail_ultra",
        input_mode="finish_time",
        target_finish_time_min=target_finish_time_min,
        climb_hike_threshold_percent=12.0,
        descent_caution="medium",
        event_start_time_local=start,
    )


def test_grf166_fast_plan_passes_all_gates() -> None:
    result = calculate_plan(_grf166_loaded(), _grf166_config(1300.0, "17:00"))

    assert result.gate_checks
    assert not any(check.missed for check in result.gate_checks)
    assert result.gate_checks[-1].distance_km is None
    assert not any(code.startswith("warning.gate") for code in result.warnings)
    assert "assumption.time_gates" in result.assumptions
    gated = [eta for eta in result.aid_station_etas if eta.barrier_time_local]
    assert {eta.barrier_time_local for eta in gated} == {
        "23:30",
        "07:00",
        "13:00",
        "20:00",
        "22:15",
    }
    assert all(eta.arrival_buffer_min is not None for eta in gated)


def test_grf166_station_buffers_match_gate_checks() -> None:
    result = calculate_plan(_grf166_loaded(), _grf166_config(1300.0, "17:00"))

    for eta in result.aid_station_etas:
        if eta.barrier_time_local is None:
            assert eta.arrival_buffer_min is None
            continue
        nearest = min(
            result.gate_checks,
            key=lambda check: abs((check.distance_km or 0.0) - eta.distance_km),
        )
        assert eta.arrival_buffer_min == pytest.approx(nearest.buffer_min)


def test_grf166_slow_plan_misses_gates_with_warnings() -> None:
    result = calculate_plan(_grf166_loaded(), _grf166_config(2400.0, "17:00"))

    assert all(check.missed for check in result.gate_checks)
    assert all(check.buffer_min < 0 for check in result.gate_checks)
    missed_codes = [code for code in result.warnings if code.startswith("warning.gate_missed")]
    assert len(missed_codes) == len(result.gate_checks)
    assert any("label=Trégarvan" in code for code in missed_codes)


def test_grf166_finish_gate_uses_total_time() -> None:
    result = calculate_plan(_grf166_loaded(), _grf166_config(1300.0, "17:00"))

    finish_check = result.gate_checks[-1]
    assert finish_check.distance_km is None
    assert finish_check.arrival_elapsed_min == pytest.approx(result.total_time_min)
    assert finish_check.deadline_elapsed_min == 1859.0


def test_grf166_without_start_time_warns_and_skips_checks() -> None:
    result = calculate_plan(_grf166_loaded(), _grf166_config(1300.0, None))

    assert result.gate_checks == []
    assert "warning.gate_start_time_missing" in result.warnings
    assert "assumption.time_gates" not in result.assumptions
    assert all(eta.arrival_buffer_min is None for eta in result.aid_station_etas)


def test_unmatched_station_gate_evaluated_in_distance_order() -> None:
    course = Course(
        course_id="synthetic",
        name="Synthetic",
        gpx_path=Path("synthetic.gpx"),
        aid_stops_km=[10.0, 30.0],
        time_gates=[
            TimeGate(label="Mid", barrier_time_local="11:50", distance_km=25.0),
            TimeGate(label="Finish", barrier_time_local="14:00"),
        ],
    )
    trackpoints = [
        TrackPoint(
            lat=48.0,
            lon=-4.0 + index * 0.0001,
            elevation=10.0,
            time="2026-01-01T00:00:00",
            distance_from_start=index * 100.0,
            grade_percent=0.0,
        )
        for index in range(501)
    ]
    loaded = LoadedCourse(course=course, trackpoints=trackpoints, total_distance_km=50.0)
    config = PacingConfig(
        race_model="half_marathon",
        input_mode="finish_time",
        target_finish_time_min=300.0,
        event_start_time_local="09:00",
    )

    result = calculate_plan(loaded, config)

    assert [check.label for check in result.gate_checks] == ["Mid", "Finish"]
    mid = result.gate_checks[0]
    assert mid.deadline_elapsed_min == 170.0
    assert mid.buffer_min == pytest.approx(20.0, abs=5.0)
    assert not mid.missed
    assert result.gate_checks[-1].distance_km is None
    assert all(eta.barrier_time_local is None for eta in result.aid_station_etas)
    assert any(code.startswith("warning.gate_tight|label=Mid") for code in result.warnings)

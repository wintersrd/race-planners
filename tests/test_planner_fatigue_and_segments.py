from race_planners.fatigue import fatigue_multiplier
from race_planners.segments import build_segment_summaries
from race_planners.models import AidStation, PaceSplit


def testfatigue_multiplier_progresses_by_model() -> None:
    assert fatigue_multiplier("road_marathon", 0.5) == 1.0
    assert fatigue_multiplier("road_marathon", 1.0) > 1.0
    assert fatigue_multiplier("technical_trail_ultra", 1.0) > fatigue_multiplier(
        "road_marathon", 1.0
    )


def test_segment_summary_groups_adjacent_split_types() -> None:
    splits = [
        PaceSplit(1.0, 6.0, 3.1, 6.0, 6.0),
        PaceSplit(2.0, 6.2, 2.8, 6.2, 12.2),
        PaceSplit(3.0, 5.5, 0.1, 5.5, 17.7),
        PaceSplit(4.0, 5.2, -3.3, 5.2, 22.9),
    ]

    segments = build_segment_summaries(splits)

    assert [s.segment_type for s in segments] == ["climb", "flat", "descent"]
    assert segments[0].distance_km == 2.0
    assert segments[-1].end_km == 4.0
    assert segments[0].block_label == "Start to Finish"


def test_segment_summary_respects_aid_station_boundaries() -> None:
    splits = [
        PaceSplit(1.0, 6.0, 3.0, 6.0, 6.0),
        PaceSplit(2.0, 6.0, 3.2, 6.0, 12.0),
        PaceSplit(3.0, 5.0, -3.0, 5.0, 17.0),
    ]

    segments = build_segment_summaries(
        splits,
        aid_stations=[AidStation(distance_km=1.5, label="Aid 1")],
        total_distance_km=3.0,
    )

    assert [(segment.start_km, segment.end_km) for segment in segments] == [
        (0.0, 1.5),
        (1.5, 2.0),
        (2.0, 3.0),
    ]
    assert [segment.block_label for segment in segments] == [
        "Start to Aid 1",
        "Aid 1 to Finish",
        "Aid 1 to Finish",
    ]
    assert segments[0].section_name == "Start to Aid 1: climb"

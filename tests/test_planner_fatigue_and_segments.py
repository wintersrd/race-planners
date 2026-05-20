from race_planners.planner import _build_segment_summaries, _fatigue_multiplier
from race_planners.models import PaceSplit


def test_fatigue_multiplier_progresses_by_model() -> None:
    assert _fatigue_multiplier("road_marathon", 0.5) == 1.0
    assert _fatigue_multiplier("road_marathon", 1.0) > 1.0
    assert _fatigue_multiplier("technical_trail_ultra", 1.0) > _fatigue_multiplier(
        "road_marathon", 1.0
    )


def test_segment_summary_groups_adjacent_split_types() -> None:
    splits = [
        PaceSplit(1.0, 6.0, 3.1, 6.0, 6.0),
        PaceSplit(2.0, 6.2, 2.8, 6.2, 12.2),
        PaceSplit(3.0, 5.5, 0.1, 5.5, 17.7),
        PaceSplit(4.0, 5.2, -3.3, 5.2, 22.9),
    ]

    segments = _build_segment_summaries(splits)

    assert [s.segment_type for s in segments] == ["climb", "flat", "descent"]
    assert segments[0].distance_km == 2.0
    assert segments[-1].end_km == 4.0

from race_planners.pacing import (
    FireRoadUltraModel,
    PacingContext,
    technical_descent_multiplier,
    technical_trail_pace,
)


def test_fire_road_bias_shifts_toward_z1_late_race() -> None:
    model = FireRoadUltraModel(
        z1_pace_min_km=7.0,
        z2_pace_min_km=6.0,
        hike_pace_min_km=11.0,
        hike_threshold_percent=12.0,
    )
    early_context = PacingContext(
        grade_percent=0.0,
        progress_ratio=0.1,
        elapsed_hours=1.0,
        climb_m_per_km=8.0,
    )
    late_context = PacingContext(
        grade_percent=0.0,
        progress_ratio=0.9,
        elapsed_hours=10.0,
        climb_m_per_km=30.0,
    )

    early_pace = model.pace_for_context(early_context)
    late_pace = model.pace_for_context(late_context)

    assert late_pace > early_pace


def test_technical_descent_caution_penalizes_steeper_grades() -> None:
    mild = technical_descent_multiplier(-5.0, "medium")
    steep = technical_descent_multiplier(-20.0, "medium")
    assert steep > mild


def test_technical_very_steep_can_be_slower_than_flat() -> None:
    pace = technical_trail_pace(
        flat_pace_min_km=8.0,
        hike_pace_min_km=13.0,
        grade_percent=-20.0,
        climb_hike_threshold_percent=12.0,
        descent_caution="high",
    )
    assert pace > 8.0

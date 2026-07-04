from pathlib import Path

from race_planners.course_library import get_course_by_id
from race_planners.fueling import (
    build_fueling_plan,
    estimate_event_kcal,
    segment_kcal,
)
from race_planners.models import PacingConfig
from race_planners.planner import calculate_plan, load_course_trackpoints


def test_segment_kcal_scales_with_mass_distance_and_grade() -> None:
    flat = segment_kcal(70.0, 1.0, 0.0)
    uphill = segment_kcal(70.0, 1.0, 5.0)
    downhill = segment_kcal(70.0, 1.0, -5.0)

    assert flat == 70.0
    assert uphill > flat
    assert downhill > flat
    assert uphill > downhill


def test_estimate_event_kcal_scales_with_distance() -> None:
    from race_planners.models import PaceSplit

    half_splits = [
        PaceSplit(
            km=i,
            actual_pace_min_km=5.0,
            grade_percent=0.0,
            segment_time_min=5.0,
            cumulative_time_min=i * 5.0,
        )
        for i in range(1, 22)
    ]
    full_splits = [
        PaceSplit(
            km=i,
            actual_pace_min_km=5.0,
            grade_percent=0.0,
            segment_time_min=5.0,
            cumulative_time_min=i * 5.0,
        )
        for i in range(1, 43)
    ]

    half_kcal = estimate_event_kcal(70.0, half_splits)
    full_kcal = estimate_event_kcal(70.0, full_splits)

    assert half_kcal > 1400
    assert full_kcal > half_kcal * 1.9


def test_fueling_plan_scales_carb_target_with_duration() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)
    config = PacingConfig(
        race_model="half_marathon",
        input_mode="finish_time",
        target_finish_time_min=100.0,
    )
    plan_result = calculate_plan(loaded, config)

    fueling = build_fueling_plan(
        plan_result,
        mass_kg=70.0,
        peak_temperature_c=16.0,
        aid_station_tiers=["water_only", "water_only", "standard"],
    )

    assert fueling.total_kcal > 1000
    assert fueling.total_carb_target_g > 0
    assert len(fueling.blocks) > 0
    assert all(block.carb_target_g >= 0 for block in fueling.blocks)


def test_fueling_plan_produces_larger_carb_target_for_ultra() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    half_course = get_course_by_id(repo_root, "semi-marathon-finistere")
    half_loaded = load_course_trackpoints(half_course)
    half_result = calculate_plan(
        half_loaded,
        PacingConfig(
            race_model="half_marathon",
            input_mode="finish_time",
            target_finish_time_min=100.0,
        ),
    )

    ultra_course = get_course_by_id(repo_root, "grf92")
    ultra_loaded = load_course_trackpoints(ultra_course)
    ultra_result = calculate_plan(
        ultra_loaded,
        PacingConfig(
            race_model="technical_trail_ultra",
            input_mode="effort_anchor",
            flat_pace_min_km=8.5,
            hike_pace_min_km=13.0,
        ),
    )

    half_fueling = build_fueling_plan(half_result, 70.0, 16.0, [])
    ultra_fueling = build_fueling_plan(ultra_result, 70.0, 20.0, [])

    assert ultra_fueling.total_kcal > half_fueling.total_kcal * 3
    assert ultra_fueling.total_fluid_target_l > half_fueling.total_fluid_target_l * 3


def test_aid_station_tier_affects_on_site_fuel() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)
    config = PacingConfig(
        race_model="half_marathon",
        input_mode="finish_time",
        target_finish_time_min=100.0,
    )
    plan_result = calculate_plan(loaded, config)

    water_only = build_fueling_plan(
        plan_result,
        70.0,
        16.0,
        ["water_only", "water_only", "water_only"],
    )
    full_service = build_fueling_plan(
        plan_result,
        70.0,
        16.0,
        ["full_service", "full_service", "full_service"],
    )

    assert sum(block.on_site_carb_g for block in full_service.blocks) > sum(
        block.on_site_carb_g for block in water_only.blocks
    )


def test_fueling_plan_warns_on_large_carb_deficit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "grf92")
    loaded = load_course_trackpoints(course)
    config = PacingConfig(
        race_model="technical_trail_ultra",
        input_mode="effort_anchor",
        flat_pace_min_km=8.5,
        hike_pace_min_km=13.0,
    )
    plan_result = calculate_plan(loaded, config)

    fueling = build_fueling_plan(plan_result, 70.0, 20.0, [])

    assert len(fueling.warnings) > 0
    assert any("gut stress" in w.lower() for w in fueling.warnings)


def test_carry_items_use_realistic_gel_sizes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)
    config = PacingConfig(
        race_model="half_marathon",
        input_mode="finish_time",
        target_finish_time_min=92.0,
    )
    plan_result = calculate_plan(loaded, config)

    fueling = build_fueling_plan(plan_result, 70.0, 16.0, ["water_only", "water_only", "standard"])

    for block in fueling.blocks:
        for item in block.carry_items:
            assert "30g gel" in item or "50g gel" in item
            assert "23g" not in item


def test_fueling_window_skips_startup_and_tail_blocks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    course = get_course_by_id(repo_root, "semi-marathon-finistere")
    loaded = load_course_trackpoints(course)
    config = PacingConfig(
        race_model="half_marathon",
        input_mode="finish_time",
        target_finish_time_min=92.0,
    )
    plan_result = calculate_plan(loaded, config)

    fueling = build_fueling_plan(plan_result, 70.0, 16.0, ["water_only", "water_only", "standard"])

    assert len(fueling.blocks) >= 4
    first_block = fueling.blocks[0]
    last_block = fueling.blocks[-1]
    assert first_block.carb_target_g == 0.0
    assert first_block.carry_items == []
    assert last_block.carb_target_g == 0.0
    assert last_block.carry_items == []

    mid_blocks = fueling.blocks[1:-1]
    assert any(b.carb_target_g > 0 for b in mid_blocks)

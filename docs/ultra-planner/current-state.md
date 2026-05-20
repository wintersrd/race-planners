# General Planner Current State

Date: 2026-05-20

## Delivered components

- Core package scaffold under `race_planners/`
- GPX parsing and grade helpers in `race_planners/grade.py`
- Pacing models in `race_planners/pacing.py`
- Reusable planner engine in `race_planners/planner.py`
  - Includes progressive fatigue multipliers by race model
  - Emits both kilometer splits and grouped segment pacing summaries
- Local course library + upload persistence in `race_planners/course_library.py`
- JSON plan I/O in `race_planners/plan_io.py`
- Beta Streamlit flow in `race_planners/streamlit_general.py`
- Legacy app mode switch in `semi-marathon-finistere/app.py`

## Test coverage summary

- `tests/test_pacing_models.py`: model behavior and descent/hike logic
- `tests/test_plan_io.py`: JSON roundtrip and missing-GPX error
- `tests/test_planner.py`: planner happy paths + input validation
- `tests/test_planner_edge_cases.py`: aid-stop edge cases
- `tests/test_course_library.py`: course discovery and upload collision handling
- `tests/test_streamlit_general.py`: plan-load state restoration and error propagation
- `tests/test_legacy_compatibility.py`: new planner parity check against legacy GAP pacing baseline
- `tests/test_python_compile.py`: Streamlit app compile guard

## Known limitations

- General planner UI is still beta-level and intentionally simpler than legacy tabbed output.
- Scenario comparison and calorie estimation remain deferred by scope decision.
- Technical trail and marathon fatigue tuning can be tightened with empirical race data.

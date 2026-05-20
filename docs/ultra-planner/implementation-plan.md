# General Race Planner Implementation Plan

Date: 2026-05-20
Status: Approved for implementation

## Objective

Evolve the current single-race Streamlit pacing app into a general race planner that supports:

- Multiple race models (half marathon, road marathon, fire-road ultra, technical-trail ultra)
- Multiple course sources (built-in library + uploaded GPX)
- Multiple input modes (finish-time and effort-anchor)
- Durable plan persistence via JSON export/import

## Scope

### In scope

- Course abstraction and local course library in repository
- GPX upload flow for ad hoc routes
- Pluggable pacing model interface and four initial models
- Aid station arrival timing and segment pacing outputs
- JSON save/reload with versioned schema and missing-GPX handling
- Progressive fatigue behavior for marathon/ultra models

### Out of scope for initial delivery

- Nutrition/hydration planning
- Full scenario comparison UI
- Shared cloud storage of plans or routes

## Architecture

### Core domain objects

- `Course`: id, name, source, GPX reference, route profile, aid stations, optional sections
- `PlanInput`: course ref, race model, input mode, assumptions, pacing controls
- `PlanResult`: splits, aid ETAs, segment pacing, totals, warnings
- `PacingModel`: model interface used by all race calculators

### Modules

- `race_planners/gpx/`: parsing, smoothing, grade calculations
- `race_planners/segments/`: rolling grade + segment detection
- `race_planners/pacing/`: model interface and model implementations
- `race_planners/persistence/`: JSON schema + load/save behavior
- `race_planners/courses/`: local built-in course metadata and loaders

## Model behavior

### Half marathon model

- Preserve existing behavior as baseline compatibility model

### Road marathon model

- Primary effort input: known marathon pace
- Optional guardrails: RPE/HR
- Progressive fatigue curve tuned for marathon duration

### Fire-road ultra model

- Explicit anchors: Z1 pace, Z2 pace, hike pace
- Uses time + distance + cumulative climb to bias pacing from Z2 toward Z1
- Run/hike transitions on climb threshold
- Moderate descent restraint

### Technical-trail ultra model

- Inputs: flat pace, hike pace, climb threshold, descent caution preset
- Descent caution applies grade-based reduction to GAP benefit
- Very steep descents can become slower than flat pace
- Stronger fatigue than fire-road model

## Implementation phases

### Phase 1: Extract and stabilize reusable logic

- Move GPX, GAP, and segment utilities out of monolithic app file
- Keep current output behavior stable

### Phase 2: Add domain and configuration abstraction

- Introduce Course/PlanInput/PlanResult structures
- Remove hardcoded single-race assumptions

### Phase 3: Multi-course support

- Add local course library loader
- Add GPX upload route path

### Phase 4: Pluggable model support

- Add model interface
- Implement half + marathon + fire-road ultra + technical-trail ultra

### Phase 5: Persistence

- Add JSON export/import with schema versioning
- Add clear missing-course/GPX error messaging

### Phase 6: UI adaptation

- Course selector
- Model selector
- Input mode toggle
- Model-specific configuration panels

### Phase 7: Optional enhancements

- Scenario comparison
- Rough calorie estimate

## Testing strategy

- Unit: grade, GAP, fatigue, run/hike thresholds, descent caution, serialization
- Integration: library course load, GPX upload, JSON save/load, missing-GPX error
- Regression: current half-marathon behavior preserved within tolerance
- Quality gates:
  - `uv run ruff check .`
  - `uv run ruff format .`
  - `uv run mypy tests`
  - `uv run pytest`
  - `uv run pre-commit run --all-files`

## Definition of done

- User can plan with built-in or uploaded route data
- User can select all four race models
- User can use finish-time or effort-anchor mode
- Aid ETAs + segment pacing are produced for each model
- JSON plan round-trip works with deterministic behavior
- Missing uploaded GPX is handled with explicit recovery guidance
- Tests and pre-commit pass

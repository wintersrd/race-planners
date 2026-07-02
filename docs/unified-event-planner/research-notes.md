# Unified Event Planner Research Notes

Date: 2026-07-02
Status: Discovery complete, implementation not yet started

## Why this note exists

This note records the factual state of the repository and the requirements
interview that led to the current unified-planner direction. It is written as a
handoff artifact for future implementation work.

## Current app split

The repository currently contains two planner experiences mounted inside the
same Streamlit app.

### Legacy planner path

- Entry point: `semi-marathon-finistere/app.py`
- UI mode key: `legacy_half_marathon`
- Characteristics:
  - polished single-race UX
  - stronger visuals and report sections
  - bilingual translation support
  - half-marathon-specific assumptions and content

### General planner path

- UI module: `race_planners/streamlit_general.py`
- UI mode key: `general_beta`
- Characteristics:
  - reusable planner engine
  - multiple pacing models
  - JSON plan import/export
  - course library abstraction
  - rougher UI and weaker product fit than legacy

### Consequence

The codebase does not yet have one planner with multiple events. It has two
different planner applications sharing only part of the underlying logic.

## Relevant code locations

### Legacy app and inline logic

- `semi-marathon-finistere/app.py`
  - Streamlit shell
  - legacy GPX loading
  - legacy pacing calculations
  - legacy elevation segmentation
  - legacy split tables, charts, pacing tips, rest-stop reporting

### Reusable package modules

- `race_planners/models.py`
- `race_planners/grade.py`
- `race_planners/pacing.py`
- `race_planners/planner.py`
- `race_planners/course_library.py`
- `race_planners/plan_io.py`
- `race_planners/streamlit_general.py`

### Existing tests that matter

- `tests/test_legacy_compatibility.py`
- `tests/test_streamlit_general.py`
- `tests/test_course_library.py`
- `tests/test_planner.py`
- `tests/test_planner_edge_cases.py`
- `tests/test_planner_fatigue_and_segments.py`
- `tests/test_pacing_models.py`

## Legacy planner strengths worth preserving

These were identified both from the code and from the requirements interview.

- stronger overall UX shell
- bilingual translation layer via `TRANSLATIONS` and `t()`
- cleaner course overview presentation
- better elevation profile presentation
- richer course sections and split analysis
- pacing tips and race-specific coaching framing
- rest-stop timing outputs
- stronger visual/tabbed presentation
- more event-like experience rather than a generic calculation form

The user explicitly said printable outputs are not required as a first-class
concern for the merged v1, although they may be derived later.

## General planner strengths worth preserving

- reusable domain package instead of one monolithic app file
- multiple race models already exist
- both `finish_time` and `effort_anchor` modes are already supported
- plan save/load exists via JSON
- course library abstraction exists
- tests already cover much of the reusable engine behavior

## Structural issues discovered

### 1. The planners are separate products, not one system

The current radio toggle hides a real product split:

- one branch renders the older single-race UI
- the other branch renders the beta general planner UI

This means feature parity cannot be achieved by a small UI patch.

### 2. Domain logic is duplicated

Examples:

- GPX parsing exists inline in `semi-marathon-finistere/app.py`
- GPX parsing also exists in `race_planners/grade.py`
- the same GAP concept exists in both code paths
- segment concepts exist in both, but not in the same representation

### 3. Data shapes differ across the two systems

Legacy code returns dict-heavy structures and bespoke elevation segment data.

The package planner returns typed dataclasses such as:

- `TrackPoint`
- `Course`
- `PacingConfig`
- `PaceSplit`
- `SegmentSummary`
- `PlanResult`

This mismatch is a direct obstacle to sharing richer UI components.

### 4. Some inputs exist in config but are not truly wired through

Example:

- `rest_duration_sec` exists in `PacingConfig`
- `calculate_plan()` does not currently consume it in a meaningful way

Legacy rest-stop handling is therefore not actually present in the canonical
planner engine.

### 5. Course metadata is too thin

Current `Course` fields are effectively:

- `course_id`
- `name`
- `gpx_path`
- `aid_stops_km`
- `terrain`

That is not enough for a curated event-first product. Missing examples include:

- display metadata
- model selection
- template inheritance
- localized names
- default pacing inputs
- event status metadata
- richer aid-station metadata
- event-level notes and warnings

## Current course-library behavior

### Built-in courses

Only one built-in course is hardcoded today:

- `semi-marathon-finistere`

It points at the Finistere half-marathon GPX and explicit aid-stop distances.

### Auto-discovered courses

`list_courses()` scans GPX files under `courses/**.gpx` and combines them with
the built-in course list.

### Important limitation

GPX files placed directly under `semi-marathon-finistere/` are not automatically
discoverable by the general planner, except for the one hardcoded built-in
entry.

### Uploaded GPX behavior

The general planner currently supports uploaded GPX files stored under
`courses/uploads/`.

This matters because the new merged direction explicitly excludes custom GPX
upload for the curated event-first experience. Existing upload code is therefore
legacy-to-beta infrastructure, not a required product feature for the next
merged milestone.

## GPX findings for GRF events

The following GPX files were added under `semi-marathon-finistere/`:

- `2026-grf56.gpx`
- `2026-grf92.gpx`
- `2026-grf166.gpx`

### `2026-grf56.gpx`

- metadata name indicates `Grand Raid du Finistere 2026 - GRF56`
- contains `<trkpt>` track geometry
- contains `<wpt>` waypoints including start, finish, and aid-like points such
  as `ravitoliquide`

### `2026-grf166.gpx`

- metadata name indicates `Grand Raid du Finistere 2026 - GRF166`
- contains `<trkpt>` track geometry
- contains many `<wpt>` waypoints including aid-like points and start/finish

### `2026-grf92.gpx`

- metadata name indicates `GRF92`
- contains waypoints such as begin/end
- route geometry is encoded as `<rtept>` inside `<rte>`, not `<trkpt>`

### Immediate parser implication

Both existing GPX parsers currently search for `.//gpx:trkpt` only.

That means `GRF92` will currently fail to produce usable course geometry unless
the parser is upgraded to support route-point-based GPX files.

### Aid metadata implication

Current library import behavior throws away waypoint-derived aid metadata for
auto-discovered GPX files by creating courses with empty `aid_stops_km`.

This conflicts with the user requirement that aid points should come from GPX
when available, with config only acting as override or fallback.

## Test and compatibility findings

### Legacy compatibility already has a foothold

`tests/test_legacy_compatibility.py` validates that the new planner's
half-marathon GAP behavior stays close to the legacy logic for the same base GAP
pace.

This is important because it gives a safe bridge: the legacy half-marathon logic
does not need to remain isolated forever if parity is maintained in tests.

### Existing general planner tests are useful but incomplete for the new scope

Current tests cover:

- pacing model math
- plan serialization
- course discovery behavior
- edge cases in planning logic
- some Streamlit state restoration

Current tests do not cover the new merged-product concerns such as:

- event catalog inheritance
- curated event selection
- GPX waypoint extraction for aid stations
- `<rtept>` support
- dynamic section generation within aid-station boundaries
- legacy-style unified UI rendering across event types

## User requirements captured during interview

These are the most important product decisions already made.

### Core product direction

- The two existing planners should be merged into a single event planning tool.
- The user should choose an event, not choose between planner implementations.
- Event-to-model mapping can be maintained as configuration.

### Required input modes

For all events, the planner must support:

- finish-time-driven planning
- effort-anchor-driven planning

The user also wants a later roadmap feature where reference data may help infer
or estimate finish time, but that is not required for the initial merged build.

### UX direction

- There should be a single UI experience for all events.
- Event selection should happen first.
- Once the event is selected, both target inputs and course overview should be
  visible together.
- Adjusting the target input should update the overview live.
- Advanced controls should appear only when appropriate for the event type,
  especially for trail and ultra events.

### Legacy features to retain

The user wants the merged experience to retain the useful legacy strengths,
including:

- course overview quality
- elevation profile
- section guidance
- pacing tips
- split analysis
- rest-stop-related guidance

Printable outputs are not required as a first-class merged-v1 feature.

### Aid-station and section logic

- Aid points should come from GPX when available.
- Config may override or provide fallback aid distances.
- Aid points are hard section boundaries.
- Within each aid-to-aid block, section guidance should be created dynamically
  from elevation-derived segmentation.

### Event catalog direction

The event catalog should support inheritance from event-type templates.

Template families already agreed:

- `road_half`
- `road_marathon`
- `trail_short`
- `trail_ultra`
- `trail_very_long_ultra`

### Explicit event classification choice

All current GRF events should be treated as `trail_ultra` for now:

- `GRF56`
- `GRF92`
- `GRF166`

### Explicitly rejected for this merged direction

- no custom GPX upload in the curated event-first experience

## Build implications

The first serious implementation step is not "add another option to the radio."
The first step is to define a canonical event/config/domain layer that can drive
one UI and one planner pipeline.

The biggest technical risks discovered before implementation are:

- duplicate logic split across app and package code
- missing `rtept` support for `GRF92`
- missing waypoint-to-aid extraction in the course pipeline
- insufficient event metadata model
- lack of unified result shapes for legacy-style reporting

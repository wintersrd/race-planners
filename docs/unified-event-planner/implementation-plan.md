# Unified Event Planner Implementation Plan

Date: 2026-07-02
Status: In progress

## Progress Log

### 2026-07-02

- Completed checkpoint commit of the pre-build groundwork and removed the tracked
  `semi-marathon-finistere/__pycache__/app.cpython-313.pyc` file from git so
  `pytest` hook runs stop dirtying the index during commits.
- Phase 1 completed:
  - added canonical curated event and event-template models
  - added repository-backed curated event catalog entries for Finistere,
    GRF56, GRF92, and GRF166
  - switched built-in course definitions to derive from the curated event
    catalog instead of a one-off hardcoded course entry
  - began event-first planner integration by selecting curated events and
    automatically deriving course/model defaults in the general planner
  - added catalog and planner-state test coverage for the new event layer
- Phase 2 completed:
  - upgraded the shared GPX parser to support route-point-only GPX files in
    addition to track-point GPX files
  - added waypoint-derived aid-stop extraction in the shared grade/course
    loading path
  - taught curated and library course loading to populate aid stops from GPX
    when catalog overrides are absent
  - added tests for `rtept` parsing and aid-stop extraction behavior
- Phase 3 completed:
  - expanded `Course` with typed aid-station metadata and explicit event/template
    linkage while preserving normalized distance access for the planner
  - upgraded GPX aid extraction to return typed station metadata instead of only
    bare distances
  - preserved catalog override precedence by marking explicit event aid stops as
    `config_override` stations and GPX-derived points as `gpx_waypoint`
  - added tests for typed aid-station extraction and built-in course metadata

## Objective

Implement a single event-first planner that unifies the legacy Finistere app UX
and the newer package-based general planner architecture.

## Delivery strategy

Build this as a sequence of vertical slices that reduce architectural risk early
instead of polishing the wrong abstraction.

## Phase 0: Guardrails and baseline capture

### Goals

- preserve the currently working app startup behavior
- preserve existing planner test coverage
- capture enough regression checks to refactor safely

### Tasks

- keep the current startup fix in place for `semi-marathon-finistere/app.py`
- add or extend tests around current course parsing and legacy outputs where
  upcoming refactors would otherwise be risky
- identify a minimal golden-path regression for Finistere half-marathon output

### Exit criteria

- existing tests still pass
- at least one targeted regression fixture exists for unified-planner migration

## Phase 1: Event catalog foundation

### Goals

- create a canonical curated-event layer
- stop hardcoding event meaning in scattered UI branches

### Tasks

- introduce event catalog data structure under `race_planners/`
- define template keys and inheritance behavior
- create initial curated entries for:
  - Finistere half marathon
  - GRF56
  - GRF92
  - GRF166
- include repository-relative GPX references and template mapping
- support aid-station override metadata in catalog definitions

### Likely code areas

- new event catalog module, likely near `course_library.py` and `models.py`
- `models.py` expansion for event/template metadata
- tests covering catalog loading and validation

### Exit criteria

- app can resolve a selected event into a known event config
- event config can provide course path, template family, and defaults
- tests validate catalog integrity

## Phase 2: GPX and course ingestion hardening

### Goals

- support all current curated event GPX formats
- make course metadata extraction useful for planning

### Tasks

- unify GPX parsing so there is one canonical parser path
- add support for `<rtept>` in addition to `<trkpt>`
- extract waypoint-derived aid markers where possible
- define how waypoint names map to aid semantics
- preserve config override capability when GPX waypoint data is incomplete

### Likely code areas

- `race_planners/grade.py`
- `race_planners/course_library.py`
- `race_planners/models.py`
- removal or reduction of duplicated parser logic in `semi-marathon-finistere/app.py`

### Exit criteria

- Finistere GPX parses correctly
- GRF56 parses correctly
- GRF92 parses correctly despite route-point encoding
- GRF166 parses correctly
- tests cover both `trkpt` and `rtept` GPX structures

## Phase 3: Canonical course and event models

### Goals

- replace thin course metadata with richer event-aware structures
- create a stable contract for downstream planning and UI rendering

### Tasks

- expand `Course` or introduce adjacent typed models for:
  - event metadata
  - aid station details
  - planner template settings
  - localized labels or display metadata
- normalize course loading output so legacy and new UI can depend on the same
  structures

### Exit criteria

- one typed model set can represent both road and trail curated events
- aid station data is available without hand-threading loose dicts

## Phase 4: Unified planner input normalization

### Goals

- keep user-facing inputs flexible while standardizing internal calculation input

### Tasks

- define how finish-time mode maps into internal planner configuration for each
  template family
- define how effort-anchor mode maps into internal planner configuration for
  each template family
- decide how legacy controls like `power_fade` fit into the canonical model
- decide how rest-stop assumptions affect elapsed-time calculations

### Exit criteria

- each template family can produce a normalized internal planning config
- tests cover normalization behavior for both input modes

## Phase 5: Planner engine enrichment

### Goals

- make the package planner capable of supporting legacy-quality outputs

### Tasks

- extend `calculate_plan()` and related models as needed for:
  - richer section summaries
  - aid-station ETAs
  - optional rest-stop timing effects
  - warnings and assumptions
- keep half-marathon compatibility guarded by tests
- avoid re-embedding core calculation logic back into the Streamlit app

### Exit criteria

- canonical planner result can drive the merged UI for both road and trail cases
- legacy compatibility tests still pass or are updated with justified changes

## Phase 6: Dynamic section generation

### Goals

- generate race-meaningful guidance sections automatically

### Tasks

- split courses into aid-station blocks
- perform terrain/elevation segmentation inside each block
- produce user-facing section summaries suitable for display in the unified UI
- tune heuristics differently if needed for road versus trail templates

### Exit criteria

- section summaries are dynamic, not hardcoded prose
- section boundaries always respect aid stations first

## Phase 7: Unified Streamlit UI

### Goals

- deliver one event-first experience

### Tasks

- replace the planner-mode-first UX with event selection first
- show course overview and target input together after event selection
- reveal advanced controls only when the event template requires them
- reuse and adapt the stronger legacy presentation patterns where possible
- keep translations where already supported and practical

### Exit criteria

- user can select any curated event from one UI
- user can use either input mode for that event
- outputs render in one coherent presentation model

## Phase 8: Cleanup and deprecation

### Goals

- reduce architectural debt created by the transition

### Tasks

- remove obsolete mode-branch code when unified flow is stable
- delete duplicated helper paths no longer needed
- update README and any event-specific docs

### Exit criteria

- there is one primary planner path in the app
- duplicate logic is materially reduced

## Testing and validation plan

### Required command suite

- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy tests`
- `uv run pytest`
- `uv run pre-commit run --all-files`

### New test coverage to add during implementation

- event catalog loading and inheritance
- curated event resolution
- GPX parsing for `trkpt` and `rtept`
- waypoint-to-aid extraction
- aid override precedence
- input normalization across both user-facing modes
- dynamic section generation behavior
- merged UI smoke coverage where practical

## Risks

### High-risk areas

- refactoring duplicated legacy logic without changing Finistere behavior
- designing an event model that is rich enough without becoming messy config soup
- interpreting inconsistent GPX waypoint naming across curated events
- preserving the good legacy UX while swapping the engine underneath it

### Risk reduction choices

- migrate toward the package layer instead of adding new inline logic in the app
- add regression tests before major extraction where feasible
- deliver unified flow incrementally, not as one giant rewrite

## Recommended first implementation slice

If implementation resumes after these docs are written, the first slice should
be:

1. add event catalog models and curated event definitions
2. upgrade GPX parsing to handle `rtept`
3. expose curated events in one selector
4. route selected event into the existing planner pipeline with minimal UI churn

That sequence creates a real unification foundation without pretending the UI is
finished before the data model is ready.

## Definition of done for the merged milestone

- one event-first planner experience exists
- Finistere half marathon and all current GRF events are selectable
- every supported event offers finish-time and effort-anchor modes
- GPX ingestion works for all curated events, including `GRF92`
- aid stations come from GPX when available or config when overridden
- dynamic sections respect aid-station boundaries
- outputs preserve the stronger legacy-style experience where it matters
- test, lint, type-check, and pre-commit gates pass

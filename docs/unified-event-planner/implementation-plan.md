# Unified Event Planner Implementation Plan

Date: 2026-07-02
Status: Historical implementation record. Superseded by the package consolidation completed in August 2026.

## Progress Log

### 2026-07-02

- Completed checkpoint commit of the pre-build groundwork and removed the tracked
  obsolete app cache file from git so
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
- Phase 4 completed:
  - added planner-side input normalization so finish-time mode now works for
    trail and ultra models instead of only road-style models
  - derived missing trail pacing anchors from target finish time by numerically
    solving against the shared pacing engine
  - updated the Streamlit planner UI so trail and ultra events can genuinely use
    either finish-time or effort-anchor mode
  - added tests covering finish-time normalization for `technical_trail_ultra`
    and `fire_road_ultra`
- Phase 5 completed:
  - expanded the canonical planner result with typed aid-station ETA data,
    moving time, total rest time, and planner assumptions/warnings so later UI
    phases can render richer outputs without re-deriving timing details in the
    app layer
  - changed rest-stop handling to act as fixed additive elapsed time while still
    preserving moving-time calculations underneath the planner engine
  - introduced a generalized `pacing_bias` control as the canonical replacement
    direction for legacy `power_fade`
  - updated the beta planner output to surface elapsed vs moving time,
    aid-station timing, and richer segment timing summaries
  - added regression coverage for additive rest timing, typed aid-station ETAs,
    pacing-bias behavior, and updated legacy compatibility expectations
- Phase 6 completed:
  - changed segment generation to respect aid-station boundaries as hard section
    breaks before applying terrain-based grouping
  - added block-aware section labels such as `Start to Aid 1` and
    `Aid 1 to Finish` so downstream UI can present race-meaningful guidance
    without re-deriving structure in the app layer
  - kept terrain grouping dynamic inside each aid block by clipping kilometer
    splits to the block boundaries before merging adjacent split types
  - updated beta planner segment output to include derived section and block
    labels
  - added regression coverage for aid-boundary-aware segment generation
- Phase 7 completed:
  - removed the top-level planner-mode switch from the Streamlit entrypoint
    so the app now enters directly into the unified event-first planner flow
  - renamed the planner shell from a beta/general planner framing to the
    canonical unified event planner experience
  - added an always-visible event/course overview beside the event setup so the
    selected event, model, terrain, distance, and aid-station sourcing are shown
    together in one flow
  - removed remaining custom-upload-oriented messaging from the active planner
    UI to align the app with the curated event-only product direction
  - added helper-level coverage for the new course overview rendering inputs and
    preserved app compile coverage for the updated single-entry Streamlit shell
- Phase 8 completed:
  - cleaned repository-facing README text so the project no longer describes the
    removed two-planner split as the active product model
  - updated the Streamlit entry metadata and repository README to describe the
    unified planner as the active experience
  - removed remaining active-product references to GPX re-upload guidance in the
    main README in favor of repository-backed curated course restoration wording
- Post-phase regression remediation completed:
  - restored meaningful trail sensitivity for climb/hike threshold decisions by
    feeding steepest local climb grades into ultra pacing contexts instead of
    relying only on kilometer-average grade
  - wired `rpe_target` and `hr_cap` into both finish-time solving and actual
    plan generation so those controls now materially change outcomes instead of
    acting as no-op UI fields
  - restored visual hierarchy in the unified planner with summary metrics,
    profile charts, and tabbed outputs rather than a flat wall of tables
  - restored segment elevation gain/loss statistics in the canonical result
    model and unified planner sections view
  - added regression coverage for GRF92 threshold sensitivity, RPE/HR guardrail
    effects, and per-segment elevation stats
- Post-phase control-model refinement completed:
  - split the unified planner controls by event family so road events now focus
    on target pace, split bias, and aid-stop time while trail/ultra events focus
    on terrain handling, fade, and aid-station timing
  - introduced preset-backed fade profiles (`Stable`, `Late Fade`,
    `Progressive Fade`, `Blow-Up Risk`) while storing explicit early/mid/late
    phase values in the planner config for later backend evolution
  - added an optional athlete profile schema and persisted it through plan JSON
    export/import so trail defaults can be seeded from known road and trail
    reference data
  - removed technical metadata such as model and GPX filename from the primary
    planner surface and replaced it with plain-language help text tied to the
    visible controls
- Athlete profile integration completed:
  - promoted the athlete profile into a first-class saved object with road
    baselines, trail adjustments, and saved preference fields for split bias,
    fade preset, and default trail effort policy
  - added athlete-profile JSON import/export plus an explicit "apply profile
    defaults to this event" flow so saved baselines can be reused without hidden
    state magic
  - used athlete profile data to derive smarter road and trail anchor defaults,
    trail fade preset defaults, and optional derived HR-guardrail behavior
  - replaced raw user-facing RPE/HR controls with higher-level trail effort
    policy and derived guardrail behavior while preserving backend flexibility
    through the canonical planner config
- Road and athlete model execution plan added:
  - documented the next implementation sequence for capability-source
    precedence, duration-sensitive road solving, event-adjusted best-likely
    outputs, universal tolerance factors, and intent-driven feasibility
    reporting
  - recorded the shared athlete model direction so road and trail refinements can
    proceed from one reference instead of ad hoc UI-only changes
- Road/athlete Phase 1 completed:
  - expanded athlete profile schema with manual best-likely HM/FM fields,
    predictor HM/FM fields and source, plus universal durability, heat-tolerance,
    and hill-tolerance factors
  - exposed those new fields in the saved athlete profile UI so the planner has a
    stable persistence layer before capability-source and road-solver logic are
    added
  - added regression coverage proving the expanded athlete profile survives plan
    JSON restoration and that the new schema fields round-trip correctly
- Road/athlete Phase 2 completed:
  - added explicit road capability-source helpers for manual profile values,
    predictor values, and LT-derived modeled values with clear precedence rules
  - updated road event defaults to seed target finish time from the selected
    capability source instead of fixed static placeholder times when profile data
    is available
  - surfaced a user-facing road capability comparison table in the planner so the
    selected source is visible instead of silently implied
  - added regression coverage for capability-source precedence, modeled road
    best-likely heuristics, and road default seeding from the chosen source
- Road/athlete Phase 3 completed:
  - replaced the old static LT-based road anchor heuristic with a duration-
    sensitive best-likely road solver that iteratively places HM/FM effort
    between LT1 and LT2 based on event duration
  - routed both the modeled road capability display and the road default anchor
    pace through that solver so half marathon and marathon best-likely estimates
    now respond differently to the same physiology inputs
  - added regression coverage proving the solver stays within the LT1/LT2 pace
    envelope, treats marathon pacing more conservatively than half-marathon
    pacing, and places faster half-marathon athletes nearer LT2 than slower ones
- Road/athlete Phase 4 completed:
  - added a road event-adjustment layer that turns selected best-likely road
    capability into event-adjusted best likely using course cost, weather cost,
    hill tolerance, and heat tolerance
  - exposed the adjusted road capability breakdown in the unified planner so the
    user can see selected best likely, course impact, weather impact, and the
    combined adjusted best-likely result before choosing intent
  - added regression coverage proving hotter and hillier road events slow the
    adjusted best-likely outcome and that tolerance factors materially reduce or
    amplify that penalty
- Road/athlete Phase 5 completed:
  - added categorical road race intent (`Best Effort`, `Strong`, `Controlled`,
    `Easy / Durable`) and stored it in planner configuration for road events
  - derived intent-based suggested targets plus user-facing feasibility,
    expected-effort, and recovery-cost labels by comparing the chosen target to
    event-adjusted best likely
  - surfaced those road intent outputs in the unified planner so marathon and
    half-marathon planning can be interpreted as deliberate choices rather than
    only raw target times
  - added regression coverage proving intent shifts suggested targets and the
    derived feasibility / effort / recovery labels in the expected direction
- Road/athlete Phase 6 completed:
  - made the universal athlete factors actually affect trail and ultra
    calculations by threading durability, heat tolerance, and hill tolerance
    through the canonical planner config
  - durability now scales fade severity, heat tolerance now scales weather
    penalty, and hill tolerance now scales trail terrain cost during actual plan
    generation
  - added regression coverage proving those universal factors materially change
    trail event finish times instead of remaining saved-profile metadata only
- Road/athlete Phase 7 completed:
  - clarified the athlete-profile and road-capability UI with more explicit
    wording around precedence, ideal-condition best-likely values, and universal
    factors that apply across road and trail events
  - added plain-language explanation of how selected best likely, adjusted best
    likely, race intent, and chosen target relate to each other in road-event
    planning
  - browser-validated the road event flow to confirm the explanation appears in
    the live unified planner next to the capability and intent controls
- Fueling and nutrition execution plan added:
  - documented the next implementation sequence for aid-station tier modeling,
    energy expenditure, carbohydrate and hydration demand, moving fuel
    schedules, and per-station fueling guidance differentiated by aid station
    capability
- Presentation-layer overhaul completed:
  - added a dedicated presentation-layer execution plan covering hero summary,
    race-story layout, action surfaces, event-aware emphasis, and compact
    race-day snapshot outputs
  - reworked the live result view so the first visible output now emphasizes the
    race story with a stronger summary surface, deterministic plan-meaning
    narrative, and always-visible primary charts before the detailed tabs
  - replaced parts of the raw output with more actionable surfaces for aid
    stations, sections, and fueling while retaining the detailed tables for
    deeper inspection
  - differentiated road and trail result emphasis by tab order and narrative so
    road races highlight target / split context while trail events highlight
    terrain, sections, aid-station experience, and fueling consequences
  - added a compact race-day snapshot surface for quick screenshot-style review

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

- keep the startup behavior in `race_planners/app.py`
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
- include package-owned curated GPX references and template mapping
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
- keep GPX parsing in the shared package parser

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

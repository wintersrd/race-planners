# Unified Event Planner Product and Architecture

Date: 2026-07-02
Status: Build-ready product and technical direction

## Objective

Build one curated event planning application that replaces the current split
between:

- the polished but race-specific legacy planner
- the flexible but less complete general beta planner

The merged application should feel like a single product, not like two apps
wearing the same coat and pretending nobody notices.

## Product definition

### Primary user flow

1. User opens the planner.
2. User selects an event from a curated catalog.
3. The app loads the event's course, template, and defaults automatically.
4. The app shows target inputs and course overview together.
5. The user chooses either finish-time mode or effort-anchor mode.
6. The app updates pacing outputs, sections, and guidance live.
7. Advanced controls appear only if relevant to that event/template.

### Product promise

For any supported event, the user should get:

- the correct course
- the correct pacing model family
- the correct event defaults
- a strong visual course overview
- split and section guidance that feels race-aware

## In-scope requirements

- one unified Streamlit experience
- event-first selection model
- both finish-time and effort-anchor input modes for all events
- legacy-quality course overview and output presentation
- trail/ultra-specific controls when needed
- aid-station-aware planning
- dynamic section generation inside aid-station boundaries
- support for curated Finistere and GRF events

## Out of scope for initial merged milestone

- custom GPX upload for end users
- printable outputs as a first-class feature
- automatic finish-time estimation from imported race history
- full nutrition or hydration planning
- broad event self-service authoring UI

## Canonical architectural direction

### One app shell

There should be one Streamlit entry flow and one planner experience. The current
planner-mode switch is transitional and should not remain the product model.

### One canonical domain layer

The package under `race_planners/` should become the canonical home for:

- event catalog loading
- course loading and GPX parsing
- pacing configuration normalization
- plan calculation
- section generation
- result shaping for UI rendering

The monolithic logic in `semi-marathon-finistere/app.py` should be reduced over
time to UI composition and compatibility glue, then eventually stop owning core
planning logic.

### One canonical data contract

The merged planner should not continue passing around two separate result
shapes. A single typed result model should back all event rendering.

That result model needs to be rich enough to support the stronger legacy-style
outputs.

## Event catalog model

### Why an event catalog is necessary

The user does not want to choose a planner type and then manually wire up a
course. The user wants to choose a real event and have the planner know what
that implies.

### Required event metadata

Each event entry should include, at minimum:

- stable `event_id`
- display name
- optional short name
- event family or template key
- GPX path
- terrain type
- default locale/display language behavior
- model key or template-derived model selection
- aid-station override data when GPX waypoint extraction is missing or noisy
- optional default assumptions per input mode
- optional event notes or warnings
- enabled/disabled state

### Template inheritance

Templates should provide shared defaults and controls by event family.

Approved template families:

- `road_half`
- `road_marathon`
- `trail_short`
- `trail_ultra`
- `trail_very_long_ultra`

### Initial concrete event mapping

At minimum, the catalog should support:

- Finistere half marathon -> `road_half`
- GRF56 -> `trail_ultra`
- GRF92 -> `trail_ultra`
- GRF166 -> `trail_ultra`

Note: `GRF166` may later deserve stronger specialization or a
`trail_very_long_ultra` mapping, but the explicit decision from the discovery
session is to treat all current GRF events as `trail_ultra` initially.

## Course ingestion direction

### Required behavior

- curated events should resolve GPX from repository-controlled files
- GPX parsing must support both track-point and route-point structures
- course metadata should extract aid-like waypoints when possible
- config should be able to override waypoint interpretation cleanly

### Parser upgrade requirement

The parser must support both:

- `<trkpt>`
- `<rtept>`

without duplicating the logic in separate implementations.

### Aid-station extraction requirement

Course ingestion should produce richer aid data than plain distance floats when
possible, for example:

- label/name
- distance along course
- source type: GPX waypoint or config override
- optional waypoint class or notes

The UI may still render simplified summaries at first, but the pipeline should
retain the richer structure.

## Section generation model

### Product rule

Aid points are hard boundaries.

Within each boundary pair, the app should generate smaller guidance sections
dynamically from elevation/grade structure.

### Consequence for implementation

The segmentation pipeline should likely become two-stage:

1. split course into aid-station blocks
2. segment each block into meaningful terrain sections

The output should support user-facing guidance such as:

- runnable rolling segment
- sustained climb
- steep technical descent
- recovery/settling section

The exact label vocabulary can start simple, but the pipeline should be designed
for extensibility.

## Input model direction

### Universal user-facing modes

All events must expose both:

- finish-time mode
- effort-anchor mode

### Template-specific advanced controls

Templates should control which advanced inputs are shown.

Examples:

- `road_half`
  - target finish time
  - base pacing assumptions
  - possibly power-fade or split preference controls
- `trail_ultra`
  - flat pace or aerobic anchors
  - hike pace
  - climb threshold or equivalent
  - descent caution where relevant
  - rest-stop assumptions if they affect total time

### Important design constraint

The UI contract can be template-specific while the planner engine still receives
a normalized internal configuration object.

## Result model direction

The canonical plan result should support at least:

- overall timing summary
- per-kilometer or per-unit splits
- aid-station ETAs
- section summaries
- warnings/assumptions
- optional rest-stop timing data
- chart-friendly course profile and pace profile data

This is broader than the current package `PlanResult` and may require expansion
instead of inventing a second competing result shape.

## UI direction

### Preserve the best of legacy

The merged tool should preserve or reintroduce:

- stronger visual hierarchy
- course profile prominence
- race-aware summaries
- tabbed or sectioned outputs that are easy to scan
- translations where already supported and valuable

### Avoid the wrong simplification

Do not flatten the merged planner into a generic table-driven utility UI merely
because it is easier to wire. That would technically merge the planners while
still throwing away the part users actually liked.

## Migration strategy

### Near-term

- keep using the existing Streamlit entry point
- introduce event catalog and unified flow behind the current app shell
- reuse package modules where possible
- extract legacy-only logic into package modules rather than copy it again

### Medium-term

- move remaining planner logic out of `semi-marathon-finistere/app.py`
- make the UI depend on the canonical package API
- remove the planner-mode split when unified rendering is ready

## Open implementation questions

These are not product-direction blockers, but they need resolution during build.

- Should `power_fade` survive as a normalized control, or should it become a
  more general pacing-bias concept?
- How much of legacy rest-stop handling should affect total elapsed planning in
  v1 versus remain informational output?
- Should event catalog data live as JSON, YAML, or Python-native config?
- How should bilingual text be represented for event metadata and dynamic labels?
- Should `GRF166` remain on `trail_ultra` defaults only, or get heavier fatigue
  and section-tuning overrides immediately?

## Build principle

Prefer one expanding canonical pipeline over temporary adapters that freeze the
legacy/general split in place. A small adapter is fine for migration. A second
permanent architecture is not.

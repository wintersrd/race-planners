# Presentation Layer Implementation Plan

Date: 2026-07-08
Status: Ready for execution

## Objective

Rework the planner presentation layer so the result view reads as the story of a
race, not as a dump of all available tables.

The target experience should answer three questions immediately:

1. What am I likely to do?
2. Where are the important moments?
3. What should I pay attention to?

## Design Principles

- Present the race before the diagnostics.
- Use the same engine, but emphasize different outputs for road vs trail.
- Turn dense tables into decision surfaces where possible.
- Keep deeper detail available, but not dominant.
- Avoid AI-style narration; summaries should be deterministic and directly
  derived from the model.

## Scope

This pass focuses on presentation and structure only. It does not change the
underlying pacing math.

## Phase 1: Race Summary Hero + Primary Race Story

### Goals

- Replace the plain metric strip with a stronger top summary surface.
- Make the first visible content tell the story of the day before tabs take over.

### Build details

- Add a summary hero panel near the top of `Plan Output` with:
  - predicted finish time
  - moving time
  - average pace
  - elevation gain / loss
  - fueling target summary
- Add a deterministic `What this plan means` block below the hero.
- Make the primary race story block always visible before tabs:
  - course profile
  - cumulative time chart
  - first-half / second-half comparison

### Deterministic summary rules

- Road events:
  - use race intent + target vs adjusted best-likely + split bias
  - examples:
    - controlled plan
    - stretch target
    - stronger back-half bias
- Trail events:
  - use fade preset + descent caution + heat + aid count
  - examples:
    - moderate fade
    - technical descents
    - heat-sensitive day

## Phase 2: Turn Output Tables into Decision Surfaces

### Goals

- Keep the numeric detail available, but add a more visual/actionable layer.

### Aid Stations

- Add card-style or grouped surfaces above the table.
- Each aid station should show:
  - label
  - distance
  - ETA / departure time
  - station type / tier
  - suggested stop duration
  - short station instruction

### Sections

- Add section cards above the detailed table.
- Each section card should show:
  - distance range
  - terrain type
  - elevation gain / loss
  - expected pace band
  - short pacing cue

### Fueling

- Add a visual fueling timeline before the fueling table.
- Each fueling block should show:
  - distance range
  - duration
  - what to carry
  - what is available at the station
  - fluid target

### Splits

- Keep the split table, but make it visually guided:
  - pace-zone tinting
  - terrain tinting
  - compact / expanded reading emphasis

## Phase 3: Differentiate Road vs Trail Result Emphasis

### Goals

- Use the same result system, but make different event families feel purpose-fit.

### Road event emphasis

- Put more weight on:
  - target vs adjusted best-likely
  - split shape
  - first-half vs second-half
  - recovery cost and feasibility framing
- De-emphasize:
  - large fueling surfaces
  - terrain-heavy framing

### Trail / ultra emphasis

- Put more weight on:
  - terrain sections
  - climb / hike behavior
  - aid-station experience
  - fueling blocks
  - heat / fade consequences
- De-emphasize:
  - road-style intent framing after the plan is calculated

### Implementation approach

- Use event-family conditionals in `streamlit_general.py` to:
  - reorder emphasis blocks
  - alter summary wording
  - suppress or soften less-relevant sections

## Phase 4: Compact Race-Day Snapshot

### Goals

- Produce a condensed race-day reference view without full printable support.

### Build details

- Add a compact snapshot section or tab containing:
  - finish estimate
  - average pace
  - most important aid stations
  - highest-cost sections
  - fueling carry summary
  - heat / caution summary
- Keep it screenshot-friendly and compact.

## Supporting details

### New helper responsibilities likely needed

- summary/narrative helpers:
  - hero metric builder
  - road narrative builder
  - trail narrative builder
- action-surface helpers:
  - aid station cards
  - section cards
  - fueling timeline rows
- style helpers:
  - pace band labels / colors
  - section cue generation

### Translation implications

- New summary labels, coaching phrases, and card labels must route through
  `race_planners/i18n.py`.
- Keep the main in-app `How the Models Work` copy high-level only.
- Deeper material stays in the technical reference docs.

## Validation plan

- Targeted:
  - `uv run ruff check race_planners tests docs/unified-event-planner`
  - `uv run pytest tests/test_streamlit_general.py tests/test_i18n.py`
- Full:
  - `uv run pytest`
  - `uv run pre-commit run --all-files`
- Browser:
  - verify road event summary prioritizes intent / target / split
  - verify trail event summary prioritizes terrain / fade / aid / fueling
  - verify aid cards and section cards render before raw tables
  - verify compact race-day snapshot is readable without scrolling through raw
    data tables first

## Definition of done

- The first visible result area clearly tells the story of the race.
- Road and trail outputs feel meaningfully different.
- Aid stations, sections, and fueling are understandable without reading raw
  spreadsheets first.
- Deep detail is still available, but no longer dominates the interface.

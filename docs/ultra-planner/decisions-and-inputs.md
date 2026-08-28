# General Race Planner Decisions and Inputs

Date: 2026-05-20
Source: User discussion and planning session
Purpose: durable handoff context for continued implementation

## Product direction

- Selected architecture direction: general planner with pluggable pacing models
- App should support road marathon, half marathon, fire-road ultra, technical-trail ultra
- Existing single-race app behavior should be retained as one model/preset, not app identity

## Course and route decisions

- Built-in courses are curated, package-owned GPX files under `race_planners/data/courses/`
- GPX upload and writable local course libraries are intentionally out of scope
- Saved JSON plans reference curated GPX basenames (no embedding required)

### Missing GPX reload behavior

If a saved plan references a curated GPX that is not bundled when reloading, the planner should fail clearly.

Preferred error style:

`Missing course file: <filename>. This plan requires a bundled curated course.`

## Input mode decisions

- Input mode must be toggleable:
  - Finish-time driven
  - Effort-anchor driven

### Marathon preference

- Marathon planning should support known base marathon pace directly
- This should feel similar to current half-marathon target-time/pace interaction

## Effort-anchor modeling decisions

### Fire-road ultra

- Explicitly model both Z1 and Z2 (not one collapsed value)
- Include hike pace
- Bias between Z1 and Z2 should be model-driven (time, distance, climb load)

### Technical-trail ultra

- Use flat pace input
- Use hike pace input
- Use descent caution preset (not direct descent pace input)

## Descent caution behavior (technical trail)

- Descent caution is an adjustment against downhill GAP
- Behavior should be grade-sensitive:
  - Mild descent (around -5%): near GAP
  - Steeper descent: progressively reduced downhill GAP benefit
  - Very steep descent (around -20%): may be slower than flat pace due to risk

## Required outputs

- Aid station arrival timing
- Segment pacing outputs
- Save/download and reload plan state (JSON)

## Nice-to-have outputs

- Scenario comparison
- Rough calorie estimate (planning-level only)

## Constraints and assumptions

- Tool is mostly for personal use but should remain clean and maintainable
- No nutrition/hydration planning required in MVP
- Initial implementation should prioritize correctness and revisitability over UI complexity

## Open items intentionally deferred

- Scenario comparison UX and data model
- Calorie estimate model choice
- Advanced terrain metadata taxonomy for built-in course library

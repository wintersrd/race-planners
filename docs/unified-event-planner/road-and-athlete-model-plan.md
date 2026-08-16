# Road And Athlete Model Plan

Date: 2026-07-04
Status: Approved for execution

## Why this exists

Trail and ultra planning now use athlete profile data, fade modeling, and
weather in a way that feels grounded. Road event planning still lags behind.

The missing layer is not just more knobs. The missing layer is an explicit model
of:

- athlete capability
- event-adjusted potential
- race intent
- feasibility and effort interpretation

The planner currently behaves more like "tell me the target and I will pace it"
than "given this athlete, course, weather, and intent, what is plausible and
how hard will it be?"

This document defines the next execution sequence for closing that gap.

## Product outcome

For road events, the planner should be able to say:

- what the model thinks the athlete's best likely time is in ideal conditions
- what that best likely time becomes after course and weather adjustments
- how the athlete's chosen target compares to that adjusted best-likely result
- whether the chosen target looks easy, controlled, strong, near-limit, or
  aggressive
- what the likely recovery cost is

This should work for both half marathon and marathon planning.

## Conceptual model

### 1. Athlete envelope

Athlete profile should describe the physiological and experiential envelope the
planner can use across both road and trail contexts.

Core fields:

- LT1 pace / HR
- LT2 pace / HR
- trail slowdown factors
- durability factor
- heat tolerance
- hill tolerance

Road-specific capability fields:

- best likely half time
- best likely marathon time
- predictor half time
- predictor marathon time
- predictor source

### 2. Capability source precedence

When multiple capability signals are present, precedence is:

1. manual best-likely override
2. predictor estimate
3. physiology-derived estimate from LT1/LT2

The planner must show all available sources and make the selected source
explicit in the UI. It should not silently override one source with another.

### 3. Duration-sensitive road effort model

Road best-likely modeling cannot assume that all half marathons sit at the same
fraction of threshold effort.

Examples:

- a faster half marathon athlete may race very close to LT2
- a slower half marathon athlete may need to race somewhere between LT1 and LT2
- marathon best-likely effort sits lower again and shifts with durability and
  conditions

The solver therefore needs to iteratively relate:

- pace
- duration
- physiological effort level

### 4. Event-adjusted best likely

After determining a capability baseline, the planner should adjust it for:

- course profile / terrain cost
- heat / weather penalty
- hill tolerance
- heat tolerance

This produces an `adjusted_best_likely_time` that is specific to the selected
event.

### 5. Intent layer

The user should choose categorical race intent, not raw numeric effort terms.

Intent categories:

- `Best Effort`
- `Strong`
- `Controlled`
- `Easy / Durable`

These categories should shift the target away from adjusted best-likely and
drive interpretation outputs such as feasibility, expected effort, and recovery
cost.

## Shared athlete factors

The following factors are universal and should influence both road and trail
modeling:

- `durability_factor`
- `heat_tolerance`
- `hill_tolerance`

### Durability factor

Purpose:

- reduces or amplifies late-race slowdown
- changes how aggressively duration drives performance decay

### Heat tolerance

Purpose:

- scales weather penalties
- affects both road and trail outcomes

### Hill tolerance

Purpose:

- scales grade-driven performance cost
- affects rolling-road courses and trail climbs/descents

## Road UX target

### Inputs

Road events should show:

- goal mode: finish time or anchor pace
- race intent
- split strategy / split bias
- aid stop time
- expected peak temperature

Athlete profile section should expose, save, and import:

- best likely half
- best likely marathon
- predictor half
- predictor marathon
- predictor source
- universal tolerance factors

### Displayed derived values

Road events should show, next to user inputs:

- modeled best likely
- predictor best likely
- selected best likely source
- adjusted best likely
- chosen target
- feasibility band
- expected effort band
- recovery cost band

### Hidden technical details

The main UI should not show template IDs, race model names, or file names.
Those belong in debug exports or logs if needed.

## Solver direction

### Phase 3 road best-likely solver

The solver should:

1. start with a candidate duration
2. map duration to an effort fraction between LT1 and LT2
3. derive a road base pace from that fraction
4. run course and weather adjustments
5. produce a new duration
6. iterate until the duration stabilizes

This is the core change that makes fast and slower half marathon athletes behave
differently in the way the user described.

### Effort fraction model

The solver should not hardcode a single fixed effort fraction by event type.
Instead it should use a tunable duration-to-effort function with durability as a
modifier.

High-level expectations:

- shorter road events sit nearer LT2
- longer road events sit further from LT2
- higher durability keeps the athlete nearer the stronger end longer
- low durability pulls longer-event best likely closer to LT1

## Feasibility and event cost

After the adjusted best-likely result is established, compare chosen target
against that result to classify:

- `feasibility_band`
- `effort_band`
- `recovery_cost_band`

Suggested output labels:

- Feasibility: `Very High`, `Reasonable`, `Stretch`, `Aggressive`
- Effort: `Controlled`, `Strong`, `Near Limit`, `Maximal`
- Recovery cost: `Low`, `Moderate`, `High`, `Very High`

## Execution phases

### Phase 1: Athlete profile schema expansion and persistence

Add profile fields for:

- `best_likely_half_time_min`
- `best_likely_marathon_time_min`
- `predictor_half_time_min`
- `predictor_marathon_time_min`
- `predictor_source`
- `durability_factor`
- `heat_tolerance`
- `hill_tolerance`

Persist them through profile JSON and saved plan JSON.

### Phase 2: Capability source layer

Compute and display:

- manual capability values
- predictor values
- physiology-derived modeled values
- selected source according to precedence

This phase is primarily about explicit comparison and source selection logic.

### Phase 3: Duration-sensitive road solver

Implement the iterative road best-likely solver described above.

This phase should produce stable modeled HM/FM best-likely values from LT1/LT2
data.

### Phase 4: Event adjustment layer

Derive event-adjusted best-likely values using:

- course cost
- weather cost
- hill tolerance
- heat tolerance

### Phase 5: Intent and feasibility layer

Add categorical road intent and derived labels for:

- feasibility
- expected effort
- recovery cost

### Phase 6: Trail integration of universal factors

Make the new universal tolerance factors affect trail behavior too:

- durability influences fade severity
- hill tolerance influences terrain cost
- heat tolerance scales weather penalty

### Phase 7: UI cleanup and explanation

Refine the UI copy and presentation so the model is understandable:

- plain-language descriptions
- visible side-by-side comparison of capability sources
- clearly explained adjusted best-likely and intent outputs

## Acceptance criteria by theme

### Profile and persistence

- new athlete profile fields are editable and saved
- import/export survives round-trip
- saved plans can restore associated profile data

### Solver behavior

- fast half marathon athletes can model effort nearer LT2 than slower athletes
- marathon modeled best-likely is duration-sensitive
- solver converges reliably

### Adjusted best likely

- same athlete gets different best-likely outcomes on flat/cool vs rolling/warm
  events
- heat tolerance and hill tolerance materially affect those results

### Intent interpretation

- `Controlled` produces a less demanding plan than `Best Effort`
- feasibility / effort / recovery labels respond coherently

### Trail coherence

- the new tolerance fields affect both road and trail
- trail control model remains intact

## Risks

- false precision from sparse athlete data
- unstable iterative solver if the effort-duration function is too aggressive
- road UI becoming too complicated
- source precedence confusing users if not displayed clearly

## Risk controls

- make capability source explicit
- use categorical intent rather than numeric event-cost sliders
- keep the effort-duration function bounded and monotonic
- prefer comparative tests and monotonicity tests over brittle exact-value tests

## Implementation note

This document is the execution reference for the next build slices. Update the
main implementation log after each phase completes, including validation and
commit references.

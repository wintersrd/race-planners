# Fueling And Nutrition Plan

Date: 2026-07-04
Status: Approved for execution

## Why this exists

The planner already models distance, elevation, pace, weather, and aid-station
timing. The missing layer is fueling: how many calories the athlete burns, how
many carbs they need, what to consume at each aid station, and what to carry
between stations.

This is especially valuable for ultra events where underfueling early leads to
late-race collapse. A per-aid-station fueling checklist tied to our existing
ETA timing would be a real differentiator.

## Key insight from real ultra data

Fueling happens in two places, not one:

1. **While moving** between aid stations (gels, chews, carried fluids)
2. **At aid stations** with varying levels of support

Real-world example from a 53km ultra:

- KM 0-18: 2 gels (110g carbs) consumed while running over 2 hours
- Aid @ KM18: small snack on-site
- KM 18-38: 2 gels (110g carbs) consumed while running over 2 hours
- Aid @ KM38: sports drink only, no food
- KM 38-45: nothing consumed over 45 minutes
- Aid @ KM45: small snack
- KM 45-53: planned gel skipped

This shows:

- moving fuel is scheduled and carried, not improvised
- aid stations vary dramatically in what they offer
- real consumption often falls short of plan, especially late

## Aid station tier model

| Tier           | What's available                                 | Realistic on-site kcal | Typical dwell |
| -------------- | ------------------------------------------------ | ---------------------- | ------------- |
| `water_only`   | Water, maybe sport drink                         | 0-40                   | 0.5-1 min     |
| `standard`     | Water, sport drink, bananas, gels, cookies, coke | 100-250                | 2-5 min       |
| `full_service` | Hot food, soup, pasta, sandwiches, drop bags     | 300-800+               | 5-15 min      |

Default tier for unknown / GPX-derived stations: `standard`.

Tier influences:

- realistic on-site consumption
- sensible default dwell time
- what the athlete needs to carry out for the next segment

## Energy expenditure model

### Running cost baseline

Level running cost is approximately `1 kcal/kg/km` (4.2 kJ/kg/km).

### Grade adjustment

Use Minetti cost-of-grade curve adjusted for body mass:

- Uphill: adds proportional to `mass * g * elevation_gain`
- Downhill: recovers ~50% at moderate grades, less at steep grades
- Net kcal still positive even on steep descents (eccentric braking cost)

### Formula approach

```
segment_kcal = mass_kg * distance_km * (
    1.0
    + uphill_cost(grade_percent)
    + downhill_cost(grade_percent)
)
```

Where:

- `uphill_cost` adds based on positive grade (Minetti coefficients)
- `downhill_cost` adds modestly for moderate negatives, more for steep
- Net result always positive

### Total outputs

- Total event kcal burn
- Hourly kcal burn curve
- Per-aid-station-block kcal burn
- Kcal/kg/hr for intensity context

## Carbohydrate demand model

### By event duration and intensity

| Duration   | Carb target (g/hr) | Notes                                    |
| ---------- | ------------------ | ---------------------------------------- |
| < 60 min   | 0-20               | Body runs on glycogen, no fueling needed |
| 60-90 min  | 20-40              | Light fueling helps                      |
| 90-180 min | 40-60              | Marathon / fast half territory           |
| 3-6 hours  | 60-90              | Standard ultra, gut training helps       |
| 6-12 hours | 40-75              | Lower intensity allows fat oxidation     |
| 12+ hours  | 30-60              | Survival mode, gut stress dominates      |

### Intensity modifier

Higher intensity (closer to LT2) shifts carb reliance up. We can use the
athlete's effort fraction from the road solver or trail effort policy to
modulate within the band.

### Output

- Target carb band per hour (low, high)
- Target carb per aid-station block
- Cumulative carb plan vs burn

## Hydration model

### Sweat rate baseline

~0.4-1.0 L/hr depending on temperature and intensity.

### Temperature scaling

- Cool (<10C): 0.4-0.6 L/hr
- Mild (10-20C): 0.5-0.8 L/hr
- Warm (20-28C): 0.7-1.2 L/hr
- Hot (>28C): 1.0-1.5 L/hr

Already have diurnal temperature model from weather integration.

### Output

- Target fluid per hour
- Target fluid per aid-station block
- Sodium recommendation (rough: 300-700 mg/hr depending on sweat rate)

## Moving fuel schedule

### What gets modeled between aid stations

For each aid-to-aid block:

- block duration
- block distance
- block kcal burn
- target carbs for the block
- suggested fuel items (gels, chews, etc.) to carry
- timing guidance (e.g., "gel every 30-40 minutes")

### Fuel item reference

Standard fuel item carb content:

- Gel: 20-25g carbs
- Chews (5-6 pieces): 20-30g carbs
- Sport drink (500ml): 25-35g carbs
- Banana: 25-30g carbs
- Waffle/wafer: 20-30g carbs

### Realistic constraints

- Max carb absorption: ~60-90 g/hr (glucose+fructose blend)
- Max carb absorption: ~30-60 g/hr (glucose only)
- Gut stress increases late in long events
- Late-race intake often drops 30-50% vs plan

## Aid station fuel plan

### By tier

- `water_only`: no on-site food assumed; carry everything needed
- `standard`: assume 100-250 kcal on-site (1-2 items); top up fluids
- `full_service`: assume 300-800 kcal on-site; substantial refuel possible

### Dwell time interaction

Tier can drive a sensible default rest time:

- `water_only`: 30-60 sec
- `standard`: 120-300 sec
- `full_service`: 300-900 sec

User override always wins.

### Output per station

- station tier
- realistic on-site kcal
- realistic on-site carbs
- recommended carry-out items for next block
- cumulative carb surplus/deficit

## Data model additions

### `AidStation` expansion

Add `tier: str = "standard"` field. Values: `water_only`, `standard`,
`full_service`.

### `AthleteProfile` expansion

Add:

- `body_mass_kg: float | None = None`
- optional `sweat_rate_l_hr: float | None = None`
- optional `gut_carb_tolerance_g_hr: float | None = None`

### New result types

```python
@dataclass
class FuelingBlock:
    start_km: float
    end_km: float
    distance_km: float
    duration_min: float
    kcal_burned: float
    carb_target_g: float
    fluid_target_l: float
    carry_items: list[str]
    consumed_while_moving_g: float
    consumed_at_aid_g: float
    cumulative_deficit_g: float

@dataclass
class FuelingPlan:
    total_kcal: float
    avg_kcal_hr: float
    total_carb_target_g: float
    total_carb_planned_g: float
    total_fluid_target_l: float
    blocks: list[FuelingBlock]
    carb_deficit_g: float
    warnings: list[str]
```

## Execution phases

### Phase 1: Aid station tier schema and catalog data

- add `tier` to `AidStation`
- allow curated events to specify per-station tiers
- default GPX-derived and config-only stations to `standard`
- update catalog where known tiers differ (e.g., Finistere water-only stops)

### Phase 2: Energy expenditure model

- implement kcal-per-segment calculation using mass, distance, grade
- integrate Minetti-style grade cost
- sum to total event kcal and per-block kcal

### Phase 3: Carb and hydration demand model

- implement duration and intensity-based carb target bands
- implement temperature-scaled hydration targets
- produce per-block carb and fluid targets

### Phase 4: Moving fuel schedule

- generate carry-out fuel recommendations per aid-to-aid block
- schedule gel/fuel timing within each block
- respect gut absorption limits
- account for realistic late-race intake drop

### Phase 5: Aid-station fuel plan by tier

- compute realistic on-site consumption per station tier
- compute carry-out needs based on next block and next station tier
- track cumulative carb surplus/deficit

### Phase 6: Fueling summary output and UI

- add fueling tab to unified planner output
- show total kcal, carb plan, hydration plan
- show per-station fueling checklist
- show cumulative deficit/surplus chart

### Phase 7: Tests and validation

- energy model sanity (half vs marathon vs ultra scaling)
- carb targets sensible by duration
- aid station tier affects on-site and carry-out correctly
- realistic ultra example produces sensible block-by-block plan
- temperature affects hydration target

## Acceptance criteria

- `AidStation` carries a tier and it round-trips through catalog and JSON
- energy model produces sensible kcal for half, marathon, and ultra distances
- carb targets scale with duration and intensity as described
- moving fuel schedule accounts for both carry-out and at-station consumption
- aid station tier materially changes the fueling recommendation
- fueling plan surfaces a cumulative carb deficit/surplus metric
- UI shows a fueling tab with actionable per-block guidance
- all existing tests still pass

## Risks

- false precision in kcal model (±15% is realistic)
- carb absorption varies wildly by individual
- fuel item reference is approximate
- trail intensity is non-steady so hourly estimates are noisier

## Risk controls

- show bands, not single numbers, for carb and hydration targets
- flag late-race gut stress explicitly
- keep fuel item suggestions as examples, not prescriptions
- acknowledge model uncertainty in the UI copy

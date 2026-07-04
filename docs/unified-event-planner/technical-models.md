# Technical Modeling Reference

This document explains the mathematical models and algorithms behind the Unified Event Planner. It is written for curious runners who want to understand the assumptions, and for developers who need to trace the logic in the code.

---

## Road Capability Model

**Module**: `race_planners/road_capability.py`

### Overview

The road capability model estimates your best-likely finish time for a given race distance based on your lactate threshold (LT1/LT2) paces, then adjusts that estimate for course difficulty and weather.

### Effort Fraction Table

The model uses an effort-fraction lookup table that maps expected race duration to a fraction of your LT1-to-LT2 pace range:

| Duration | Half Marathon Fraction | Marathon Fraction |
| -------- | ---------------------- | ----------------- |
| 60 min   | 0.45                   | —                 |
| 90 min   | 0.38                   | —                 |
| 120 min  | 0.32                   | 0.15              |
| 180 min  | —                      | 0.05              |
| 240 min  | —                      | -0.02             |
| 360 min  | —                      | -0.10             |

A fraction of 0.0 means you race at the midpoint between LT1 and LT2 pace. Positive fractions mean closer to LT2 (faster). Negative fractions mean below LT1 (slower, for longer events).

### Fixed-Point Solver

The solver iterates because duration depends on pace, and pace depends on duration:

1. Guess an initial duration (e.g., 90 min for half marathon)
2. Look up the effort fraction for that duration
3. Interpolate between LT1 and LT2 pace using the fraction
4. Calculate a new duration from the interpolated pace × race distance
5. Repeat until the duration stabilizes (up to 24 iterations)

This converges quickly because the fraction changes slowly with duration.

### Course and Weather Adjustments

The adjusted best-likely time applies two multipliers:

- **Course multiplier**: Derived from the course's Grade-Adjusted Pace (GAP) factor. Hilly courses have a multiplier > 1.0. Scaled by your hill tolerance.
- **Weather multiplier**: Derived from the average heat multiplier across the expected duration. Hot conditions have a multiplier > 1.0. Scaled by your heat tolerance.

These are applied iteratively (4 passes) because weather cost depends on duration, which depends on weather cost.

### Classification Thresholds

| Metric      | Delta %  | Label      |
| ----------- | -------- | ---------- |
| Feasibility | >= 6%    | Very High  |
|             | >= 2%    | Reasonable |
|             | >= -1.5% | Stretch    |
|             | < -1.5%  | Aggressive |
| Effort      | >= 8%    | Controlled |
|             | >= 3%    | Strong     |
|             | >= -1%   | Near Limit |
|             | < -1%    | Maximal    |
| Recovery    | >= 8%    | Low        |
|             | >= 3%    | Moderate   |
|             | >= -1%   | High       |
|             | < -1%    | Very High  |

Delta % = (chosen target - adjusted best likely) / adjusted best likely × 100.

---

## Weather / Heat Penalty

**Module**: `race_planners/weather.py`

### Temperature-to-Pace Penalty Curve

The penalty is based on commonly cited running performance data:

| Temperature | Pace Penalty              |
| ----------- | ------------------------- |
| <= 10°C     | 0%                        |
| 15°C        | 1.5%                      |
| 20°C        | 3.5%                      |
| 25°C        | 6.5%                      |
| 28°C        | 8.5%                      |
| 30°C        | 10%                       |
| 35°C        | 15%                       |
| > 35°C      | +0.5% per degree above 35 |

Between table points, the penalty is linearly interpolated.

### Diurnal Temperature Model

Rather than applying a fixed penalty, the model estimates the actual temperature at each point in the race:

- Uses a cosine curve with the coldest point around 4:00 AM
- Assumes a 12°C swing between daily low and peak temperature
- The event's `start_time_local` anchors the curve
- For multi-day events (e.g., 166 km ultras), the curve naturally wraps via modulo 24 hours

This means a race starting at 6:30 AM in 20°C peak conditions will model cooler temperatures in the early kilometers and warmer temperatures around midday.

### Heat Tolerance Scaling

The athlete's heat tolerance (-1.0 to +1.0) scales the penalty:

- Positive values reduce the penalty (better heat coping)
- Negative values amplify it (worse heat coping)
- The scaling uses a `tolerance_penalty_scale` function that clamps the effect between 0.6× and 1.4× the base penalty

---

## Fatigue, Fade, and Durability

**Module**: `race_planners/fatigue.py`

### Per-Model Fatigue

Each race model has a fatigue multiplier that increases pace (slows you down) as progress through the race increases:

| Race Model            | Base Slope | Max Fatigue at 100% |
| --------------------- | ---------- | ------------------- |
| Half Marathon         | 0.04       | +4% pace            |
| Road Marathon         | 0.06       | +6% pace            |
| Fire Road Ultra       | 0.06       | +6% pace            |
| Technical Trail Ultra | 0.08       | +8% pace            |

The fatigue multiplier is: `1.0 + slope × progress_ratio`, clamped to a maximum (typically 0.65× additional cost).

### Fade Profile Presets

Fade profiles describe how pace deteriorates across an ultra event. Each preset defines three phase values (early, mid, late) that are interpolated by progress ratio:

| Preset           | Early | Mid  | Late |
| ---------------- | ----- | ---- | ---- |
| Stable           | 0.0   | 0.75 | 1.5  |
| Late Fade        | 0.0   | 1.25 | 3.5  |
| Progressive Fade | 0.5   | 2.0  | 4.5  |
| Blow-Up Risk     | 1.5   | 4.0  | 7.0  |

The fade multiplier is: `1.0 + fade_bias × 0.006`, where `fade_bias` is interpolated across the three phases.

### Durability Multiplier

The durability factor (-1.0 to +1.0) applies a time-and-distance-weighted pace penalty:

- `progress_load = progress_ratio^2.4` — exponential load curve
- `duration_load = elapsed_hours / 8.0` — scales with event length
- `breakdown_load = progress_load × duration_load` — combined load
- `multiplier = 1.0 - (durability × 0.08 × breakdown_load)`, clamped to [0.8, 1.2]

This means durability has minimal effect on a half marathon but dramatically affects a 100-mile ultra. For example:

- 21 km event: ~0.9 min swing between durable (+1.0) and fragile (-1.0)
- 92 km event: ~98 min swing
- 166 km event: ~198 min swing

---

## HR Guardrail

**Module**: `race_planners/guardrails.py`

### Overview

The HR guardrail estimates your heart rate at each segment of the course and applies a pace penalty if it exceeds a dynamic ceiling. It requires LT1 and LT2 heart rate data from the athlete profile.

### Segment HR Estimation

The model estimates HR using a three-piece linear interpolation:

1. **Below LT1 pace**: HR scales linearly from resting toward LT1 HR
2. **Between LT1 and LT2 pace**: HR scales linearly from LT1 HR toward LT2 HR
3. **Above LT2 pace**: HR scales steeply above LT2 HR

Trail terrain slows your pace for a given effort, so the model converts trail pace to road-equivalent pace before estimating HR.

### Dynamic HR Ceiling

The ceiling isn't a fixed number — it varies by race phase:

| Effort Policy | Early Offset | Mid Offset | Late Offset |
| ------------- | ------------ | ---------- | ----------- |
| Conservative  | -5 bpm       | -3 bpm     | +2 bpm      |
| Steady        | -3 bpm       | 0          | +3 bpm      |
| Aggressive    | -1 bpm       | +2 bpm     | +5 bpm      |

Negative offsets lower the ceiling (more conservative). Positive offsets raise it (allow harder effort). Terrain difficulty further lowers the ceiling temporarily on steep segments.

### Pace Penalty

When estimated HR exceeds the dynamic ceiling:

- `severity = min(1.0, overage_bpm / 10.0)`
- `climb_weight = min(1.0, steepest_climb / 15.0)`
- `late_weight = progress_ratio`
- `multiplier = 1.0 + severity × 0.12 × (0.4 + climb_weight × 0.35 + late_weight × 0.25)`

This slows you down more on steep terrain and late in the race, which is exactly when HR management matters most.

---

## Fueling and Nutrition Model

**Module**: `race_planners/fueling.py`

### Energy Expenditure

Energy cost is based on the standard running economy approximation of ~1 kcal per kg body mass per km, modified by grade:

- **Flat/road**: ~1 kcal/kg/km
- **Uphill**: up to +4.5× the grade fraction (e.g., 10% grade adds ~45% cost)
- **Downhill**: +0.6× at moderate grades (downhill is slightly cheaper but not free)
- **Steep downhill** (below -20%): capped to avoid negative costs

This is derived from Minetti's metabolic cost data for gradient running.

### Carbohydrate Target Bands

Carb targets are duration-banded and reflect gut absorption limits:

| Duration | Carb Target (g/hr) |
| -------- | ------------------ |
| < 1 hr   | 0–20               |
| < 1.5 hr | 20–40              |
| < 3 hr   | 40–60              |
| < 6 hr   | 60–90              |
| < 12 hr  | 40–75              |
| 12+ hr   | 30–60              |

The reduction for very long events reflects practical gut absorption limits during sustained endurance effort.

### Hydration

Sweat rate is estimated from temperature:

| Temperature | Sweat Rate (L/hr) |
| ----------- | ----------------- |
| 10°C        | 0.5               |
| 20°C        | 0.8               |
| 30°C        | 1.2               |

The athlete can override with a known sweat rate. Total fluid target = sweat rate × event duration.

### Gel Recommendations

Gels are sized at real-world values: **30g** and **50g** carbohydrate portions. The model finds the combination of gels that meets the carb target with the fewest total gels and minimal overshoot.

### Fueling Windows

Carb fueling is skipped:

- In the first 20 minutes (startup grace — no benefit from carbs just consumed)
- In the last 20 minutes (tail cutoff — wouldn't take effect before finish)

This prevents unrealistic recommendations like "gel at minute 5 of a half marathon."

### Aid Station Tiers

| Tier                       | On-site kcal | On-site Carb (g) |
| -------------------------- | ------------ | ---------------- |
| Water only                 | 0            | 0                |
| Standard (race snacks)     | 150          | 30               |
| Full service (or drop bag) | 500          | 80               |

The model accounts for both carried fuel (gels between stations) and on-site fuel (consumed at stations).

---

## Grade-Adjusted Pace (GAP)

**Module**: `race_planners/grade.py`, `race_planners/pacing.py`

GAP uses a polynomial to convert uphill/downhill grades into pace equivalents. The core polynomial adjusts flat running pace for gradient:

- Uphill grades slow pace progressively
- Moderate downhill (3-5%) slightly speeds pace
- Steep downhill (>10%) starts slowing pace again due to braking and impact

This allows the planner to compare effort across hilly courses as if they were flat, which is essential for deriving pace targets from road-based threshold data.

# Unified Event Planner — User Guide

## Getting Started

The Unified Event Planner is a Streamlit application that helps runners plan pacing, fueling, and timing for road and trail events.

### Launching the app

```bash
uv run streamlit run semi-marathon-finistere/app.py
```

The app opens in your browser at `http://localhost:8501`.

## How to Use

### 1. Select an Event

Choose a curated event from the sidebar dropdown. The planner automatically loads the correct course, pacing model, and terrain settings. Available events include:

- **Semi-Marathon du Finistere** — road half marathon
- **Marathon des Etoiles de la Baie** — road marathon
- **Grand Raid du Finistere 56/92/166** — trail ultras
- **Trail de l'Odet Ultra** — trail ultra

### 2. Choose Your Target Mode

- **Finish Time**: Enter a goal finish time. The planner derives all pacing from this target.
- **Effort Anchor**: Enter your known training paces. The planner builds a plan from your anchors.

Both modes work for all events. Road events default to Finish Time. Trail events default to Effort Anchor.

### 3. Adjust Strategy Controls

**Road events** show:

- **Race Intent**: How aggressively you want to race (Best Effort → Easy/Durable)
- **Split Bias**: Hold back early (negative) or front-load (positive)
- **Aid Stop Time**: Seconds per aid station for water or brief walking

**Trail/ultra events** show:

- **Climb-to-Hike Threshold**: Grade percentage where hiking becomes more efficient
- **Descent Caution**: How conservatively you descend technical terrain
- **Fade Profile**: How much pace deteriorates across the day (Stable → Blow-Up Risk)
- **Aid Stop Time**: Minutes per aid station

### 4. Set Weather Conditions

Enter the expected peak temperature. The planner models a diurnal temperature curve from the event start time and applies a pace penalty when it's hot.

### 5. Configure Athlete Profile (Optional)

The Athlete Profile section lets you enter:

- **LT1/LT2 HR and pace**: Your lactate threshold heart rates and road paces
- **Road capability**: Your best-likely half marathon or marathon times
- **Predictor values**: External estimates from COROS, Strava, or Intervals.icu
- **Trail adjustments**: How much slower flat and technical trail are for you vs road
- **Body mass**: Used for calorie and fueling calculations
- **Durability/Heat/Hill tolerance**: Universal factors affecting performance across all events

The profile is saved in plan JSON exports and used to seed smarter defaults.

### 6. Advanced Trail Controls (Trail events only)

- **Effort Policy**: Conservative, Steady, or Aggressive pacing attitude
- **HR Guardrail**: Optional derived heart-rate ceiling based on your LT1/LT2 profile

### 7. Calculate and Read Results

Click **Calculate Plan** to generate:

- **Summary**: Distance, elapsed time, moving time, rest time, average pace
- **Course Profile**: Elevation chart and pace-per-kilometer bar chart
- **Aid Stations**: Arrival/departure times at each station (wall clock and elapsed)
- **Sections**: Terrain-aware pacing segments with elevation gain/loss
- **Fueling**: Per-block calorie, carbohydrate, and fluid targets with gel recommendations
- **Splits**: Pacing by distance blocks (1/2/5/10 km depending on event length)

### 8. Save and Reload Plans

- Click **Download plan JSON** to save your plan
- Use **Load Saved Plan** in the sidebar to reload a previously saved plan
- Athlete profile is included in plan JSON exports

## Language Toggle

Use the **Language** radio at the top of the sidebar to switch between English and French. All labels, help text, and messages translate instantly.

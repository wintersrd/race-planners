# Race Prediction Modeler — User Guide

## What this tool does

The Race Prediction Modeler helps you estimate pacing, finish time, aid-station timing, fueling, and split structure for curated road and trail events.

It is not just a split calculator. It combines:

- event course data
- athlete profile reference data
- weather assumptions
- event-specific pacing models
- fueling and hydration logic

## Launching the app

```bash
uv run streamlit run semi-marathon-finistere/app.py
```

The app opens in your browser at `http://localhost:8501`.

## Recommended workflow

### 1. Choose your language

Use the **Language** toggle at the top of the sidebar.

### 2. Read the guide first

Near the top of the sidebar you will find the **User Guide**.

It explains how to operate the planner. The separate **How the Models Work** section is only there if you want deeper technical background.

### 3. Fill in the Athlete Profile before planning seriously

The **Athlete Profile** section is one of the most important parts of the tool.

Add the reference data you actually know, especially:

- **LT1 HR** and **LT1 road pace**
- **LT2 HR** and **LT2 road pace**
- **best-likely road results** for half marathon and/or marathon if you know them
- **predictor values** from COROS, Strava, Intervals.icu, or a coach if you trust them
- **flat trail slowdown** and **technical trail slowdown** if you know how trail affects your pace
- **body mass** for fueling calculations
- **durability**, **heat tolerance**, and **hill tolerance** to reflect how you tend to hold up across long or difficult events

### 4. Apply the profile to the selected event

After editing the profile, click **Apply Profile Defaults To This Event**.

This matters because the planner does **not** automatically rewrite the current event configuration every time you type into the profile. The button tells the app to:

- reseed the current event defaults from the profile
- update things like road anchors, trail anchors, fade presets, and HR guardrail availability
- clear the stale plan result so the next calculation reflects the updated assumptions

If you skip this step, the current event may still be using older defaults.

### 5. Select an event

Choose a curated event from the sidebar dropdown. The planner automatically loads the correct course, terrain, and pacing model.

Current curated events include:

- **Semi-Marathon du Finistere** — road half marathon
- **Marathon des Etoiles de la Baie** — road marathon
- **Grand Raid du Finistere 56 / 92 / 166** — trail ultras
- **Trail de l'Odet Ultra** — trail ultra

### 6. Choose target mode

- **Finish Time**: you give the target finish time and the planner derives the pacing structure
- **Effort Anchor**: you give known paces and the planner builds the event from those anchors

Road events usually make the most sense in **Finish Time** mode. Trail and ultra events often make sense in **Effort Anchor** mode.

## Event-specific controls

### Road events

Road events show a simpler control set:

- **Race Intent**
- **Split Bias**
- **Aid Stop Time (sec)**
- **Peak Temperature**

These events use a simpler pacing setup than trail and ultra events.

### Trail / ultra events

Trail and ultra events expose more terrain-specific controls:

- **Climb-to-Hike Threshold**
- **Descent Caution**
- **Fade Profile**
- **Aid Stop Time (min)**
- **Peak Temperature**
- optional **Advanced Trail Controls** such as **Effort Policy** and **Use Derived HR Guardrail**

## What the key controls mean

### Race Intent

For road races, this shifts your target away from the event-adjusted best-likely estimate.

- **Best Effort**: race close to your adjusted capability ceiling
- **Strong**: serious effort with a little buffer
- **Controlled**: race firmly but sensibly
- **Easy / Durable**: finish well inside your red zone

### Split Bias

Road-only control.

- negative values: save a little early, finish stronger
- zero: closer to even pacing
- positive values: front-load effort earlier

### Fade Profile

Trail/ultra-only control.

This controls how much pace is expected to deteriorate through the day.

- **Stable**: low fade
- **Late Fade**: mostly okay early, more collapse late
- **Progressive Fade**: gradual degradation throughout
- **Blow-Up Risk**: aggressive fade assumptions

### Effort Policy

Trail advanced control.

This influences pacing attitude and interacts with the derived HR guardrail logic.

- **Conservative**
- **Steady**
- **Aggressive**

### Derived HR Guardrail

Trail advanced control.

Only useful if you entered enough athlete profile data for the planner to estimate HR meaningfully. The app will tell you if LT data is missing.

### Peak Temperature

The planner uses the expected peak temperature and the event start time to estimate a diurnal heat curve across the race.

## Reading the results

### Summary

Shows:

- distance
- elapsed time
- moving time
- rest time
- average pace

### Course Profile

Shows:

- elevation profile with terrain shading
- pace profile with color-coded pace bands
- cumulative time curve

### Aid Stations

Shows:

- arrival / departure elapsed time
- arrival / departure wall clock time
- split time and pace to each station
- aid station tier / expected station experience

### Sections

Shows terrain-aware sections with:

- section start/end
- elevation gain and loss
- section pace and duration

### Fueling

Shows:

- total calories
- carb target
- fluid target
- per-block carry suggestions
- aid station contribution
- cumulative carb deficit

### Splits

Shows pacing by configurable block size (1 / 2 / 5 / 10 km depending on event length).

### Split Analysis

Shows:

- first-half vs second-half comparison
- terrain breakdown by distance and time

## Saving and reloading

### Plan JSON

Use **Download plan JSON** to save the current plan. Reload it later from **Load Saved Plan**.

### Athlete Profile JSON

Use **Download Athlete Profile JSON** to save your profile separately. This is useful if you want to reuse your profile across events or machines.

## Practical advice

- Enter athlete profile data first if you have it
- Click **Apply Profile Defaults To This Event** after changing it
- Use event-specific controls rather than trying to force the model into another race type's mental model
- For ultras, think in terms of **terrain + fade + heat + fueling**, not only target split pace

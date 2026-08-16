# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A bilingual (English/French) Streamlit web application for race pacing, fueling, and timing strategy. Supports road events (half marathon, marathon) and trail/ultra events via a unified event-first planner.

## Commands

```bash
# Install dependencies (from repository root)
uv sync

# Run the app (from repository root)
uv run streamlit run semi-marathon-finistere/app.py

# Syntax check
uv run python -m py_compile semi-marathon-finistere/app.py

# Export Streamlit deployment requirements
scripts/export_streamlit_requirements.sh
```

## Architecture

The entry point is `semi-marathon-finistere/app.py`, a thin wrapper that delegates to the unified planner in `race_planners/streamlit_general.py`.

### Core Modules

- `race_planners/streamlit_general.py` — Streamlit UI orchestration
- `race_planners/planner.py` — Main plan calculation engine
- `race_planners/i18n.py` — Bilingual translation infrastructure (`TRANSLATIONS` dict + `t()` function)
- `race_planners/road_capability.py` — Road best-likely solver and feasibility classifiers
- `race_planners/weather.py` — Heat penalty and diurnal temperature model
- `race_planners/fatigue.py` — Fatigue, fade, and durability multipliers
- `race_planners/guardrails.py` — HR guardrail and effort policy logic
- `race_planners/fueling.py` — Energy expenditure, carb/hydration, and fueling plan
- `race_planners/segments.py` — Aid-aware segment builder
- `race_planners/profile.py` — Athlete profile defaults and preset definitions
- `race_planners/splits.py` — Course overview and split aggregation
- `race_planners/formatting.py` — Pace, duration, and clock formatting (locale-aware)
- `race_planners/plots.py` — Matplotlib chart helpers
- `race_planners/grade.py` — GPX parsing, elevation, and GAP calculations
- `race_planners/models.py` — Dataclasses (Course, PacingConfig, PlanResult, etc.)
- `race_planners/event_catalog.py` — Curated event definitions
- `race_planners/course_library.py` — Course loading from GPX files
- `race_planners/plan_io.py` — Plan JSON export/import

### Internationalization

- `race_planners/i18n.py` contains the `TRANSLATIONS` dict with all UI strings keyed by dot-notation identifiers
- `t(key, locale, **kwargs)` retrieves translated strings with named-placeholder interpolation
- The domain layer (planner, fueling, road_capability) emits i18n **keys** (not translated text)
- The UI boundary translates keys at render time using `t()`
- Assumption/warning keys may include inline kwargs: `"assumption.weather_heat|temp=30"`

### Key Data Flow

1. User selects a curated event from the sidebar
2. Event catalog resolves the course, model, and defaults
3. GPX is parsed into trackpoints with elevation and distance
4. User configures target mode (finish time or effort anchor) and strategy controls
5. `calculate_plan()` runs the pacing simulation with fatigue, fade, weather, and guardrail multipliers
6. Results render across tabs: Summary, Course Profile, Aid Stations, Sections, Fueling, Splits

### Locale

- `st.session_state["general_locale"]` stores the active locale (`"en"` or `"fr"`)
- Set via a sidebar radio at the top of `render_general_planner()`
- Clock formatting switches between 12-hour (English) and 24-hour (French) via `format_clock_time(time, event, locale)`

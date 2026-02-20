# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A bilingual (English/French) Streamlit web application for race pacing strategy. Uses Grade-Adjusted Pace (GAP) to calculate realistic pace targets for the Semi-Marathon du Finistère half marathon (21.1 km, ~153m elevation gain).

## Commands

```bash
# Install dependencies
pip install -r requirements-streamlit.txt

# Run the app
streamlit run app.py

# Syntax check
python -m py_compile app.py
```

## Architecture

Single-file Streamlit app (`app.py`) with these key sections:

### Data Layer
- **GPX Parsing**: `parse_gpx()` reads course data from `WR-GPX-Semi-marathon-du-Finistere.gpx`
- **TrackPoint dataclass**: Stores lat, lon, elevation, distance_from_start, grade_percent
- **Caching**: `@st.cache_data` on `load_gpx_data()` for performance

### Core Algorithms
- **GAP Factor**: `gap_factor(grade_percent)` - polynomial formula for grade-adjusted pacing
- **Elevation Segments**: `detect_elevation_segments()` - state machine with rolling window to classify climbs/descents/flats
- **Pacing Calculator**: `calculate_pacing()` - distributes target time across kilometers with GAP adjustment

### UI Structure (5 tabs)
1. Summary - strategy overview, split analysis, pace chart
2. Course Sections - predefined sections with pacing tips (COURSE_SECTIONS constant)
3. Elevation Segments - algorithm-detected segments with configurable thresholds
4. Tables - rest stops and kilometer splits
5. Print/Export - pocket card and wristband downloads

### Internationalization
- `TRANSLATIONS` dict contains all UI strings in English ('en') and French ('fr')
- `t(key, lang)` function retrieves translated strings
- All display text must go through translations

## Key Constants
- `REST_STOPS = [5.3, 9.1, 14.5]` - water station distances in km
- `COURSE_SECTIONS` - predefined course segments with strategy tips
- `SMOOTHING_WINDOW = 5` - elevation smoothing factor

## Data Flow
1. GPX file loaded once (cached)
2. User inputs: target time/pace, power fade, rest duration, segment thresholds
3. `calculate_pacing()` generates km_splits and rest_stops data
4. `detect_elevation_segments()` creates elevation-based segments
5. Tabs display different views of the same pacing data

# race-planners

Race planning applications and tools, centered on a single Streamlit-based
unified event planner.

## Python and environment

- Python: `3.13`
- Dependency and environment manager: `uv`
- Virtual environment: `.venv` at repository root

Setup:

```bash
uv sync
uv run pre-commit install
```

## Run the app

Run Streamlit from repository root so path behavior matches Streamlit Community Cloud:

```bash
uv run streamlit run race_planners/app.py
```

Do not start the app with `uv run race_planners/app.py` as a normal Python script. This repository's app is a Streamlit entrypoint, not a plain CLI program.

If you do run the script directly, it now re-launches itself through Streamlit, but the supported command is still the `streamlit run` form above.

The active app now presents one event-first planning flow:

- curated events currently include Finistere half marathon plus GRF56, GRF92,
  and GRF166
- race models currently include `half_marathon`, `road_marathon`,
  `fire_road_ultra`, and `technical_trail_ultra`
- input modes supported across the unified planner are `finish_time` and
  `effort_anchor`
- plan persistence is available through JSON download/reload

## Quality commands

```bash
uv run ruff check .
uv run ruff format .
uv run mypy tests
uv run pytest
uv run pre-commit run --all-files
```

## Streamlit deployment dependencies

`race_planners/requirements.txt` is generated from the `uv` project state.

```bash
scripts/export_streamlit_requirements.sh
scripts/check_streamlit_requirements.sh
```

Do not hand-edit `race_planners/requirements.txt`.

## Repository conventions for courses and plans

- Curated GPX assets are shipped in `race_planners/data/courses/`
- Plan files are JSON exports from the app and reference curated GPX basenames
- A plan can be reloaded only while its referenced curated course remains bundled

## Devcontainer

The repository includes `.devcontainer/devcontainer.json` with Python 3.13 and `uv` workflow parity.

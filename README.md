# race-planners

Race planning applications and tools, centered on a Streamlit app with two planner modes:

- `Legacy Half Marathon`: original Semi-Marathon du Finistere experience
- `General Planner (Beta)`: pluggable race models + local course library + JSON plan save/load

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
uv run streamlit run semi-marathon-finistere/app.py
```

Do not start the app with `uv run semi-marathon-finistere/app.py` as a normal Python script. This repository's app is a Streamlit entrypoint, not a plain CLI program.

If you do run the script directly, it now re-launches itself through Streamlit, but the supported command is still the `streamlit run` form above.

In-app mode switch:

- `Legacy Half Marathon` keeps existing race-specific behavior and output tabs.
- `General Planner (Beta)` supports:
  - race models: `half_marathon`, `road_marathon`, `fire_road_ultra`, `technical_trail_ultra`
  - input modes: `finish_time`, `effort_anchor`
  - course sources: built-in local routes + GPX uploads persisted to `courses/uploads/`
  - plan persistence: JSON download/reload (with explicit missing-GPX recovery message)

## Quality commands

```bash
uv run ruff check .
uv run ruff format .
uv run mypy tests
uv run pytest
uv run pre-commit run --all-files
```

## Streamlit deployment dependencies

`semi-marathon-finistere/requirements.txt` is generated from the `uv` project state.

```bash
scripts/export_streamlit_requirements.sh
scripts/check_streamlit_requirements.sh
```

Do not hand-edit `semi-marathon-finistere/requirements.txt`.

## Repository conventions for courses and plans

- Local route library root: `courses/`
- Uploaded GPX files are persisted in `courses/uploads/`
- Plan files are JSON exports from the app and can reference local GPX filenames
- If a referenced GPX is missing on reload, the app asks for re-upload

## Devcontainer

The repository includes `.devcontainer/devcontainer.json` with Python 3.13 and `uv` workflow parity.
